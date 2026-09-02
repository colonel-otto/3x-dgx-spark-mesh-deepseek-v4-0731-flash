#!/usr/bin/env python3
"""Apply the DeepSeek-V4 TP=3 padding patch to an installed vLLM tree.

TRANSACTIONAL: a preflight validates the vLLM version and EVERY anchor before
anything is written. A MISS/STALE/SKIP aborts with zero files modified; any
apply-phase failure rolls back this run's writes from the .tp3bak backups, and
a post-apply verify pass confirms every edit is actually installed.

Idempotent: each edit is guarded by a body-hash marker, so re-running is a
no-op. Every TP-related edit is a NO-OP at TP=1 and TP=2 --
pad_heads_for_groups(64,2,8) == 64 -- so applying this to a 2-node deployment
changes nothing. The ONE exception is the profiler edit (CUDA-only profiling,
an OOM workaround unrelated to TP); skip it with --no-profiler-fix.

The lenient pad-aware weight loaders in parameter.py are GATED on
dsv4_pad_active(): env VLLM_DSV4_TP_PAD truthy AND a DSV4 model has actually
been constructed in this process (mark_dsv4_model(), armed from the patched
DeepseekV4Attention __init__) AND the live TP world size is a validated pad
size (currently only 3). Every other model/TP configuration keeps stock
strict behaviour, so a genuinely corrupt checkpoint still crashes instead of
being zero-filled into fluent garbage. Unvalidated TP sizes that would need
padding (5, 6, 7, 9, ...) FAIL CLOSED with NotImplementedError.

Still OUTSIDE this script's reach (needs the running model -- validate_tp3.sh):
  * MoE expert pad columns being ZERO (expert loaders narrow() to param shape;
    the padded tail is whatever the allocation held unless the quant backend
    zeroes it)
  * attn_sink pad lanes being -inf on the padded 9th group's heads
  * TP=2 vs TP=3 logit equivalence -- the only proof that actually matters

Usage:
    python3 apply_tp3_patch.py [--vllm-root /usr/local/lib/python3.12/dist-packages/vllm]
                               [--check]            # report status, change nothing
                               [--revert]           # restore from .tp3bak
                               [--selftest]         # geometry unit checks only
                               [--no-profiler-fix]  # omit the CUDA-only profiler edit
                               [--expect-vllm-version 0.25.1]
                               [--force-version]    # proceed despite version mismatch
"""
from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import re
import shutil
import sys

MARK = "# --- DSV4_TP3_PAD ---"

# The vLLM release baked into the Anemll/B12X image this patch was written
# against and validated on (14/14 anchors). An anchor still existing on some
# other release only proves the TEXT survived, not that the surrounding
# semantics did -- so a different version is a hard stop without
# --force-version.
EXPECTED_VLLM_VERSION = "0.25.1"

PAD_MODULE = '''# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 TP padding (). Injected by apply_tp3_patch.py.

R2 group semantics: pad the GROUP COUNT to a multiple of tp_size and keep
heads_per_group CONSTANT. Groups 8 -> 9 at TP=3 => heads 64 -> 72, 3 groups and
24 heads per rank, still 8 heads per group.

*** WHY NOT R1 (replicate groups, shard heads within them) -- it was tried and
it is STRUCTURALLY WRONG. o_proj is a per-group BMM:

    fp8_einsum("bhr,hdr->bhd", o_fp8, wo_a.weight, z)

`r` is a HARD CONTRACT: r == heads_per_group * head_dim, and the checkpoint
fixes it at 4096 (= 8 heads/group * 512). Verified on the live weights:
layers.3.attn.wo_a.weight = [8192, 4096].

Stock vLLM shards the GROUPS and holds heads_per_group at 8, so r is invariant
at 4096 for TP = 1, 2, 4, 8 -- every divisor of n_groups. R1 changes
heads_per_group to 3 => r = 1536, and the kernel rejects it:

    einsum.hpp:164: m == m_ and n == n_ and k == k_

Padding wo_a to 4608 moves it FURTHER from both the activation (1536) and the
checkpoint (4096). The only padded head count restoring hpg=8 under R1 is 192,
3x the model. ⇒ Preserving heads_per_group is not a preference, it is the
kernel's contract. R2 preserves it; R1 cannot.

⛔ The padded 9th group is PURE PAD: its wo_a/wo_b slabs must be ZERO and its 8
heads must carry attn_sink = -inf (exp(0)=1 would add a phantom softmax
denominator term and attenuate the real heads).

NO-OP at TP=1 and TP=2 by construction: 8 % 1 == 8 % 2 == 0, so no padding.

FAIL CLOSED: only the TP sizes in _SUPPORTED_PAD_TP have been validated to
obey the kernel and quantization assumptions this padding relies on. Any other
tp_size that would actually need padding raises NotImplementedError instead of
mechanically producing a geometry nobody has ever tested.
"""
import os

# The ONLY TP sizes this padding has been validated for. TP=1/2/4/8 divide the
# 8 groups natively and never reach the padding math at all.
_SUPPORTED_PAD_TP = (3,)

_MAX_LOCAL_HEADS = 128


def dsv4_tp_pad_enabled():
    return os.environ.get("VLLM_DSV4_TP_PAD", "1") not in ("", "0", "false", "False")


_dsv4_model_constructed = False


def mark_dsv4_model():
    """Arm the loader gate. Called from the patched DeepseekV4Attention
    __init__, so the lenient paths stay DEAD in any process that never
    constructs a DSV4 model -- a non-DSV4 model served at TP=3 with the env
    flag on (its default) still gets stock strict loading."""
    global _dsv4_model_constructed
    _dsv4_model_constructed = True


def dsv4_pad_active():
    """True only when padding is enabled AND a DSV4 model was constructed in
    this process AND the live TP world size is one this patch has been
    validated for.

    Generic vLLM code (the parameter.py loaders) gates its lenient pad-aware
    paths on this, so every other deployment -- other models, other TP sizes,
    a corrupt checkpoint -- keeps stock STRICT behaviour and crashes loudly
    instead of zero-filling into fluent garbage.
    """
    if not dsv4_tp_pad_enabled():
        return False
    if not _dsv4_model_constructed:
        return False
    try:
        from vllm.distributed import get_tensor_model_parallel_world_size
        return get_tensor_model_parallel_world_size() in _SUPPORTED_PAD_TP
    except Exception:
        # Distributed not initialised (or import surface moved): no padding.
        return False


def pad_groups_for_tp(o_groups, tp_size):
    """Smallest group count >= o_groups divisible by tp_size.

    This is the whole technique: TP must divide n_groups, so pad n_groups.
    8 -> 9 at TP=3. A no-op whenever tp already divides o_groups (TP=1,2,4,8).
    Fails closed for any other tp_size that would need padding.
    """
    if tp_size <= 1 or not o_groups or o_groups <= 0:
        return o_groups
    if o_groups % tp_size == 0:
        return o_groups
    if tp_size not in _SUPPORTED_PAD_TP:
        raise NotImplementedError(
            f"DSV4 TP pad: tp_size={tp_size} would need group padding "
            f"(o_groups={o_groups}), but only TP in {_SUPPORTED_PAD_TP} has "
            f"been validated. TP=1/2/4/8 divide the groups natively; every "
            f"other size fails closed rather than booting an untested "
            f"kernel geometry."
        )
    return ((o_groups + tp_size - 1) // tp_size) * tp_size


def pad_heads_for_groups(num_heads, tp_size, o_groups):
    """Head count implied by the padded group count, at CONSTANT heads/group.

    heads_per_group is read from the REAL geometry (num_heads // o_groups) and
    preserved exactly -- that is the kernel contract. 64 heads / 8 groups = 8
    hpg; padded to 9 groups => 72 heads.
    """
    if tp_size <= 1:
        return num_heads
    if not o_groups or o_groups <= 0:
        # No group structure -> plain TP divisibility (same fail-closed rule).
        if num_heads % tp_size == 0:
            return num_heads
        if tp_size not in _SUPPORTED_PAD_TP:
            raise NotImplementedError(
                f"DSV4 TP pad: tp_size={tp_size} would need head padding "
                f"(num_heads={num_heads}), but only TP in "
                f"{_SUPPORTED_PAD_TP} has been validated."
            )
        return ((num_heads + tp_size - 1) // tp_size) * tp_size
    if num_heads % o_groups != 0:
        raise ValueError(
            f"DSV4 TP pad: num_heads={num_heads} not divisible by "
            f"o_groups={o_groups}; cannot infer heads_per_group"
        )
    hpg = num_heads // o_groups
    padded = pad_groups_for_tp(o_groups, tp_size) * hpg
    local = padded // tp_size
    if local > _MAX_LOCAL_HEADS:
        raise ValueError(
            f"DSV4 TP pad: local heads {local} exceeds max dispatchable "
            f"{_MAX_LOCAL_HEADS} (heads={num_heads} tp={tp_size} groups={o_groups})"
        )
    return padded


def local_heads_per_group(padded_heads, tp_size, o_groups):
    """Heads per group per rank -- INVARIANT under R2, and it must be.

    Returns the same hpg the checkpoint was built with, so
    r = hpg * head_dim matches wo_a's real input dim.
    """
    padded_groups = pad_groups_for_tp(o_groups, tp_size)
    if padded_groups <= 0:
        # Without this a zero/absent o_groups reaches `% padded_groups` below
        # and dies with a bare ZeroDivisionError instead of a diagnosis.
        raise ValueError(
            f"DSV4 TP pad: o_groups={o_groups} yields no group structure; "
            f"cannot compute heads per group"
        )
    if padded_groups % tp_size != 0:
        raise ValueError(
            f"DSV4 TP pad: padded_groups={padded_groups} not divisible by "
            f"tp={tp_size}"
        )
    if padded_heads % padded_groups != 0:
        raise ValueError(
            f"DSV4 TP pad: padded_heads={padded_heads} not a multiple of "
            f"padded_groups={padded_groups}"
        )
    return padded_heads // padded_groups


def real_groups(o_groups):
    """Number of groups backed by real checkpoint data (the pad tail is above)."""
    return o_groups
'''

# (file, anchor, replacement, description)
EDITS = [
    (
        "models/deepseek_v4/attention.py",
        """        self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size""",
        """        {MARK} head pad: 64 -> 72 at TP=3 (no-op at TP<=2)
        from vllm.models.deepseek_v4.dsv4_tp_pad import (
            dsv4_tp_pad_enabled, mark_dsv4_model, pad_heads_for_groups,
        )
        # Arms the loader gate: parameter.py's lenient pad-aware paths stay
        # dead in any process that never constructs a DSV4 model.
        mark_dsv4_model()
        self.n_heads_real = config.num_attention_heads
        if dsv4_tp_pad_enabled():
            self.n_heads = pad_heads_for_groups(
                config.num_attention_heads, tp_size,
                getattr(config, "o_groups", 0),
            )
        else:
            self.n_heads = config.num_attention_heads
        assert self.n_heads % tp_size == 0
        self.n_local_heads = self.n_heads // tp_size""",
        "attention.py: head pad + drop hard assert",
    ),
    (
        "models/deepseek_v4/attention.py",
        """        self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size""",
        """        {MARK} R2: PAD the group count so tp divides it; keep heads/group.
        # Stock code did `n_groups // tp_size` -> 8//3 == 2 SILENTLY (no assert),
        # dropping six of eight HCA groups and producing fluent garbage.
        #
        # ⛔ Do NOT replicate groups and shard heads within them (the old "R1").
        # o_proj is a per-group BMM whose `r` dim is a HARD CONTRACT:
        #     r == heads_per_group * head_dim == 8 * 512 == 4096
        # fixed by the checkpoint (wo_a = [8192, 4096]). R1 makes hpg 3 => r=1536
        # and the kernel rejects it (einsum.hpp:164 m==m_ and n==n_ and k==k_).
        # Padding wo_a to 4608 only moves it further away. ⇒ heads_per_group must
        # be PRESERVED, which is exactly what padding the GROUP count does.
        self.n_groups_real = config.o_groups
        from vllm.models.deepseek_v4.dsv4_tp_pad import (
            dsv4_tp_pad_enabled, local_heads_per_group, pad_groups_for_tp,
        )
        if dsv4_tp_pad_enabled():
            self.n_groups = pad_groups_for_tp(config.o_groups, tp_size)
        else:
            self.n_groups = config.o_groups
        self.n_local_groups = self.n_groups // tp_size
        self.heads_per_group_local = local_heads_per_group(
            self.n_heads, tp_size, self.n_groups_real
        )
        # The invariant that matters: hpg is UNCHANGED from the real geometry,
        # so `r` still matches the checkpoint. if/raise, NOT assert: this is
        # the R2 kernel contract, not a debug check, and it must survive
        # `python -O`.
        if self.heads_per_group_local != (
                self.n_heads_real // self.n_groups_real):
            raise RuntimeError(
                f"DSV4 R2 contract broken: hpg={self.heads_per_group_local} != "
                f"real hpg={self.n_heads_real // self.n_groups_real} -- the "
                f"o_proj BMM `r` dim would no longer match the checkpoint"
            )
        if self.n_local_heads != self.n_local_groups * self.heads_per_group_local:
            raise RuntimeError(
                f"DSV4 R2 invariant broken: n_local_heads={self.n_local_heads} "
                f"!= n_local_groups={self.n_local_groups} * "
                f"hpg={self.heads_per_group_local}"
            )""",
        "attention.py: R2 group padding + BMM contract assert",
    ),
    (
        # The config layer validates head divisibility LONG before any model
        # class is constructed -- SpeculativeConfig hits it first. Patching only
        # the model code is not enough: vLLM dies in arg parsing with
        # "Total number of attention heads (64) must be divisible by tensor
        # parallel size (3)". This is the site that actually blocks the boot.
        "config/model.py",
        """        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            raise ValueError(
                f"Total number of attention heads ({total_num_attention_heads})"
                " must be divisible by tensor parallel size "
                f"({tensor_parallel_size})."
            )""",
        """        {MARK} allow a padded head count for DSv4 (heads 64->72 at TP=3).
        # Reached only when heads do NOT divide tp; pad_heads_for_groups fails
        # closed (NotImplementedError) for any unvalidated tp_size, which lands
        # in the except below and re-raises the stock divisibility error.
        total_num_attention_heads = self.model_arch_config.total_num_attention_heads
        tensor_parallel_size = parallel_config.tensor_parallel_size
        if total_num_attention_heads % tensor_parallel_size != 0:
            _padded_ok = False
            try:
                from vllm.models.deepseek_v4.dsv4_tp_pad import (
                    dsv4_tp_pad_enabled, pad_heads_for_groups,
                )
                _og = getattr(self.hf_config, "o_groups", 0)
                if dsv4_tp_pad_enabled() and _og:
                    _p = pad_heads_for_groups(
                        total_num_attention_heads, tensor_parallel_size, _og
                    )
                    _padded_ok = (_p % tensor_parallel_size == 0)
                    if _padded_ok:
                        import logging
                        logging.getLogger(__name__).info(
                            "DSv4 TP pad: heads %d -> %d for TP=%d (o_groups=%d)",
                            total_num_attention_heads, _p,
                            tensor_parallel_size, _og,
                        )
            except Exception as _e:
                # Never swallow silently: a bug in the pad helper must not
                # masquerade as an ordinary divisibility error.
                import logging
                logging.getLogger(__name__).warning(
                    "DSv4 TP pad probe failed; falling back to the stock "
                    "divisibility error: %r", _e,
                )
                _padded_ok = False
            if not _padded_ok:
                raise ValueError(
                    f"Total number of attention heads "
                    f"({total_num_attention_heads})"
                    " must be divisible by tensor parallel size "
                    f"({tensor_parallel_size})."
                )""",
        "config/model.py: allow padded head count (THE boot blocker)",
    ),
    (
        # The MERGED-column loader (fused gate/up). A third loader, distinct from
        # load_column_parallel_weight -- patching that one does NOT cover this path:
        #   RuntimeError: start (12) + length (6) exceeds dimension size (16)
        # It narrows the CHECKPOINT by tp_rank*shard_size, which overruns on the
        # last rank once the destination is padded. Same clamped-copy remedy.
        "model_executor/parameter.py",
        """        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):""",
        """        param_data = param_data.narrow(self.output_dim, shard_offset, shard_size)
        {MARK} pad-aware merged-column load: clamp to what really exists.
        # GATED on dsv4_pad_active(): env on AND live TP size is a validated
        # pad size. Everything else keeps stock strict behaviour, so a corrupt
        # checkpoint for some unrelated model still crashes instead of being
        # zero-filled into fluent garbage.
        try:
            from vllm.models.deepseek_v4.dsv4_tp_pad import dsv4_pad_active
            _pad_active = dsv4_pad_active()
        except Exception:
            _pad_active = False
        _real = loaded_weight.shape[self.output_dim]
        _start = self.tp_rank * shard_size
        _avail = max(0, min(_real - _start, shard_size))
        if _pad_active and _avail != shard_size:
            param_data.zero_()
            if _avail > 0:
                param_data.narrow(self.output_dim, 0, _avail).copy_(
                    loaded_weight.narrow(self.output_dim, _start, _avail)
                )
            return
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)

    def load_qkv_weight(self, loaded_weight: torch.Tensor, **kwargs):""",
        "parameter.py: pad-aware load_merged_column_weight",
    ),
    (
        # Row-parallel mirror of the column loader: shards the INPUT dim instead.
        # Same clamped-copy + zero-fill, same self-disabling property.
        "model_executor/parameter.py",
        """    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.input_dim]
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)""",
        """    def load_row_parallel_weight(self, loaded_weight: torch.Tensor):
        {MARK} pad-aware: copy the real intersection, zero-fill any padded tail.
        # Handles both a short SHARDED dim and any widened NON-sharded dim.
        # GATED on dsv4_pad_active() -- see load_column_parallel_weight.
        try:
            from vllm.models.deepseek_v4.dsv4_tp_pad import dsv4_pad_active
            _pad_active = dsv4_pad_active()
        except Exception:
            _pad_active = False
        shard_size = self.data.shape[self.input_dim]
        _real = loaded_weight.shape[self.input_dim]
        # ⚠️ See the column loader: a REPLICATED param still carries the real
        # tp_rank; offsetting by it zeroes the whole tensor on ranks 1..N-1.
        _replicated = (shard_size == _real)
        _start = 0 if _replicated else self.tp_rank * shard_size
        _avail = max(0, min(_real - _start, shard_size))
        _id_short = (_avail != shard_size or _start >= _real)
        _other_pad = any(
            self.data.shape[_d] != loaded_weight.shape[_d]
            for _d in range(loaded_weight.dim())
            if _d != self.input_dim
        )
        if _pad_active and _replicated and not _other_pad:
            # Fully replicated and shape-exact: load whole on every rank; the
            # stock fallthrough would narrow past the end on ranks 1..N-1.
            self.data.copy_(loaded_weight)
            return
        if _pad_active and (_id_short or _other_pad):
            self.data.zero_()
            if _avail > 0:
                _src = loaded_weight.narrow(self.input_dim, _start, _avail)
                _dst = self.data.narrow(self.input_dim, 0, _avail)
                for _d in range(_src.dim()):
                    if _d == self.input_dim:
                        continue
                    _n = min(_src.shape[_d], _dst.shape[_d])
                    _src = _src.narrow(_d, 0, _n)
                    _dst = _dst.narrow(_d, 0, _n)
                _dst.copy_(_src)
            return
        loaded_weight = loaded_weight.narrow(
            self.input_dim, self.tp_rank * shard_size, shard_size
        )

        if len(loaded_weight.shape) == 0:
            loaded_weight = loaded_weight.reshape(1)

        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)""",
        "parameter.py: pad-aware load_row_parallel_weight",
    ),
    (
        # THE PAD LOADERS. Every site above changes SHAPES; this is what makes a
        # padded parameter loadable at all.
        #
        # Stock: shard_size = padded_local; narrow(dim, tp_rank*shard_size, shard_size)
        # assumes tp_rank*shard_size + shard_size <= checkpoint size. Under padding it
        # does not (e.g. wq_b padded local 12288 rows x 3 ranks = 36864 > 32768 real),
        # so the bare `assert self.data.shape == loaded_weight.shape` fires.
        #
        # Fix: copy only the intersection with what really exists, zero the rest.
        #   dest.zero_(); dest[:n].copy_(src[start:start+n])   where n may be 0
        # A zero pad lane is exactly inert: zero weights -> zero activation -> zero
        # contribution through the all-reduce.
        #
        # ⚠️ attn_sink is NOT loaded through here (it is a bare nn.Parameter with its
        # own loaders, sites 8-10) -- which is what keeps its pad lanes at -inf. If a
        # future change routes it through this path, the zero_() below would silently
        # enable phantom sinks.
        #
        # Self-disabling twice over: the lenient path needs BOTH a shape mismatch
        # AND dsv4_pad_active() (env on + live TP size validated for padding).
        "model_executor/parameter.py",
        """    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        shard_size = self.data.shape[self.output_dim]
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)""",
        """    def load_column_parallel_weight(self, loaded_weight: torch.Tensor):
        {MARK} pad-aware: copy the real intersection, zero-fill any padded tail.
        # Padding widens TWO kinds of dim, so both must be handled:
        #   - the SHARDED dim (output_dim) -- e.g. wq_b, where tp_rank*shard may
        #     run past the end of the real checkpoint tensor;
        #   - a NON-sharded dim -- e.g. wo_a's input, which widens 4096->4608 when
        #     heads pad 64->72 and which disable_tp leaves unsharded entirely.
        # A dim-by-dim clamped copy covers both without special-casing either.
        # GATED on dsv4_pad_active(): only the validated DSV4 TP=3 deployment
        # gets the lenient path; everything else keeps stock strict behaviour.
        try:
            from vllm.models.deepseek_v4.dsv4_tp_pad import dsv4_pad_active
            _pad_active = dsv4_pad_active()
        except Exception:
            _pad_active = False
        shard_size = self.data.shape[self.output_dim]
        _real = loaded_weight.shape[self.output_dim]
        # ⚠️ A REPLICATED param (disable_tp=True, e.g. wo_a under R1) still carries
        # the real tp_rank. Offsetting by it would push ranks 1..N-1 past the end of
        # the checkpoint and zero the WHOLE tensor -- silently, on 2 of 3 ranks.
        # If the param is not actually sharded on this dim, every rank loads it whole.
        _replicated = (shard_size == _real)
        _start = 0 if _replicated else self.tp_rank * shard_size
        _avail = max(0, min(_real - _start, shard_size))
        _od_short = (_avail != shard_size or _start >= _real)
        _other_pad = any(
            self.data.shape[_d] != loaded_weight.shape[_d]
            for _d in range(loaded_weight.dim())
            if _d != self.output_dim
        )
        if _pad_active and _replicated and not _other_pad:
            # Fully replicated and shape-exact: every rank loads it whole.
            # Without this short-circuit the stock fallthrough below would
            # offset by tp_rank*shard_size and narrow out of bounds on ranks
            # 1..N-1 -- the _start=0 correction above only reaches the padded
            # branch, not the fallthrough.
            self.data.copy_(loaded_weight)
            return
        if _pad_active and (_od_short or _other_pad):
            self.data.zero_()
            if _avail > 0:
                _src = loaded_weight.narrow(self.output_dim, _start, _avail)
                _dst = self.data.narrow(self.output_dim, 0, _avail)
                # Clamp every remaining dim to the smaller of the two.
                for _d in range(_src.dim()):
                    if _d == self.output_dim:
                        continue
                    _n = min(_src.shape[_d], _dst.shape[_d])
                    _src = _src.narrow(_d, 0, _n)
                    _dst = _dst.narrow(_d, 0, _n)
                _dst.copy_(_src)
            return
        loaded_weight = loaded_weight.narrow(
            self.output_dim, self.tp_rank * shard_size, shard_size
        )
        assert self.data.shape == loaded_weight.shape
        self.data.copy_(loaded_weight)""",
        "parameter.py: pad-aware load_column_parallel_weight",
    ),
    (
        # moe_intermediate_size=2048 does not divide 3:
        #   fused_moe/config.py:1314  assert self.intermediate_size % tp_size == 0
        #
        # This image has NO pad support (no intermediate_size_real, no pad_moe_*) --
        # that was the MiaAI image, not this one. But it DOES carry
        # `intermediate_size_per_partition_unpadded`, which downstream quant/backend
        # code reads to recover the true size. So pad up to lcm(tp, 64) and record
        # the real per-partition size there:
        #   lcm(3,64) = 192 -> 2048 pads to 2112; 2112/3 = 704, and 704 % 64 == 0
        #   so NVFP4's 64-alignment on the per-rank K axis still holds.
        #
        # The pad columns must be ZERO for correctness. Expert weight loaders
        # narrow() to param.shape, so the padded tail is whatever the parameter was
        # allocated with -- verify it is zero-initialised, or a garbage tail leaks
        # into the SwiGLU product. THIS IS THE UNVERIFIED PART: see validate_tp3.sh.
        "model_executor/layers/fused_moe/config.py",
        """        tp_size = self.moe_parallel_config.tp_size
        assert self.intermediate_size % tp_size == 0
        self.intermediate_size_per_partition = self.intermediate_size // tp_size""",
        """        tp_size = self.moe_parallel_config.tp_size
        {MARK} pad intermediate to lcm(tp, 64) when it does not divide tp.
        # Gated the same way as the loaders: env on + validated pad TP size.
        # Anything else falls through to the stock assert below.
        if tp_size > 1 and self.intermediate_size % tp_size != 0:
            from vllm.models.deepseek_v4.dsv4_tp_pad import (
                _SUPPORTED_PAD_TP, dsv4_tp_pad_enabled,
            )
            if dsv4_tp_pad_enabled() and tp_size in _SUPPORTED_PAD_TP:
                import math as _math
                _lcm = _math.lcm(tp_size, 64)
                _padded = ((self.intermediate_size + _lcm - 1) // _lcm) * _lcm
                if self.intermediate_size_per_partition_unpadded is None:
                    # ⚠️ 2048//3 == 682 is a UNIFORM value for a NON-uniform
                    # reality: real per-rank widths are 704/704/640 (rank 2
                    # carries the 64 pad columns). If downstream consumes this
                    # as "my real width", ranks 0/1 would DROP real columns
                    # 682-703. validate_tp3.sh must confirm how this field is
                    # consumed AND that the pad columns load as zero.
                    self.intermediate_size_per_partition_unpadded = (
                        self.intermediate_size // tp_size
                    )
                self.intermediate_size = _padded
        assert self.intermediate_size % tp_size == 0
        self.intermediate_size_per_partition = self.intermediate_size // tp_size""",
        "fused_moe/config.py: pad MoE intermediate to lcm(tp,64)",
    ),
    (
        # n_routed_experts=256 does not divide 3, but with EP OFF this assert
        # guards values that are DEAD on the FusedMoE path:
        #   - fused_moe/layer.py:250 gates the real expert-count check on `use_ep`
        #   - experts_start_idx / n_local_physical_experts are consumed only by
        #     MegaMoE (model.py:255-257), which is SM100-gated and not our path
        #   - experts are TP-sharded on the INTERMEDIATE dim (2048), handled elsewhere
        #
        # DO NOT follow the assert's own advice: num_redundant_experts=2 (-> 258)
        # hits a DIFFERENT assert at fused_moe/layer.py:257, "Redundant experts are
        # only supported with EPLB." The error message's remedy is not the diagnosis.
        #
        # So: scope it to the paths that actually consume the values.
        "models/deepseek_v4/nvidia/model.py",
        """        assert self.n_physical_experts % self.tp_size == 0, (
            f"n_physical_experts={self.n_physical_experts} must be divisible by "
            f"tp_size={self.tp_size}. Adjust num_redundant_experts."
        )""",
        """        {MARK} scope to MegaMoE / EP -- dead values on the FusedMoE path.
        _experts_shard_by_count = bool(
            getattr(self, "use_mega_moe", False)
            or getattr(parallel_config, "enable_expert_parallel", False)
        )
        if _experts_shard_by_count:
            assert self.n_physical_experts % self.tp_size == 0, (
                f"n_physical_experts={self.n_physical_experts} must be divisible "
                f"by tp_size={self.tp_size}. Adjust num_redundant_experts."
            )""",
        "model.py: scope the 256-expert assert to MegaMoE/EP",
    ),
    (
        # R1 REQUIRED wo_a/wo_b to stay UNSHARDED on the group axis. Setting
        # n_local_groups = n_groups (the R1 edit) is not enough on its own: the
        # LAYERS still shard. ColumnParallelLinear divides out-features by tp, and
        # n_groups*o_lora_rank = 8*1024 = 8192 is not divisible by 3:
        #   linear.py:441  self.output_size_per_partition = divide(output_size, tp)
        #   AssertionError: 8192 is not divisible by 3
        # wo_b is the mirror case (RowParallelLinear divides its INPUT dim, same 8192).
        #
        # disable_tp=True on both makes them replicated, exactly as V4 already does
        # for fused_wqa_wkv. That is what makes the all-reduce sum exact per-rank
        # partials -- the property the R1 test measured at rel 2.3e-06.
        "models/deepseek_v4/attention.py",
        """        self.wo_a = ColumnParallelLinear(
            self.n_heads * self.head_dim // self.n_groups,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )""",
        """        {MARK} R2: wo_a/wo_b shard NORMALLY on the padded group axis.
        # n_groups is already PADDED (8->9 at TP=3), so n_groups*o_lora_rank =
        # 9216 divides 3 cleanly -> 3072 per rank. No disable_tp needed; that was
        # an R1 workaround for 8192 % 3 != 0.
        #
        # ⭐ The input dim uses heads_per_group * head_dim, NOT n_heads*head_dim//
        # n_groups. They coincide only when nothing is padded. This is the o_proj
        # BMM `r` contract -- it MUST stay 8*512 == 4096 to match the checkpoint.
        self.wo_a = ColumnParallelLinear(
            self.heads_per_group_local * self.head_dim,
            self.n_groups * self.o_lora_rank,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_a",
        )
        self.wo_a.is_bmm = True
        self.wo_a.bmm_batch_size = self.n_local_groups
        self.wo_b = RowParallelLinear(
            self.n_groups * self.o_lora_rank,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            return_bias=False,
            prefix=f"{prefix}.wo_b",
        )""",
        "attention.py: wo_a/wo_b shard on the padded group axis (R2)",
    ),
    (
        # vocab_size=129280 is not divisible by 3. This image's padding_size
        # defaults to a plain 64, so pad_vocab_size(129280, 64) == 129280 and the
        # embedding shard dies with "129280 is not divisible by 3" AFTER all three
        # ranks have already joined the NCCL group -- which reads as a distributed
        # problem rather than a padding one.
        #
        # lcm(64, tp) keeps the existing 64-alignment AND makes it divide tp:
        #   lcm(64,3) = 192 -> pad_vocab_size(129280, 192) = 129408; 129408/3 = 43136.
        # A no-op whenever tp already divides padding_size (TP=1,2,4,8), and benign
        # for any model: extra vocab pad rows are stock vLLM behaviour, just wider.
        "model_executor/layers/vocab_parallel_embedding.py",
        """        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        self.padding_size = padding_size""",
        """        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings = num_embeddings
        {MARK} pad vocab to lcm(padding_size, tp) so it divides TP (no-op when tp | padding_size)
        import math as _math
        self.padding_size = _math.lcm(padding_size, self.tp_size)""",
        "vocab_parallel_embedding.py: pad vocab to lcm(padding_size, tp)",
    ),
    (
        "models/deepseek_v4/nvidia/model.py",
        """        n_head = self.config.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)""",
        """        {MARK} attn_sink window from the PADDED head count, clamped to real.
        # Stock read raw num_attention_heads -> 64//3 == 21 -> windows
        # [0:21][21:42][42:63]: head 63 never loads on ANY rank, silently.
        from vllm.models.deepseek_v4.dsv4_tp_pad import pad_heads_for_groups
        n_head = self.config.num_attention_heads
        _n_head_padded = pad_heads_for_groups(
            n_head, tp_size, getattr(self.config, "o_groups", 0)
        )
        n_local_head = _n_head_padded // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = min(head_rank_start + n_local_head, n_head)""",
        "model.py: attn_sink loader window",
    ),
    (
        # NOT a TP edit: a deployment-specific OOM workaround (the CPU profiler's
        # memory bloat). Kept here because the same image needs it, but it is the
        # one edit that is NOT a no-op at TP<=2 -- omit it with --no-profiler-fix.
        "v1/worker/gpu_worker.py",
        """                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )""",
        """                    {MARK} profile CUDA only to prevent CPU profiler memory bloat and OOM
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CUDA"],
                    )""",
        "gpu_worker.py: profile CUDA only to prevent CPU profiler memory bloat",
    ),
]

PROFILER_EDIT_FILE = "v1/worker/gpu_worker.py"

# The two other attn_sink loaders use the same idiom with a different receiver.
SINK_VARIANTS = [
    ("models/deepseek_v4/nvidia/mtp.py", "mtp.py: attn_sink loader window"),
    ("models/deepseek_v4/nvidia/dspark.py", "dspark.py: attn_sink loader window"),
]
# Two idioms occur in the wild: `head_start/head_end` (dspark.py) and
# `head_rank_start/head_rank_end` preceded by an `n_head = ...` line (mtp.py).
SINK_ANCHORS = [
    ("""        n_local_head = {recv}.num_attention_heads // tp_size
        head_start = n_local_head * tp_rank
        head_end = n_local_head * (tp_rank + 1)""", "head_start", "head_end"),
    ("""        n_head = {recv}.num_attention_heads
        n_local_head = n_head // tp_size
        head_rank_start = n_local_head * tp_rank
        head_rank_end = n_local_head * (tp_rank + 1)""",
     "head_rank_start", "head_rank_end"),
]


def _atomic_write(path: pathlib.Path, content: str) -> None:
    """Write via temp file + os.replace so a crash mid-write can never leave
    a truncated .py in the live tree (a truncated module would import-error
    every vLLM boot until manually repaired)."""
    tmp = path.with_name(path.name + ".tp3tmp")
    tmp.write_text(content, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def patch_file(path: pathlib.Path, anchor: str, repl: str, desc: str,
               check: bool, version: str | None = None,
               touched: list[pathlib.Path] | None = None) -> str:
    """Apply one edit. Idempotent and multi-occurrence safe.

    Traps this guards against, all found the hard way:
      1. A whole-file MARK check is wrong when one file carries several edits --
         it reports 'already patched' for edits that never applied. The guard
         must be per-edit, so we tag each replacement with its own marker.
      2. The same anchor text can occur in MORE THAN ONE CLASS (attention.py's
         anchor appears in both DeepseekV4Attention and the MTP attention
         subclass). Replacing only the first occurrence silently leaves the
         second unpatched; replacing all of them is what we actually want.
      3. the tag was derived from `desc` ALONE, so a change to a
         patch BODY kept the same tag. A container whose writable layer still
         held the OLD patch therefore reported 'already patched' and was skipped
         forever -- the fixed loader never landed, and the only symptom was a
         rank-2 shape assert. `docker compose` restarts (not recreates) a
         container, so that stale layer survived every stop/start cycle.
         The tag now carries a hash of the replacement body, so a changed body
         yields a DIFFERENT tag: a stale patch no longer looks current, and the
         mismatch is reported instead of silently skipped.
    """
    if not path.exists():
        return f"SKIP  {desc} (file not found: {path})"
    text = path.read_text(encoding="utf-8")
    # The version identifies WHICH revision of this edit is installed. Callers
    # whose `repl` is composed at call time (the attn_sink variants pick one of
    # several receiver expressions) MUST pass an explicit, stable `version` --
    # hashing their generated body would yield a different tag per run and make
    # an already-correct file look STALE.
    body_hash = version or hashlib.sha256(repl.encode()).hexdigest()[:12]
    tag = f"{MARK} [{desc}] v={body_hash}"
    if tag in text:
        return f"OK    {desc} (already patched)"
    # Same edit, DIFFERENT body => a stale patch is baked into this file.
    # Never silently skip it: the file cannot be re-anchored (the stock text is
    # gone), so this needs a pristine file -- i.e. `docker rm -f` the container.
    stale = f"{MARK} [{desc}]"
    if stale in text:
        return (f"STALE {desc} -- an OLDER version of this patch is present. "
                f"The anchor is consumed, so this CANNOT be re-applied in place. "
                f"Recreate the container (`docker rm -f`) to get a pristine file.")
    n = text.count(anchor)
    if n == 0:
        return f"MISS  {desc} -- anchor not found, NEEDS MANUAL REVIEW"
    if check:
        return f"TODO  {desc} ({n} site{'s' if n > 1 else ''} found)"
    bak = path.with_suffix(path.suffix + ".tp3bak")
    if not bak.exists():
        shutil.copy2(path, bak)
    if touched is not None:
        touched.append(path)
    _atomic_write(path, text.replace(anchor, repl.replace("{MARK}", tag)))
    return f"PATCH {desc} ({n} site{'s' if n > 1 else ''})"


def collect_results(root: pathlib.Path, edits: list, check: bool,
                    touched: list[pathlib.Path] | None = None):
    """Run every edit (EDITS + attn_sink variants). Returns (edit_results,
    sink_results) -- kept separate because a SKIP is fatal for EDITS (core
    vLLM files must exist) but tolerable for a sink variant file absent from
    this image."""
    edit_results = [
        patch_file(root / f, a, r, d, check, touched=touched)
        for f, a, r, d in edits
    ]
    sink_results = []
    for rel, desc in SINK_VARIANTS:
        p = root / rel
        done = False
        for tmpl, s_name, e_name in SINK_ANCHORS:
            for recv in ("self.config", "config", "self.model_config.hf_config"):
                anchor = tmpl.format(recv=recv)
                repl = (
                    "        {MARK} attn_sink window from PADDED head count.\n"
                    "        from vllm.models.deepseek_v4.dsv4_tp_pad import "
                    "pad_heads_for_groups\n"
                    f"        n_head = {recv}.num_attention_heads\n"
                    f"        _padded = pad_heads_for_groups(n_head, tp_size, "
                    f"getattr({recv}, 'o_groups', 0))\n"
                    "        n_local_head = _padded // tp_size\n"
                    f"        {s_name} = n_local_head * tp_rank\n"
                    f"        {e_name} = min({s_name} + n_local_head, n_head)"
                )
                # Stable version: this edit's body varies by receiver, so bump
                # this string by hand when the attn_sink logic itself changes.
                out = patch_file(p, anchor, repl, desc, check,
                                 version="sink1", touched=touched)
                if not out.startswith("MISS"):
                    sink_results.append(out)
                    done = True
                    break
            if done:
                break
        if not done:
            sink_results.append(
                f"MISS  {desc} -- anchor not found, NEEDS MANUAL REVIEW")
    return edit_results, sink_results


def detect_vllm_version(root: pathlib.Path) -> str | None:
    pat = re.compile(
        r'__version__(?:\s*:\s*\w+)?\s*=\s*["\']([^"\']+)["\']')
    for name in ("version.py", "_version.py", "__init__.py"):
        p = root / name
        if p.exists():
            m = pat.search(p.read_text(encoding="utf-8"))
            if m:
                return m.group(1)
    return None


def module_status(mod: pathlib.Path) -> str:
    """OK = installed and byte-identical to this script's PAD_MODULE;
    STALE = installed but different (an older script version wrote it);
    TODO = absent."""
    if not mod.exists():
        return "TODO"
    if hashlib.sha256(mod.read_bytes()).hexdigest() == \
            hashlib.sha256(PAD_MODULE.encode()).hexdigest():
        return "OK"
    return "STALE"


def geometry_selftest() -> list[str]:
    """Unit-check the geometry of the ACTUAL module we ship, by exec'ing
    PAD_MODULE. Pure python, runs on every invocation -- a patcher whose own
    math is wrong must never touch the tree."""
    import os
    ns: dict = {}
    exec(compile(PAD_MODULE, "<PAD_MODULE>", "exec"), ns)
    pg, ph, lh = (ns["pad_groups_for_tp"], ns["pad_heads_for_groups"],
                  ns["local_heads_per_group"])
    fails: list[str] = []

    def ok(name, cond):
        if not cond:
            fails.append(name)

    def raises(name, exc, fn):
        try:
            fn()
        except exc:
            return
        except Exception as e:  # noqa: BLE001
            fails.append(f"{name}: wrong exception {e!r}")
        else:
            fails.append(f"{name}: did not raise {exc.__name__}")

    # The R2 core: 8 groups -> 9 at TP=3, heads 64 -> 72, hpg invariant at 8.
    ok("groups 8@tp3 -> 9", pg(8, 3) == 9)
    ok("heads 64@tp3 -> 72", ph(64, 3, 8) == 72)
    ok("hpg invariant at 8", lh(72, 3, 8) == 8)
    ok("hpg == real hpg", ph(64, 3, 8) // pg(8, 3) == 64 // 8)
    # No-op at every native divisor.
    for tp in (1, 2, 4, 8):
        ok(f"groups no-op @tp{tp}", pg(8, tp) == 8)
        ok(f"heads no-op @tp{tp}", ph(64, tp, 8) == 64)
    # No-group path: divides -> untouched; would-pad at unvalidated tp -> raise.
    ok("no-group divides @tp2", ph(64, 2, 0) == 64)
    raises("no-group pad @tp5 fails closed", NotImplementedError,
           lambda: ph(64, 5, 0))
    # Fail closed on unvalidated TP sizes that would need padding.
    for tp in (5, 6, 7, 9):
        raises(f"groups pad @tp{tp} fails closed", NotImplementedError,
               lambda tp=tp: pg(8, tp))
    # Malformed geometry must raise, never guess.
    raises("heads%groups!=0 raises", ValueError, lambda: ph(60, 3, 8))
    raises("o_groups=0 hpg raises ValueError", ValueError,
           lambda: lh(72, 3, 0))
    # The loader gate must be dead until a DSV4 model construction arms it.
    ok("gate dead before model construction",
       ns["dsv4_pad_active"]() is False)
    # Env kill-switch.
    old = os.environ.get("VLLM_DSV4_TP_PAD")
    try:
        os.environ["VLLM_DSV4_TP_PAD"] = "0"
        ok("env gate off", ns["dsv4_tp_pad_enabled"]() is False)
        os.environ["VLLM_DSV4_TP_PAD"] = "1"
        ok("env gate on", ns["dsv4_tp_pad_enabled"]() is True)
    finally:
        if old is None:
            os.environ.pop("VLLM_DSV4_TP_PAD", None)
        else:
            os.environ["VLLM_DSV4_TP_PAD"] = old
    return fails


def rollback(root: pathlib.Path, touched: list[pathlib.Path],
             mod: pathlib.Path, prev_mod: str | None, wrote_mod: bool) -> None:
    for p in dict.fromkeys(touched):  # de-dupe, preserve order
        bak = p.with_suffix(p.suffix + ".tp3bak")
        if bak.exists():
            shutil.copy2(bak, p)
            print(f"ROLLBACK {p.relative_to(root)}", file=sys.stderr)
    if wrote_mod:
        if prev_mod is None:
            mod.unlink(missing_ok=True)
        else:
            _atomic_write(mod, prev_mod)
        print("ROLLBACK dsv4_tp_pad.py", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Apply the DSV4 TP=3 padding patch (transactional).")
    ap.add_argument("--vllm-root",
                    default="/usr/local/lib/python3.12/dist-packages/vllm")
    ap.add_argument("--check", action="store_true",
                    help="report status, change nothing")
    ap.add_argument("--revert", action="store_true",
                    help="restore this tool's known files from their .tp3bak")
    ap.add_argument("--yes", action="store_true",
                    help="skip the interactive --revert confirmation "
                         "(non-TTY runs never prompt)")
    ap.add_argument("--selftest", action="store_true",
                    help="run the geometry self-tests and exit")
    ap.add_argument("--no-profiler-fix", action="store_true",
                    help="omit the CUDA-only profiler edit (the one edit that "
                         "is not TP-related)")
    ap.add_argument("--expect-vllm-version", default=EXPECTED_VLLM_VERSION,
                    help="vLLM version prefix this patch was validated on")
    ap.add_argument("--force-version", action="store_true",
                    help="proceed despite a vLLM version mismatch (anchors "
                         "are still preflighted, but anchor survival is NOT "
                         "proof of unchanged semantics)")
    args = ap.parse_args()

    # 0. The patcher's own math, before anything else. Always.
    fails = geometry_selftest()
    if fails:
        print("GEOMETRY SELFTEST FAILED -- refusing to touch anything:",
              file=sys.stderr)
        for f in fails:
            print(f"  FAIL {f}", file=sys.stderr)
        return 2
    if args.selftest:
        print("Geometry selftest: all checks passed.")
        return 0

    root = pathlib.Path(args.vllm_root)
    if not root.is_dir():
        print(f"ERROR: vllm root not found: {root}", file=sys.stderr)
        return 2

    if args.revert:
        # Restore ONLY the files this tool patches. A stray or planted
        # *.tp3bak elsewhere in the tree must never become a write primitive,
        # and symlinked backups/targets are refused for the same reason.
        known = {(root / f).resolve() for f, _, _, _ in EDITS}
        known |= {(root / rel).resolve() for rel, _ in SINK_VARIANTS}
        targets = []
        for bak in sorted(root.rglob("*.tp3bak")):
            orig = bak.with_suffix("")
            if bak.is_symlink() or orig.is_symlink():
                print(f"WARN  refusing symlinked backup: {bak}",
                      file=sys.stderr)
            elif orig.resolve() in known:
                targets.append((bak, orig))
            else:
                print(f"WARN  ignoring unrecognised backup: {bak}",
                      file=sys.stderr)
        mod = root / "models/deepseek_v4/dsv4_tp_pad.py"
        if not targets and not mod.exists():
            print("Nothing to revert.")
            return 0
        for _, orig in targets:
            print(f"will restore {orig.relative_to(root)}")
        if mod.exists():
            print("will remove dsv4_tp_pad.py")
        if not args.yes and sys.stdin.isatty():
            try:
                resp = input(f"Restore {len(targets)} file(s)? [y/N] ")
            except (EOFError, KeyboardInterrupt):
                # isatty() can be true on a stdin that still EOFs (Windows
                # console handles under redirection). No answer = no consent.
                resp = ""
            if resp.strip().lower() not in ("y", "yes"):
                print("Aborted; nothing was modified.")
                return 1
        for bak, orig in targets:
            shutil.copy2(bak, orig)
            bak.unlink()
            print(f"REVERT {orig.relative_to(root)}")
        if mod.exists():
            mod.unlink()
            print("REVERT dsv4_tp_pad.py removed")
        print(f"\n{len(targets)} file(s) reverted.")
        return 0

    # 1. Version pin. Anchors surviving on a different release proves the TEXT
    #    survived, not the semantics around it.
    detected = detect_vllm_version(root)
    if detected is None:
        print(f"WARN  could not detect a vLLM version under {root} "
              f"(expected {args.expect_vllm_version}); anchors are still "
              f"preflighted.", file=sys.stderr)
    elif not (detected == args.expect_vllm_version
              or detected.startswith((args.expect_vllm_version + ".",
                                      args.expect_vllm_version + "+",
                                      args.expect_vllm_version + "-"))):
        # Segment-aware: "0.25.1" accepts 0.25.1, 0.25.1.post1, 0.25.1+b12x --
        # but NOT 0.25.10, which a bare startswith would wave through.
        msg = (f"vLLM version mismatch: detected {detected}, this patch was "
               f"validated on {args.expect_vllm_version}.")
        if args.force_version:
            print(f"WARN  {msg} Proceeding on --force-version.",
                  file=sys.stderr)
        else:
            print(f"ERROR {msg}\n"
                  f"      Re-validate the edits against this release, then "
                  f"re-run with --force-version\n"
                  f"      (or --expect-vllm-version {detected} once "
                  f"validated).", file=sys.stderr)
            return 2
    else:
        print(f"vLLM {detected} matches expected {args.expect_vllm_version}")

    edits = EDITS
    if args.no_profiler_fix:
        edits = [e for e in EDITS if e[0] != PROFILER_EDIT_FILE]

    mod = root / "models/deepseek_v4/dsv4_tp_pad.py"

    if args.check:
        ms = module_status(mod)
        print(f"{ms:5} dsv4_tp_pad.py module"
              + (" -- content differs from this script's PAD_MODULE; a "
                 "re-run will rewrite it" if ms == "STALE" else ""))
        er, sr = collect_results(root, edits, check=True)
        for r in er + sr:
            print(r)
        bad = [r for r in er + sr if r.startswith(("MISS", "STALE"))]
        bad += [r for r in er if r.startswith("SKIP")]
        if bad or ms == "STALE":
            print(f"\n{len(bad)} problem(s) found. See lines above.",
                  file=sys.stderr)
            return 1
        print("\nCheck complete.")
        return 0

    # 2. PREFLIGHT -- prove every edit is applicable before writing anything.
    er, sr = collect_results(root, edits, check=True)
    pre = er + sr
    misses = [r for r in pre if r.startswith("MISS")]
    stales = [r for r in pre if r.startswith("STALE")]
    skips = [r for r in er if r.startswith("SKIP")]
    if misses or stales or skips:
        for r in pre:
            print(r)
        if misses:
            print(f"\n{len(misses)} anchor(s) not found. Do NOT boot TP=3 "
                  "until resolved -- an unpatched attn_sink loader fails "
                  "SILENTLY.", file=sys.stderr)
        if stales:
            print(f"\n{len(stales)} edit(s) present at an OLDER version. The "
                  "container is reusing a writable layer that already holds a "
                  "previous patch.\n"
                  "Fix: `docker rm -f <container>` on EVERY node, then start "
                  "again -- a compose restart REUSES the layer and will not "
                  "clear this.", file=sys.stderr)
        if skips:
            print(f"\n{len(skips)} required core file(s) missing -- wrong "
                  "--vllm-root or a broken install.", file=sys.stderr)
        print("\nPREFLIGHT FAILED -- nothing was modified.", file=sys.stderr)
        return 1

    # 3. APPLY -- preflight passed, so this should be all-or-nothing; if it
    #    somehow is not, roll this run's writes back.
    touched: list[pathlib.Path] = []
    prev_mod = mod.read_text(encoding="utf-8") if mod.exists() else None
    wrote_mod = False
    try:
        if prev_mod != PAD_MODULE:
            # Flag BEFORE the write (belt) + atomic write (braces): a write
            # that dies partway is both rolled back and never partial.
            wrote_mod = True
            _atomic_write(mod, PAD_MODULE)
            print("WRITE dsv4_tp_pad.py module")
        else:
            print("OK    dsv4_tp_pad.py module (current)")
        er, sr = collect_results(root, edits, check=False, touched=touched)
        results = er + sr
        bad = [r for r in results
               if not r.startswith(("PATCH", "OK", "SKIP"))]
        if bad:
            raise RuntimeError("apply-phase failure: " + "; ".join(bad))
        # 4. VERIFY -- every edit must now report installed-and-current.
        ver, vsr = collect_results(root, edits, check=True)
        vbad = [r for r in ver + vsr
                if not r.startswith(("OK", "SKIP"))]
        if vbad:
            raise RuntimeError("post-apply verify failed: " + "; ".join(vbad))
    except Exception as e:  # noqa: BLE001
        print(f"\nERROR during apply: {e}", file=sys.stderr)
        rollback(root, touched, mod, prev_mod, wrote_mod)
        print("All modifications from this run were rolled back.",
              file=sys.stderr)
        return 1

    for r in results:
        print(r)
    print("\nAll edits applied and verified. Runtime pad correctness "
          "(MoE zero columns, attn_sink -inf, TP2-vs-TP3 logits) still needs "
          "validate_tp3.sh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
