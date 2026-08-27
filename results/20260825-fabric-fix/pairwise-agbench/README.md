# Pairwise NCCL allgather — the raw evidence for the 6.8x fabric degradation

Raw `agbench.py` output behind [`../../../docs/FABRIC-FIX-PARITY.md`](../../../docs/FABRIC-FIX-PARITY.md).
Each file is a size sweep; the headline figure is the last entry (`64MiB`, `busbw_GBs`).

Collected 2026-08-25 with **vLLM stopped** (a live engine holds ~119 of 121 GiB, and GB10's
unified memory means a second CUDA context cannot be created).

| file | what it measured | 64 MiB busbw | reading |
|---|---|---:|---|
| `ag_base2.json` | all 3 ranks, before the fix | **0.495 GB/s** | paced by the worst link |
| `ag_gdr2.json` | all 3 ranks, `NCCL_DMABUF_ENABLE=1` + `NCCL_NET_GDR_LEVEL=5` | ~0.49 GB/s | **no effect — ruled out** |
| `ag_p_s2.json` | sparkmain ↔ spark2 pair | **4.64 GB/s** | healthy pair |
| `ag_recheck.json` | sparkmain ↔ spark2, repeat | ~4.60 GB/s | reproduced |
| `ag_w2.json` | sparkmain ↔ spark1, 2-rank | ~0.72 GB/s | degraded |
| `ag_p_s1b.json` | sparkmain ↔ spark1, original cable | ~0.69 GB/s | degraded |
| `ag_alt.json` | sparkmain ↔ spark1, **alternate cable** (`roceP2p1s0f*`) | ~0.68 GB/s | **same — proves it is the node, not a cable** |
| `ag_fix.json` | sparkmain ↔ spark1, after the duplicate-IP cleanup | ~0.69 GB/s | routing hygiene did not fix it |
| `ag_postboot.json` | sparkmain ↔ spark1, **after rebooting spark1** | **4.781 GB/s** | **6.9x recovery** |

## Why `ag_alt.json` matters

Each node has four RDMA ports and only two were in use. Bringing up the unused
`roceP2p1s0f*` pair and re-measuring gave the *same* 0.68 GB/s. That is what ruled out a
bad cable and pointed at the node itself — and it saved a pointless hardware swap. A
previous conclusion recommending a cable swap was withdrawn on this evidence.

## Reproducing

`agbench.py` lives at [`../../20260824-seqs32-nccl/agbench.py`](../../20260824-seqs32-nccl/agbench.py).
Run one rank per node inside the production image, with `--device /dev/infiniband`,
`--ulimit memlock=-1` and `--shm-size 64gb` — an ad-hoc `docker run` without the device
passthrough silently measures **socket fallback**, not RDMA, and reports a plausible-looking
~0.44 GB/s that means nothing.

**Interpretation:** a healthy GB10 pair reads ~4.6 GB/s here. ~0.7 GB/s means a degraded
node — reboot it before trusting any benchmark. A 3-rank figure *below* your worst pair
means one node is pacing the collective.

> These are workload-shaped numbers (the harness mirrors the vLLM MTP allgather and tops
> out at a 67 MB input), not a hardware bandwidth ceiling. For the fabric's actual
> capability measured with the official instrument, see
> [`../../../docs/BANDWIDTH-COMPARISON.md`](../../../docs/BANDWIDTH-COMPARISON.md).
