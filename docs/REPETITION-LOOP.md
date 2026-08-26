# The enumerate-task repetition loop — root cause and fix

**Symptom:** at long context the model becomes "untrustworthy and stuck in an endless
loop." Originally reported from a coding agent against a ~180K-token session.

**The server was never at fault.** No config change and no rollback was needed. The cause
is a *sampling* problem, triggered by a *task shape*, and merely amplified by context
depth.

Measured 2026-08-19 on the 3-node TP=3 cluster, image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`.

## What it actually is

DeepSeek-V4-Flash loops on **enumerate-style tasks** — "list EVERY item", "for EACH
handler, state...", "do not summarize or abbreviate". It emits near-identical lines with
only an index changing, burns the entire `max_tokens` budget, and returns
`finish_reason: length`. It never stops on its own.

Greedy, 2,600 synthetic handlers, 195,258 prompt tokens:

```
- **handler_563**: Extracts payload, validates non-empty, calls `process_563`. **Phase 1**
- **handler_564**: Extracts payload, validates non-empty, calls `process_564`. **Phase 1**
- **handler_565**: Extracts payload, validates non-empty, calls `process_565`.
```

Note this **defeats naive loop detection**: because the index increments, there is no
byte-exact repeated substring. A check like `s.count(tail) >= 4` finds nothing. Detect it
by stripping digits and counting duplicate line *templates*.

## Context depth is an amplifier, not the cause

| Test | Prompt | Output | Finish | Near-repetition |
|---|---:|---:|---|---:|
| Small ctx + enumerate task | 7,138 | 3,757 | `stop` | **94%** |
| Large ctx + *summarize* task | 159,830 | 238 | `stop` | 10% |
| Large ctx + enumerate task | 195,258 | 16,000 | **`length`** | **99%** |

At only **7K tokens** the repetition is already 94% — the task alone is sufficient. At
160K with a *bounded* task the output is clean and stops in 238 tokens.

Depth matters only because it supplies more items to enumerate, so the loop runs long
enough to exhaust the budget. That is why it *feels* like a threshold around 160–185K:
that is where "long but finite" becomes "effectively endless".

## Temperature is NOT the fix

This is the counter-intuitive part, and the reason this page exists:

| Sampling | Output | Finish | Repetition |
|---|---:|---|---:|
| `temp=0, top_p=1.0` | 16,000 | `length` | **99%** |
| `temp=1.0, top_p=0.95` (**DeepSeek's own recommendation**) | 16,000 | `length` | **99%** |
| `temp=1.0, top_p=0.95, frequency_penalty=0.3` | **502** | **`stop`** | **10%** |

DeepSeek's README-recommended sampling loops **identically to greedy**. Raising
temperature does nothing. Only `frequency_penalty` breaks the cycle.

Verified it does not damage normal work: an ordinary codegen prompt returned clean,
correctly-terminated output both with and without the penalty (800 vs 719 tokens, both
`finish=stop`).

Verified it survives an agent-level temperature override: with `temperature=0` forced
*and* `frequency_penalty=0.3`, output was 418 tokens, `finish=stop`, 0% repetition. This
matters because many agent harnesses pin `temperature: 0` globally.

## The fix

Set on the client/provider:

```json
{ "temperature": 1.0, "top_p": 0.95, "frequency_penalty": 0.3 }
```

## Two adjacent traps found while diagnosing this

**Declared context must match the server.** An agent config declaring
`context: 1048576` against a server serving `460800` does not merely mis-report — many
harnesses derive their compaction budget from the *declared* limit. With a 25% retention
target, a declared 1M gives a 262,144-token target, larger than the entire session, so
compaction never meaningfully fires and history accumulates raw. Always read
`GET /v1/models` -> `max_model_len` and declare that.

**The context ceiling is not enforced.** `max_model_len` was advertised as 460,800, but a
deliberate 380,005-token prompt returned **HTTP 200**, not a 400. Context-overflow
rejection is not a safety net, so client-side `ContextOverflowError` paths that match on
"prompt is too long" will never fire.

## Diagnostic lesson

Two wrong theories were pursued before the reproduction — one blaming a recent tuning
change, one blaming a hosted endpoint. Both came from auditing configuration instead of
reproducing the failure. Both died in under a minute once the loop was actually
reproduced.

**Reproduce before diagnosing.** Peak KV usage during the "failure" was 26.7% with
`Waiting: 0` — the engine was never under pressure.
