# GB10 clocks cannot be locked — the "locked 3003 MHz" claim is withdrawn

**Established 2026-08-29** while preparing the configuration-identical 2v3 comparison.
Affects every claim in this repository that benchmarks ran "clock-locked at 3003 MHz".

---

## The claim

[`results/20260828-issue36-locked-clocks-suite/README.md`](../results/20260828-issue36-locked-clocks-suite/README.md)
states:

> Evidence bundle evaluating ... under **hardware-locked 3003 MHz GPU clocks** ...
> **GPU Clocks**: Locked to `(min: 3003 MHz, max: 3003 MHz)` via
> `sudo nvidia-smi -lgc 3003,3003` with persistence mode enabled.

[`DETACHED-EXECUTION-AND-VERIFICATION.md §2A`](DETACHED-EXECUTION-AND-VERIFICATION.md)
prescribes the same protocol.

## The claim is false, and the bundle's own evidence shows it

`gpu_clocks.csv`, committed inside that very bundle, contains exactly **one** data row:

```csv
name, clocks.current.graphics [MHz], clocks.max.graphics [MHz], temperature.gpu, power.draw [W]
NVIDIA GB10, 2522 MHz, 3003 MHz, 50, 15.57 W
```

Two independent problems:

1. **The clock is 2522 MHz, not 3003 MHz.** The lock did not take effect.
2. **The sample is from idle, not from the run.** 50 °C and 15.57 W is an unloaded GB10.
   The README describes this file as *"live hardware clock telemetry during execution"*;
   a single pre-run idle sample cannot support that description.

## Why the lock silently fails on GB10

`nvidia-smi -lgc 3003,3003` **returns success** on all three nodes:

```
GPU clocks set to "(gpuClkMin 3003, gpuClkMax 3003)" for GPU 0000000F:01:00.0
All done.
```

and then does not pin the clock. Observed immediately afterwards, GPUs idle:

| node | SM clock | applications clock | max | temp |
|---|---:|---:|---:|---:|
| sparkmain | 2502 MHz | 2418 MHz | 3003 MHz | 59 °C |
| spark1 | 2522 MHz | 2418 MHz | 3003 MHz | 68 °C |
| spark2 | 2476 MHz | 2418 MHz | 3003 MHz | 70 °C |

The reason is visible in the driver's own reporting (driver 580.173.02, CUDA 13.0):

```
Supported Clocks : N/A          # no discrete clock table to pin to
power.limit      : [N/A]        # no controllable power cap
power.max_limit  : [N/A]
```

GB10 is an **integrated SoC**: the GPU shares a package power budget with the CPU, and
the driver exposes neither a supported-clock table nor a settable power limit. `-lgc` is
accepted and recorded, but there is no mechanism behind it. This is a platform property,
not a misconfiguration — it reproduces identically on all three nodes.

## What actually governs the clock: power, not heat

From `nvidia-smi -q` cumulative event counters on sparkmain:

| Clocks event reason | Cumulative |
|---|---:|
| **SW Power Capping** | **10,800,240,649 µs** (~3 h) |
| HW Thermal Slowdown | 1,195,386 µs (~1.2 s) |
| HW Power Brake | 0 |
| Sync Boost | 0 |

Power capping dominates thermal slowdown by roughly **four orders of magnitude**. The
part floats its clock against a package power budget. Under a dense MoE decode the
sustained clock settles around **2420–2520 MHz**, not the 3003 MHz headline maximum.

Instantaneous `clocks_throttle_reasons.active` reads `0x0` even while this is happening,
so **the flag is not a safe check** — read the cumulative counters instead.

## Consequence for measurement

Clock is not a controllable variable on this hardware. It is a *floating* one, and it
moves with sustained load and with per-node thermals. During the first (aborted) matched
2v3 attempt, workers reached 83–86 °C while the head node sat at 75 °C. Under `TP=3`
every rank waits at the collective barrier for the slowest, so an unequal power/thermal
state across nodes propagates into the measured rate.

Observed in that aborted run — per-rep decode at 32K, in issue order:

```
54.9 → 53.8 → 52.8 → 50.8 → 46.0 → 51.4 → 46.9 tok/s     (spread 17.3%)
```

with TTFT declining monotonically alongside it (19.3 s → 17.3 s). Per-cell spread grew
with depth and elapsed time: **5.0 % (2K) → 12.2 % (8K) → 17.3 % (32K)**, against an
Issue #31 noise floor of 6.6–11.7 %.

**A 17.3 % spread cannot resolve the 7–17 % effect the 2v3 comparison exists to measure.**

## What to do instead

Clock locking is unavailable, so control the *thermal and power state* rather than the
clock:

1. **Do not claim locked clocks.** Remove the assertion from bundle READMEs and from
   `DETACHED-EXECUTION-AND-VERIFICATION.md §2A`, or restate it as an attempt that GB10
   does not honour.
2. **Sample clocks throughout every run**, not once. One idle sample proves nothing; the
   interesting behaviour is under sustained load.
3. **Record per-node temperature with every cell**, and publish it beside the spread.
4. **Equalise thermal state between arms.** A cool-down to a stated temperature band
   before each arm, so a 2-node arm is not measured against a heat-soaked 3-node arm.
5. **Treat spread as a first-class result.** A cell whose spread exceeds the Issue #31
   floor is a cell that cannot adjudicate a sub-20 % difference, and should be reported
   as such rather than reduced to a median.

## Scope of the correction

This does **not** invalidate the measurements in the Issue #36 bundle. Those runs
happened, under whatever clocks the parts chose. What is withdrawn is the *claim of
controlled clock conditions* — the runs were not clock-locked, so any inference resting
on "clocks were held constant, therefore this delta is attributable to X" needs rework.
Bundles asserting locked clocks should be relabelled, not deleted, per
[`BENCHMARK-POLICY.md`](BENCHMARK-POLICY.md).
