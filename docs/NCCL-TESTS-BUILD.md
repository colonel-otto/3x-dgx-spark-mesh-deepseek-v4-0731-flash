# Building nccl-tests to match the published comparison

Prerequisite for [#18](../../issues/18). Scouted 2026-08-25 while the soak held the
cluster; nothing here required a restart.

## The version trap

The container ships **two** NCCL libraries, and picking the wrong one silently invalidates
the comparison:

| library | version | used by |
|---|---|---|
| `/usr/lib/aarch64-linux-gnu/libnccl.so.2` | **2.28.9** | nothing — system copy, unused |
| `/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2` | **2.30.7** | **the running engine** |

The engine log confirms it: `NCCL version 2.30.7+cuda13.3`. That matches the NCCL version
in the forum result we are comparing against.

**`/usr/include/nccl.h` is the 2.28.9 header** (`NCCL_MAJOR 2 / MINOR 28 / PATCH 9`).
Building against it while running 2.30.7 is a mismatch. Use the headers that ship beside
the 2.30.7 library instead:

```
/usr/local/lib/python3.12/dist-packages/nvidia/nccl/include/nccl.h
/usr/local/lib/python3.12/dist-packages/nvidia/nccl/lib/libnccl.so.2
```

## Toolchain: build inside the container, not on the host

| | host | container |
|---|---|---|
| `nvcc` | ❌ absent | ✅ CUDA 13.0.88 |
| `nccl.h` | ❌ absent | ✅ (both versions — pick 2.30.7) |
| `make` / `g++` | — | ✅ |
| `mpirun` | ✅ Open MPI 4.1.6 | ❌ **absent** |

**Neither environment is complete on its own.** The container has the compiler and NCCL;
the host has MPI.

## Two ways forward

### Option A — build without MPI (simpler, recommended first)

`nccl-tests` builds fine with `MPI=0` and can be launched as one process per node using
its own bootstrap. This avoids installing MPI into the image.

```bash
docker run --rm -v /tmp/nt:/out --entrypoint bash <image> -c '
  git clone --depth 1 https://github.com/NVIDIA/nccl-tests /src &&
  cd /src &&
  make -j MPI=0 \
    NCCL_HOME=/usr/local/lib/python3.12/dist-packages/nvidia/nccl \
    CUDA_HOME=/usr/local/cuda &&
  cp build/all_gather_perf /out/'
```

Note `NCCL_HOME` points at the **2.30.7** package, so `include/` and `lib/` both resolve
there.

### Option B — MPI in-container (matches the forum recipe exactly)

Add `libopenmpi-dev` to the build and set `MPI=1 MPI_HOME=/usr/lib/aarch64-linux-gnu/openmpi`.
Only worth it if Option A leaves launcher differences as a live variable.

## Running it

Mirror the serving container's flags or the measurement is not comparable:

```
--device /dev/infiniband --ulimit memlock=-1 --shm-size 64gb --network host
```

and pass the same NCCL environment under test (`NCCL_IB_HCA`, `NCCL_IB_MERGE_NICS`,
`NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_NET_PLUGIN=none`, `NCCL_DEBUG=INFO`).

**Engine must be STOPPED** — a 16 GiB all_gather needs tens of GiB of buffer.

## Sizes to match the forum

```
all_gather_perf -b 32M -e 32M -f 2 -n 20     # their 18.64 GB/s point
all_gather_perf -b 16G -e 16G -f 2 -n 20     # their 20.84 GB/s point
```

`-n 20` matches their iteration count; ours used 10.

## Verify before trusting a number

1. `NCCL version 2.30.7` in the output — not 2.28.9.
2. `via NET/IB/*`, never `NET/Socket`.
3. Record the `Made virtual device ... name=` line: it names which HCAs NCCL merged, which
   is the whole NIC-merge question.

**Related:** [`BANDWIDTH-NEXT-TEST.md`](BANDWIDTH-NEXT-TEST.md) ·
[`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md) · [#18](../../issues/18)
