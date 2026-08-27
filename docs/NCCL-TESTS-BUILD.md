# NCCL tests build note — retired

The reusable requirements are now in [setup](setup.md): build against the exact NCCL
library used by the engine, mirror the serving RDMA environment, stop the engine for the
16 GiB test, and record runtime version, `#wrong=0`, and `NET/IB` transport.

For the controlled result and its evidence, see [bandwidth comparison](BANDWIDTH-COMPARISON.md)
and [`20260826-nccl-controlled`](../results/20260826-nccl-controlled/). The original
build narrative remains in Git history.
