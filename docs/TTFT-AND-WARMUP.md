# TTFT and warm-up note — retired

Warm benchmark shapes after restart, keep `JIT_MONITOR_MODE=warn`, and discard a sweep
containing JIT compilation. Do not use a periodic keep-alive: idle costs about 22 ms,
while a new shape can compile during inference and add seconds.

These current rules are owned by [benchmark policy](BENCHMARK-POLICY.md). The original
measurements are available at the
[last full revision](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/TTFT-AND-WARMUP.md).
