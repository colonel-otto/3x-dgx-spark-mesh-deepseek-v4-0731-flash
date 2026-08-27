# Context Garble Sweep — 2026-08-26 22:34

model: deepseek-v4-flash-0731 | endpoint: http://127.0.0.1:8100/v1 | runs/length: 2 | cold prefill: forced (unique nonce)

| ctx_len | run | verdict | finish | secs | reasoning_ch | tool_calls | flags | content |
|---|---|---|---|---|---|---|---|---|
| 2048 | 0 | CLEAN | stop | 104.4 | 0 | - | - | 'The capital of Idaho is **Boise**. Its metro area population is roughl' |
| 2048 | 1 | CLEAN | stop | 2.1 | 0 | - | - | 'The capital of Idaho is **Boise**. Its rough population is about **235' |
| 8192 | 0 | CLEAN | stop | 3.9 | 0 | - | - | 'The capital of Idaho is **Boise**. Its metro area population is roughl' |
| 8192 | 1 | CLEAN | stop | 19.7 | 0 | - | - | 'The capital of Idaho is **Boise**, with a metro-area population of rou' |
| 32768 | 0 | CLEAN | stop | 19.2 | 0 | - | - | 'The capital of Idaho is **Boise**, with a metro-area population of rou' |
| 32768 | 1 | CLEAN | stop | 2.9 | 0 | - | - | 'The capital of Idaho is **Boise**, with a metro-area population of rou' |
| 131072 | 0 | CLEAN | stop | 101.5 | 0 | - | - | 'The capital of Idaho is **Boise**. Its metro area population is roughl' |
| 131072 | 1 | CLEAN | stop | 260.5 | 0 | - | - | 'The capital of Idaho is **Boise**, with a metro-area population of rou' |
