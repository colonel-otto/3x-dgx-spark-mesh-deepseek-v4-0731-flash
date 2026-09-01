# Context Garble Sweep — 2026-09-01 15:54

model: deepseek-v4-flash-dspark-abliterated | endpoint: http://127.0.0.1:8100/v1 | runs/length: 2 | cold prefill: forced (unique nonce)

| ctx_len | run | verdict | finish | secs | reasoning_ch | tool_calls | flags | content |
|---|---|---|---|---|---|---|---|---|
| 8192 | 0 | CLEAN | stop | 4.3 | 118 | - | - | 'Boise is the capital of Idaho, with a metro-area population of roughly' |
| 8192 | 1 | CLEAN | stop | 1.6 | 114 | - | - | 'The capital of Idaho is **Boise**. Its metro area population is roughl' |
| 32768 | 0 | CLEAN | stop | 10.6 | 131 | - | - | 'Boise is the capital of Idaho. Its metro area population is roughly 80' |
| 32768 | 1 | CLEAN | stop | 1.6 | 133 | - | - | 'The capital of Idaho is **Boise**. Its rough population is about **235' |
| 65536 | 0 | CLEAN | stop | 14.5 | 131 | - | - | 'Boise is the capital of Idaho. Its metro area population is roughly 80' |
| 65536 | 1 | CLEAN | stop | 1.6 | 114 | - | - | 'Boise is the capital of Idaho. Its metro area population is roughly 80' |
| 131072 | 0 | CLEAN | stop | 30.7 | 114 | - | - | 'Boise is the capital of Idaho, with a metro-area population of roughly' |
| 131072 | 1 | CLEAN | stop | 1.6 | 118 | - | - | 'Boise is the capital of Idaho, with a metro population of roughly 800,' |
