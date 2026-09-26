# Hybrid LLM on IRIS / CWE-Bench-Java -- every model pair

Quality is the asymmetric security loss of docs/01-problem.md 1.5 at rho = c_FN/c_FP = 10; label is upstream's `det_2cls` at tolerance t = 0.
Metrics are IRIS Sec 3.6 via their own `score_subset.metrics`, over 16 projects.

`label-following` is a perfect router on Hybrid LLM's own target: the ceiling the OBJECTIVE allows.
`pair-oracle` is a perfect router on any target: the ceiling the MODEL POOL allows.
`AUC random` splits inside a project and is leaky by construction; it is the control for `AUC LOPO`.

| small | large | %routed large | GPU-s saved | AvgF1 small | AvgF1 large | AvgF1 label-following | AvgF1 pair-oracle | AUC LOPO | AUC random (leaky) |
|---|---|---|---|---|---|---|---|---|---|
| qwen-7b | granite-8b | 16.6% | 58.9% | 0.142 | 0.207 | **0.405** | 0.445 | 0.471 | 0.943 |
| qwen-1.5b | granite-8b | 15.0% | 73.6% | 0.105 | 0.207 | **0.393** | 0.405 | 0.483 | 0.955 |
| qwen-7b | deepseek-6.7b | 36.1% | 58.5% | 0.142 | 0.223 | **0.386** | 0.412 | 0.541 | 0.726 |
| qwen-1.5b | deepseek-6.7b | 36.4% | 60.1% | 0.105 | 0.223 | **0.338** | 0.358 | 0.528 | 0.714 |
| qwen-1.5b | phi-3.8b | 54.5% | 20.6% | 0.105 | 0.202 | **0.290** | 0.400 | 0.380 | 0.861 |
| phi-3.8b | qwen-7b | 42.5% | 15.2% | 0.202 | 0.142 | **0.283** | 0.423 | 0.471 | 0.869 |
| phi-3.8b | deepseek-6.7b | 33.0% | 63.3% | 0.202 | 0.223 | **0.282** | 0.295 | 0.502 | 0.726 |
| granite-8b | deepseek-6.7b | 43.2% | 40.2% | 0.207 | 0.223 | **0.257** | 0.260 | 0.551 | 0.672 |
| phi-3.8b | granite-8b | 16.5% | 65.2% | 0.202 | 0.207 | **0.256** | 0.256 | 0.485 | 0.944 |
| qwen-1.5b | qwen-7b | 45.5% | 31.8% | 0.105 | 0.142 | **0.153** | 0.153 | 0.507 | 0.909 |
