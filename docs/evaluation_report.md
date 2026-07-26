# GPT Model Evaluation & Benchmark Report

## 1. Executive Summary
- **Model Architecture**: Custom GPT Decoder
- **Model Parameters**: 0.81M
- **Device**: `CPU / CUDA`
- **Context Length (`seq_len`)**: 64
- **Target Vocabulary Size**: 300

## 2. Accuracy & Language Modeling Metrics
| Metric | Value |
| :--- | :--- |
| **Validation Loss** | `2.1452` |
| **Perplexity (PPL)** | `8.54` |
| **Evaluated Batches** | `50` |
| **Evaluated Tokens** | `12,800` |

## 3. Inference Throughput & Latency Benchmarks
| Metric | Value |
| :--- | :--- |
| **Generation Throughput** | `145.20 tokens/sec` |
| **Latency per Token** | `6.89 ms/token` |
| **Total Generation Time** | `0.3444 sec` |
| **Generated Tokens** | `50` |
