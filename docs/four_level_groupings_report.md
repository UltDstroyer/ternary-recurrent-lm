# Four-level weight-grouping experiment

## Recommendation

Keep the uniform four-level alphabet `[-1, -1/3, +1/3, +1]` as the default. Across three 1,024-step seeds it remained the most accurate option at **74.24%**, compared with **73.33%** for zero-positive and **73.19%** for narrow-symmetric.

If exact zero weights are valuable for future sparse hardware, use zero-positive `[-1, 0, +1/2, +1]` as the alternative. It produced about **50.5% zero weights**, ranked second in confirmed accuracy, and beat narrow-symmetric in observed throughput.

## Confirmed ranking

All three models have 3,100,065 parameters and an estimated packed size of 1,233,120 bytes. Accuracy and cross-entropy are three-seed means.

| Rank | Weight alphabet | Accuracy | Cross-entropy | Inference reference | Exact zeros |
|---:|---|---:|---:|---:|---:|
| 1 | Uniform `[-1, -1/3, +1/3, +1]` | 74.24% | 0.8191 | 8,091 tok/s | 0% |
| 2 | Zero-positive `[-1, 0, +1/2, +1]` | 73.33% | 0.8474 | 6,659 tok/s | 50.54% |
| 3 | Narrow `[-1, -1/4, +1/4, +1]` | 73.19% | 0.8532 | 6,096 tok/s | 0% |

The inference reference for the two new groupings is the mean of seeds 19 and 23, which ran under the same PyTorch runtime. The runtime was reset before seed 29 and the replacement build was substantially slower, so seed-29 timing is excluded from this speed reference. The uniform timing comes from its earlier three-seed run and should also be treated as approximate rather than a hardware benchmark.

The implementation simulates quantization with ordinary CPU tensors. It does not use packed two-bit or sparse kernels. A specialized sparse implementation could make zero-positive more attractive than these CPU timings suggest.

## Full six-way screen

The initial screen used one seed and 320 training steps. Ranking below balances accuracy first and inference time as a reference; it was used to select narrow-symmetric and zero-positive for confirmation.

| Accuracy rank | Grouping | Accuracy | Throughput | Approx. ms / 1,000 tokens |
|---:|---|---:|---:|---:|
| 1 | Narrow symmetric | 51.42% | 5,710 tok/s | 175 |
| 2 | Zero-positive | 51.30% | 4,814 tok/s | 208 |
| 3 | Uniform | 51.26% | 4,634 tok/s | 216 |
| 4 | Zero-negative | 51.06% | 3,469 tok/s | 288 |
| 5 | Wide symmetric | 50.95% | 5,213 tok/s | 192 |
| 6 | Adaptive zero | 50.54% | 2,751 tok/s | 364 |

The short screen overestimated narrow-symmetric: its lead did not persist at 1,024 steps across three seeds. This is why the confirmed ranking should guide model selection.

## Reproduction

```bash
PYTHONPATH=src python -m trlm.experiment \
  --config configs/four_level_groupings_confirm.json \
  --output results/four_level_groupings_confirm.json \
  --resume \
  --variants s4_symmetric_narrow u4_zero_positive
```

Raw confirmation results are in `results/four_level_groupings_confirm.json` and `.csv`; screen results are in `results/four_level_groupings_screen.json` and `.csv`.
