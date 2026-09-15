# Improvement ablation on WikiText-2

Date: 2026-09-13

## Method

The experiment independently tested every proposed change against the same memory-v2
starting point, then combined the mechanisms that looked promising in a one-seed pilot.
Every row used a byte vocabulary, 64-byte context, width 64, six possible loops, 320
optimizer updates, and 327,680 sampled training tokens per seed. Results are mean ± sample
standard deviation across seeds 19, 23, and 29.

The predefined WikiText-2 training split supplied 10,797,141 usable bytes and its validation
split supplied 2,191 evenly spaced windows. The test split was never read. Lower
cross-entropy (CE) and perplexity are better. `CE change` is relative to the original
memory-v2 fixed model, so a negative value is an improvement.

## Results

| Variant | Validation CE | CE change | Accuracy | Expected loops | Validation tokens/s | Packed KiB |
|---|---:|---:|---:|---:|---:|---:|
| Full untied control | **2.3839 ± 0.0066** | −0.0439 | **34.21%** | 6.000 | 70,433 | 1,205.2 |
| Original memory v2 | 2.4279 ± 0.0016 | — | 33.27% | 6.000 | 49,600 | 100.5 |
| Loop conditioning | 2.4267 ± 0.0060 | −0.0011 | 33.45% | 6.000 | 50,837 | 103.5 |
| Low-rank adapters | 2.4257 ± 0.0028 | −0.0022 | 33.34% | 6.000 | 48,879 | 124.5 |
| Chunk-level memory | 2.4258 ± 0.0048 | −0.0021 | 33.41% | 6.000 | 43,366 | 100.5 |
| Progressive supervision | 2.4236 ± 0.0049 | −0.0042 | 33.38% | 6.000 | 51,051 | 100.5 |
| Staged memory | 2.4242 ± 0.0073 | −0.0037 | 33.12% | 6.000 | 50,704 | 100.5 |
| Gradual ternarization | **2.4078 ± 0.0038** | **−0.0200** | 33.63% | 6.000 | 49,280 | **100.5** |
| Teacher distillation | 2.4306 ± 0.0051 | +0.0027 | 33.16% | 6.000 | 50,986 | 100.5 |
| Prediction-stability halt | 2.4299 ± 0.0052 | +0.0020 | 33.14% | 5.004 ± 0.127 | 51,403 | 100.5 |
| Adaptive + stability halt | 2.4436 ± 0.0027 | +0.0158 | 32.65% | **2.876 ± 0.129** | 50,906 | 100.8 |
| Quality combination | **2.4072 ± 0.0033** | **−0.0206** | **33.69%** | 6.000 | 49,050 | 124.5 |
| Quality combination + stability halt | 2.4116 ± 0.0016 | −0.0162 | 33.48% | 4.033 ± 0.033 | **59,209** | 124.5 |

Bold among compact models marks the best relevant value. Throughput is a CPU measurement
from this run and should not be generalized to other hardware.

## What was fruitful

### Clear win: gradual ternarization

Training first with continuous weights and gradually increasing projection strength was the
only individual mechanism that produced a large, consistent improvement. It reduced CE by
0.0233, 0.0201, and 0.0166 in the three paired seeds. Its mean CE of 2.4078 closed 45.6% of
the original compact model's gap to the untied control without adding parameters or packed
storage.

### Small possible wins: progressive supervision and adapters

Progressive supervision improved mean CE by 0.0042 and low-rank adapters by 0.0022. Adapters
helped in all three seeds, but the gains were small. Progressive supervision helped strongly
in two seeds and was neutral in the third. When combined with gradual ternarization, the two
mechanisms improved CE by only another 0.0006, far below run-to-run variation, while adapters
raised packed size from 100.5 to 124.5 KiB. The simpler gradual-only model is therefore the
better current quality recommendation.

### Useful efficiency option: stability halting after quality training

Adding prediction-stability halting to the quality combination reduced expected depth from
6.000 to 4.033 (32.8%) and increased measured validation throughput from 49,050 to 59,209
tokens/s (20.7%). CE rose from 2.4072 to 2.4116, but remained 0.0162 better than the original
memory-v2 model. This is the best current quality/compute tradeoff.

### Not fruitful in their current forms

- Loop conditioning and chunk memory were inconsistent across seeds and too small to
  distinguish from noise. Chunk memory also reduced throughput because it performs several
  writes per loop.
- Delaying memory was inconsistent, helping one seed strongly but slightly hurting another.
- Distillation worsened mean CE by 0.0027. The short-trained teacher may not provide a useful
  enough probability target, or the distillation weight may still be too high.
- Adaptive + stability halting reduced expected depth to 2.876, but worsened CE by 0.0158
  relative to the compact baseline. It is stopping before useful refinement is complete.

## Comparison with the untied model

The compact models did not surpass the full untied control in this training budget. The
quality combination reached CE 2.4072 versus 2.3839, a remaining gap of 0.0233 and a
perplexity difference of about 2.4%. It nevertheless used 90,561 versus 308,544 parameters
(70.6% fewer) and an estimated 124.5 versus 1,205.2 packed KiB (89.7% smaller).

The next focused experiment should keep gradual ternarization, remove the mechanisms that
did not survive three seeds, and spend the saved compute on a longer training schedule. A
small adapter-rank sweep is justified, but further distillation should wait for a more fully
trained teacher.

## Boundaries

- Three seeds support an engineering decision, not a general language-model claim.
- Each run sampled windows across the training split but did not complete a full corpus epoch.
- Expected loops are probability-weighted depth. Some examples in every evaluation batch
  continued to loop six, so the batch-level maximum remained six even for halted variants.
- Packed size assumes two bits per ternary weight plus FP32 row scales. Current PyTorch
  execution still uses ordinary tensors and simulated projections.
