# Million-token training comparison on WikiText-2

Date: 2026-09-13

## Method

This follow-up trained the two full-precision controls, the original compact model, and the
three most useful compact variants from the earlier ablation for 1,024 optimizer updates.
With batches of 16 sequences and 64 tokens per sequence, each model sampled exactly
1,048,576 training tokens: 3.2 times the earlier 320-step budget. The corpus itself remained
WikiText-2; this is a larger training budget on the same data, not a larger dataset.

Every row reports mean ± sample standard deviation across seeds 19, 23, and 29. All models
used the same 10,797,141-byte training split and the same 2,191 fixed validation windows.
The predefined test split was not read. Lower cross-entropy (CE) and perplexity are better.
`CE change` is relative to the original compact memory-v2 model.

## Results

| Variant | Validation CE | CE change | Perplexity | Accuracy | Expected loops | Validation tokens/s | Packed KiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full untied control | **1.9355 ± 0.0183** | −0.1621 | **6.9285** | **44.42%** | 6.000 | **70,772** | 1,205.2 |
| Full tied control | 2.0446 ± 0.0076 | −0.0531 | 7.7258 | 41.54% | 6.000 | 66,642 | 281.5 |
| Original compact memory v2 | 2.0977 ± 0.0082 | — | 8.1473 | 39.39% | 6.000 | 51,640 | **100.5** |
| Gradual ternarization | 2.0714 ± 0.0064 | −0.0263 | 7.9359 | 40.27% | 6.000 | 49,258 | **100.5** |
| Quality combination | **2.0507 ± 0.0060** | **−0.0469** | **7.7735** | **40.84%** | 6.000 | 45,122 | 124.5 |
| Quality combination + stability halt | 2.0559 ± 0.0072 | −0.0418 | 7.8137 | 40.78% | **5.661 ± 0.266** | 47,647 | 124.5 |

Bold among compact models marks the best relevant value. Throughput is a CPU measurement
from this run. Packed size is an analytical estimate, not the size or speed of the current
PyTorch checkpoint.

## Main findings

### The quality combination became a real long-training win

The quality combination—gradual ternarization, rank-8 loop adapters, and progressive
supervision—reached CE 2.0507. It improved on gradual ternarization alone by 0.0207 mean CE
and did so in all three paired seeds. That is different from the 320-step experiment, where
the combination beat gradual training by only 0.0006. The adapters and auxiliary losses
appear to need the longer budget to pay off, although this combined experiment cannot say
which of the two supplied the gain.

Relative to the original compact model, the combination reduced CE by 0.0469. It closed
88.4% of that model's gap to the full-precision tied control. The remaining tied-control gap
was 0.0062 CE; all three seeds favored the tied model, but three seeds are too few to make a
strong general claim from such a small difference.

The combination uses 90,561 parameters versus 72,064 for the tied control because of its
adapters, but ternary packing gives an estimated footprint of 124.5 KiB versus 281.5 KiB:
55.8% smaller.

### More training widened the gap to the untied model

The compact model did not surpass the full untied control. From 320 to 1,024 steps, the
untied model improved by 0.4484 CE, while the quality combination improved by 0.3565. Their
gap therefore grew from 0.0233 to 0.1152 CE. The extra untied per-loop capacity benefited
more from the longer schedule than the reused compact core did.

This does not erase the compact model's storage advantage. The quality combination uses
70.6% fewer parameters and has an estimated packed footprint 89.7% smaller than the untied
model. It does show that more training alone is not the route to beating the untied model at
this scale.

### Fixed-threshold early exit became less valuable

Stability halting lowered expected depth from 6.000 to 5.661 loops, a 5.7% reduction, and
raised measured throughput by 5.6%. Mean CE worsened by 0.0051. At 320 steps the same rule
cut expected depth by 32.8%, so the fixed KL threshold did not transfer cleanly to the more
fully trained models. A calibrated threshold sweep, or a target-compute controller, is
needed before calling early exit a long-training win.

## Comparison with the 320-step run

| Variant | CE at 320 steps | CE at 1,024 steps | Improvement |
|---|---:|---:|---:|
| Full untied control | 2.3839 | **1.9355** | **0.4484** |
| Original compact memory v2 | 2.4279 | 2.0977 | 0.3302 |
| Gradual ternarization | 2.4078 | 2.0714 | 0.3364 |
| Quality combination | 2.4072 | **2.0507** | **0.3565** |
| Quality combination + stability halt | 2.4116 | 2.0559 | 0.3558 |

Among compact variants, the quality combination made the best use of the extra updates.
The full untied model nevertheless had the steepest improvement overall.

## Recommended next experiment

Keep gradual ternarization, progressive supervision, and the loop adapters. To address the
remaining untied gap, test a small amount of loop-specific capacity rather than simply
training longer: adapter ranks 8, 16, and 32, plus selectively untied feed-forward blocks.
Match packed-size budgets so the comparison remains meaningful. Separately sweep the
stability threshold to target 5.0, 4.5, and 4.0 expected loops; do not mix that efficiency
sweep into the quality architecture search.

## Boundaries

- Three seeds are enough for this engineering choice, not a broad scaling-law claim.
- Runs sampled windows from the training split and did not complete a full corpus epoch.
- The long run changed optimizer steps only; it did not tune the learning-rate schedule for
  the longer horizon.
- The implementation simulates ternary projection with ordinary PyTorch tensors. It does
  not contain packed ternary kernels, so measured CPU throughput does not reflect packed
  storage savings.
- Expected loops are probability-weighted depth. Some validation batches still executed all
  six loops for the halted model.
