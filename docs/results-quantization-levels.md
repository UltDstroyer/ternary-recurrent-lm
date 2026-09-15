# Quantization-level comparison

## Outcome

The 5-level model was the most accurate in all three seeds. The 4-level model also beat the
3-level model in all three seeds while requiring the same ordinary packed binary size. The
5-level model's stronger result costs more storage under ordinary binary packing.

All variants used the same winning recurrent architecture: width 352, eight attention heads,
feed-forward width 1888, a shared core reused for six passes, external memory, and gradual
quantization. Each run trained on 1,048,576 sampled WikiText-2 training bytes. Validation used
the predefined validation split. There were three seeds: 19, 23, and 29.

| Weight levels | Validation loss (mean ± SD) | Perplexity | Accuracy | Estimated packed size |
|---:|---:|---:|---:|---:|
| 3 | 0.8467 ± 0.0181 | 2.3321 ± 0.0421 | 73.25% ± 0.60 pp | 1,233,120 bytes |
| 4 | 0.8191 ± 0.0097 | 2.2686 ± 0.0221 | 74.24% ± 0.26 pp | 1,233,120 bytes |
| 5 | **0.7820 ± 0.0198** | **2.1862 ± 0.0433** | **75.28% ± 0.72 pp** | 1,606,284 bytes |

Lower validation loss and perplexity are better; higher accuracy is better. Perplexity can be
read as the model's effective number of plausible next-byte choices, so a reduction is useful.

## Paired seed comparisons

Using the same seeds makes the comparisons less noisy:

- Four levels reduced mean validation loss by 0.0276 (3.25%) versus three levels and raised
  accuracy by 0.98 percentage points. The loss improvement occurred in every seed.
- Five levels reduced mean validation loss by 0.0647 (7.64%) versus three levels and raised
  accuracy by 2.03 percentage points. It won in every seed.
- Five levels reduced mean validation loss by 0.0371 versus four levels and raised accuracy by
  1.05 percentage points.

The loss changes by seed (comparison minus reference; negative is better) were:

| Comparison | Seed 19 | Seed 23 | Seed 29 | Mean |
|---|---:|---:|---:|---:|
| 4 levels − 3 levels | -0.0095 | -0.0394 | -0.0338 | -0.0276 |
| 5 levels − 3 levels | -0.0660 | -0.0655 | -0.0624 | -0.0647 |
| 5 levels − 4 levels | -0.0566 | -0.0261 | -0.0286 | -0.0371 |

## Storage and sparsity

Three and four levels both need two bits per quantized weight in an ordinary binary format.
Five levels need three bits, making the full packed model 30.3% larger than the three- and
four-level versions. With an ideal base-5 entropy code, the five-level estimate would be about
1,353,252 bytes, only 9.7% above the four-level binary estimate, but that is not the current
packing assumption.

The 3-level and 5-level learned weights remained about 39% and 40% exactly zero, respectively.
The symmetric 4-level alphabet has no zero value. That means the 4-level model gives a modest
same-bit-width accuracy gain but loses the sparsity that specialized ternary hardware could
exploit.

## Simulated training speed

The measured CPU throughputs were approximately 9,084 tokens/s for three levels and 8,091
tokens/s for four levels. The 5-level seeds ran across two temporary compute environments, so
their timing is not directly comparable as one benchmark. The generic multi-level quantizer
performs extra assignment and scale-refinement work during every projection, while the legacy
3-level path is optimized. These simulated training speeds are implementation costs, not
evidence about eventual packed inference hardware.

## Interpretation

The 4-level version is the cleanest next candidate when the storage budget is strict: it gains
quality without increasing the conventional two-bit weight allocation. The 5-level version is
the best quality candidate and its improvement is consistent, but it is not a same-size result
under ordinary binary packing. A useful next experiment is to compare 4 and 5 levels at an
exact total-byte budget, reallocating the 5-level model's width so both packed models occupy the
same space.

## Reproducibility note

All nine runs completed successfully. Complete per-run measurements are preserved in
`results/quantization_levels.json` and `results/quantization_levels.csv`; aggregate statistics
are in `results/quantization_levels_summary.csv`. The final 5-level seed was resumed from the
eight-run checkpoint after a temporary workspace reset.
