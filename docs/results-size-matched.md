# Packed-size-matched compressed model experiment

Date: 2026-09-14

## Question

Can ternary compression build a much more capable recurrent model in the same stored-model
footprint as a full-precision model with six separate processing blocks?

Here, "same size" means estimated packed inference-weight storage. It does not mean equal
parameter count, training memory, activation memory, arithmetic work, or latency. The
current PyTorch implementation simulates ternary weights with FP32 tensors; the packed sizes
require a future exporter and packed ternary kernels.

## Method

The reference has width 64 and a different full-precision core for each of six passes. Its
308,544 parameters occupy 1,234,176 bytes in FP32.

Four recurrent candidates reuse one ternary core for all six passes. Their widths,
feed-forward sizes, and full-precision adapter ranks were chosen so every estimated packed
footprint falls within 0.1% of the reference. All use gradual ternarization. The adapter
candidates also use intermediate supervision, matching the earlier quality recipe.

Each run used 1,024 optimizer updates, batches of 16 sequences, and 64-byte sequences, for
exactly 1,048,576 sampled training tokens. The predefined WikiText-2 training and validation
splits were used; the test split was not read. A seed-19 screen compared all allocations.
The winner was confirmed on seeds 23 and 29. Lower cross-entropy (CE) and perplexity are
better.

The original detailed result checkpoint was lost when the transient compute container was
replaced. The summary CSV in this repository was reconstructed from the complete metrics
logged in the project conversation; it contains every metric used in this report.

## Seed-19 storage-allocation screen

| Friendly description | Width | Feed-forward | Adapter rank | Parameters | Packed bytes | Validation CE | Accuracy | Validation tokens/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Six separate full-precision blocks | 64 | 160 | — | 308,544 | 1,234,176 | 1.9560 | 44.02% | **103,206** |
| **Maximum shared ternary capacity** | **352** | **1,888** | — | **3,100,065** | 1,233,120 | **0.8274** | **73.86%** | 8,885 |
| Wide core + small pass adapters | 352 | 1,392 | 8 | 2,610,081 | 1,233,376 | 0.8602 | 72.74% | 10,958 |
| Medium core + medium adapters | 336 | 1,152 | 16 | 2,238,769 | 1,233,692 | 0.8973 | 72.04% | 12,757 |
| Narrower core + large adapters | 288 | 1,072 | 32 | 1,794,529 | 1,234,384 | 1.0163 | 68.79% | 13,311 |

The best allocation put the byte budget into the reusable ternary core rather than
full-precision pass-specific adapters. Increasing adapter rank progressively reduced shared
width and quality. This screen changes several dimensions together to satisfy a fixed byte
budget, so it identifies the best tested allocation rather than isolating adapter rank alone.

## Three-seed confirmation of the winner

Results are mean ± sample standard deviation across seeds 19, 23, and 29.

| Model | Validation CE | Perplexity | Accuracy | Expected passes | Parameters | Packed bytes |
|---|---:|---:|---:|---:|---:|---:|
| Six separate full-precision blocks | 1.9355 ± 0.0183 | 6.9285 ± 0.1272 | 44.42% ± 0.35% | 6.000 | 308,544 | 1,234,176 |
| **Wide reusable ternary core** | **0.8467 ± 0.0181** | **2.3321 ± 0.0421** | **73.25% ± 0.60%** | 6.000 | **3,100,065** | **1,233,120** |

| Seed | Separate-block reference | Wide compressed model | CE improvement |
|---:|---:|---:|---:|
| 19 | 1.9560 | **0.8274** | 1.1287 |
| 23 | 1.9297 | **0.8494** | 1.0803 |
| 29 | 1.9209 | **0.8632** | 1.0577 |

The wide compressed model reduced mean CE by 1.0889, or 56.3%, and increased exact
next-byte accuracy by 28.84 percentage points. It contains 10.05 times as many trainable
parameters while its estimated packed representation is 1,056 bytes, or 0.086%, smaller
than the reference.

## Interpretation and cost

Compression made room for a much wider shared representation and feed-forward network.
Because that capable core is reused six times, every stored ternary parameter contributes to
every processing pass. For this workload and storage budget, that allocation was dramatically
more effective than six small independent full-precision blocks.

Equal stored-weight size does not imply equal computation. In the matched seed-19 run, the
wide compressed model took 6.68 times as long to train and processed validation tokens 11.62
times more slowly than the reference. It also occupies 12,400,260 bytes as current FP32
PyTorch parameters; 1,233,120 bytes is the estimated exported ternary representation.

Therefore the supported claim is:

> At the same estimated packed weight-storage budget, the wide recurrent ternary model is
> much more accurate than the six-block full-precision reference, at a substantial compute
> and latency cost in the current implementation.

It is not yet valid to claim a same-speed, same-memory, or deployable 1.21 MiB model.

## Recommended next tests

1. Compare 3-, 4-, and 5-level weights with the winning architecture unchanged.
2. Add a packed quantized exporter and verify predicted file sizes exactly.
3. Build a quality/latency frontier at widths between 192 and 352.
4. Compare against a full-precision reference matched on arithmetic work, not only bytes.
5. Retune stability-based early termination on the winning wide model.

## Boundaries

- Three seeds strongly support this engineering result but not a general scaling law.
- The experiment uses a small byte-level model and one corpus.
- Training sampled windows from the training split and did not complete a corpus epoch.
- The learning rate was inherited rather than independently tuned for each width.
- The one-seed adapter rankings need replication if distinctions among losing candidates
  become important.
