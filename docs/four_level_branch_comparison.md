# Four-level recurrent versus bounded branches versus untied Transformer

Experiment branch: `experiment/four-level-branch-comparison`. The tested commit
is `728d2f2e6aec3a98110e31f4359950be2bf55eec`.
[GitHub Actions run](https://github.com/UltDstroyer/ternary-recurrent-lm/actions/runs/36027502593)
contains the raw JSON artifacts and exact job log.

## Commands and environment

```bash
python -m pip install -e .
bash scripts/run_four_level_branch_probe.sh --device cpu --steps 2
bash scripts/run_four_level_branch_probe.sh --device cpu --passes 6 --steps 400 --train-count 2048 --validation-count 256 --batch-size 8 --output results/four_level_branch_pilot.json
bash scripts/run_four_level_branch_probe.sh --device cpu --dataset wikitext2 --passes 6 --steps 400 --train-count 4096 --validation-count 256 --batch-size 8 --output results/four_level_branch_wikitext2.json
bash scripts/run_four_level_branch_probe.sh --device cpu --full-size --steps 1 --train-count 16 --validation-count 8 --batch-size 1 --output results/four_level_branch_full_size_shapecheck.json
```

GitHub hosted Ubuntu runner; Python 3.11.16; PyTorch 2.14.0+cu130;
`torch.cuda.is_available() == False`. Default seed 19. WikiText-2 downloaded
with the repository's checksum-verifying script; train and validation are the
official separate files, and the test split was not evaluated.

All models use the same prefabricated training and validation inputs and labels,
the same sampled minibatch indices, 400 optimizer updates, AdamW at 5e-4, and
the same width (64), feed-forward width (192), prefix length (12), six passes,
and batch size (8) in the two pilots. For both quantized models, the repository's
0.15 warmup / 0.70 ramp schedule is used, with fully quantized validation.
The full-precision baseline has six independent causal Transformer blocks;
the recurrent models share one core across passes. The branching model reuses
that core with up to four live states, batched on each pass, reading one memory
snapshot and committing writes into one shared scratchpad for the next pass.
Branches split when memory novelty exceeds 0.15 and merge when their final
state cosine similarity exceeds 0.98 after the third pass. These thresholds
are hand-set controls, not trained or tuned against held-out accuracy.

Every target is the one byte/token **after** the complete supplied prefix.
Only the final-position output is scored or trained. Scratchpad summaries
can therefore access the supplied prefix but never a scored target. Synthetic
targets copy one of two prefix positions, selected by a marker. WikiText-2
uses sampled byte windows of length 12 and predicts the next byte.

## Six-pass CPU pilots

Each validation set has 256 next-token examples. Training times and inference
times are from one hosted CPU runner and should not be interpreted as GPU speed.

| Dataset | Model | Params | Held-out accuracy | Cross-entropy (nats) | Training (s) | Inference (ms / batch of 8) | Logical core calls / example |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Synthetic | Serial four-level | 71,873 | 100.0% | 0.0659 | 13.24 | 19.51 | 6 |
| Synthetic | Shared-memory branches | 72,129 | 100.0% | 0.0705 | 16.82 | 27.02 | 14 |
| Synthetic | Untied FP32 Transformer | 322,560 | 100.0% | 0.0406 | 3.36 | 2.11 | 6 |
| WikiText-2 | Serial four-level | 87,233 | 24.22% | 2.7638 | 13.39 | 20.60 | 6 |
| WikiText-2 | Shared-memory branches | 87,489 | 24.22% | 2.7702 | 16.49 | 23.61 | 14 |
| WikiText-2 | Untied FP32 Transformer | 337,920 | 25.00% | 2.8504 | 3.27 | 2.11 | 6 |

The branching model reached four live states, with 5 splits and 5 merges per
validation example in each 400-step pilot. A count above four is possible:
after a merge the model can split again on a later pass. Its extra compute
did not improve validation cross-entropy in these pilots. WikiText-2 outcomes
are based on only 256 sampled next-byte targets and a single seed, so small
accuracy and loss differences are inconclusive. On the simple synthetic task
all three models achieved ceiling accuracy; its cross-entropy remains
distinguishable but does not establish general language-model quality.

## Full-size architecture check (not a trained comparison)

The one-step CPU check successfully constructed and evaluated the requested
width-352, feed-forward-1888, six-pass shapes, with prefix length 32. The
validation scores after one update have no useful interpretation.

| Model | Trainable parameters | Four-level quantized weight parameters |
| --- | ---: | ---: |
| Serial recurrent | 3,004,321 | 2,985,312 |
| Shared-memory branches | 3,005,729 | 2,985,312 |
| Untied FP32 Transformer | 14,960,000 | 0 |

The branching model adds 1,408 trainable parameters for the idea offsets and
gate relative to serial. It reuses the same quantized core but can call it
more than twice as many times per example. The baseline has six separately
parameterized cores. Four-level projection is simulated in PyTorch with FP32
master weights: parameter count is not packed inference storage, and no
four-level hardware acceleration is measured. Equal optimization steps and
minibatches here give equal sample exposure, not equal compute.

The requested trained CUDA full-size run was **not performed**: the GitHub
hosted runner reports CUDA unavailable. Run it on a CUDA-capable PyTorch
installation (or a supported ROCm installation where `torch.cuda.is_available()`
returns true) with:

```bash
bash scripts/run_four_level_branch_probe.sh --device cuda --full-size
```

For a real-text full-size comparison, use:

```bash
bash scripts/run_four_level_branch_probe.sh --device cuda --full-size --dataset wikitext2 --output results/four_level_branch_full_wikitext2.json
```
