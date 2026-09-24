# Run the expanded branch comparison on your CPU

This branch compares three language models using the **same sampled training
and validation examples and the same minibatch indices**: the shared-core
four-level quantized recurrent model, the shared-memory recurrent model with
up to eight live ideas, and an untied full-precision Transformer.

On Linux or macOS, in a terminal with Python 3.10 or newer:

```bash
git clone --branch experiment/expanded-branch-cpu https://github.com/UltDstroyer/ternary-recurrent-lm.git
cd ternary-recurrent-lm
bash scripts/run_expanded_branch_cpu.sh
```

If you already have the repository, switch to the branch and pull its latest
changes before running the last command. The launcher creates `.venv` and
installs PyTorch and NumPy if needed. On Linux it uses PyTorch's CPU wheel
index. Set `PYTHON_BIN=/path/to/python` to use a Python installation that
already has both packages; add `--skip-install` to avoid pip entirely.

The launcher first runs a two-step smoke test, then a 300-step comparison with
width 352, feed-forward width 1888, six passes, a 32-token prefix, batch size
four, and seed 19. The branching model allows eight live idea threads and
up to two splits per pass; sufficiently similar mature branches may merge.
The task asks each model to copy one of eight tokens from the entire supplied
prefix. The target is the next token **after** that prefix. The scratchpad
therefore cannot see a scored target.

Outputs are written to `results/expanded_branch_cpu_smoke.json` and
`results/expanded_branch_cpu_pointer.json`, with matching `.log` files. The
JSON records held-out cross-entropy, accuracy, parameter counts, time per
training run and inference batch, and branch split/merge/core-call statistics.
It records Python and PyTorch versions, CPU thread count, commit, and script
hash. Wall times depend on your CPU, and training can take a while.

The script saves JSON after each completed model. If interrupted, rerun
`bash scripts/run_expanded_branch_cpu.sh --resume`; any unfinished model
restarts, while completed models are skipped. Run `--torch-threads 8` if you
want eight PyTorch CPU threads; use the same option when resuming.

For an optional real-text byte-level next-token comparison, run:

```bash
bash scripts/run_expanded_branch_cpu.sh --resume --wikitext2
```

This downloads and checks the WikiText-2 train/validation splits, runs the
same 300-step three-model setup, and writes
`results/expanded_branch_cpu_wikitext2.json`. It scores only the byte just
beyond each 32-byte prefix. This is a small diagnostic, not a standard
WikiText-2 language-model perplexity benchmark.

The shared-core models reuse one set of core weights over six passes; the
untied baseline has six separately parameterized full-precision cores.
Branching adds a small number of learned idea offsets and a gate, but its
multiple live states execute the shared core repeatedly. Compare
`core_calls_per_example` and measured wall time alongside parameter counts:
equal optimization steps do **not** imply equal compute. Four-level weights
are simulated with PyTorch floating-point tensors; timing is not a measure
of packed low-bit hardware speed. The branch decisions use a batch-level
novelty/similarity average, so branch statistics depend on evaluation batch
size.

The automated two-step smoke check verifies that all three models run, but
its scores are not a trained comparison. Run the command above to produce
the 300-step results on your machine.
