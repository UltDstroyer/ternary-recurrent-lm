# Run the expanded branch comparison on your RX 7700 XT

Use the ROCm-enabled PyTorch installation that already sees your GPU. ROCm
PyTorch uses the `cuda` device name in its Python API, even for an AMD Radeon
GPU. The launcher checks the ROCm build, GPU availability, and a GPU matrix
multiplication before starting. It does not fall back to CPU.

On a ROCm-capable Linux setup, in a terminal with the ROCm Python environment
active:

```bash
git clone --branch experiment/expanded-branch-cpu https://github.com/UltDstroyer/ternary-recurrent-lm.git
cd ternary-recurrent-lm
bash scripts/run_expanded_branch_rocm.sh
```

If you already cloned the repository, switch to
`experiment/expanded-branch-cpu` and pull before running the last command.
If PyTorch is installed in a different Python environment, use
`PYTHON_BIN=/path/to/rocm/python bash scripts/run_expanded_branch_rocm.sh`.
The Python environment must have PyTorch with ROCm support and NumPy.

The script first runs a two-step three-model smoke test. It then trains the
shared-core four-level model, expanded shared-memory branches, and untied
full-precision Transformer for 300 steps each. Each gets the same 4,096
training examples, 512 validation examples, seed 19, and minibatch indices.
Width is 352, feed-forward width is 1888, and each model uses six passes or
blocks. Branching allows eight live threads with up to two splits per pass.
The eight-way pointer target is strictly beyond the complete 32-token input
prefix, so the scratchpad cannot see the scored token.

The smoke and larger results appear as JSON and matching logs in
`results/expanded_branch_rocm_smoke.*` and
`results/expanded_branch_rocm_pointer.*`. JSON is saved after each completed
model. If a run stops, use:

```bash
bash scripts/run_expanded_branch_rocm.sh --resume
```

If the full run exceeds available GPU memory, rerun with
`--batch-size 2` (without `--resume` when changing batch size). For an
optional WikiText-2 byte-level next-token comparison, run
`bash scripts/run_expanded_branch_rocm.sh --resume --wikitext2`.
That scores only the byte after each full 32-byte prefix. It is a diagnostic,
not standard WikiText-2 perplexity.

The JSON includes held-out accuracy and cross-entropy, parameter counts,
training time, inference batch time, and branch split/merge and core-call
statistics, plus PyTorch, ROCm, and device identification. The branching
model reuses one quantized core but evaluates multiple live states; equal
optimizer steps do not imply equal compute. Quantization is simulated with
floating-point tensors, so timings do not represent packed four-level kernels.

The launcher expects an existing working ROCm stack because GPU drivers,
operating systems, and Python wheels must match. AMD lists the RX 7700 XT in
its [Linux compatibility matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/native_linux/native_linux_compatibility.html)
and provides [PyTorch installation instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html).
Its current [Windows matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)
does not list the RX 7700 XT. The two-step workflow in GitHub Actions uses a
CPU runner to test the model code; it cannot verify your ROCm hardware.
