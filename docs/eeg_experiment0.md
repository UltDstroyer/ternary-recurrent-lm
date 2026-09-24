# Experiment 0: EEG pipeline validation

The standing Neurotech Evaluation Roadmap defines this experiment as rest versus mental arithmetic. This implementation uses PhysioNet EEG During Mental Arithmetic Tasks (EEGMAT 1.0.0), 36 subjects and two recordings per subject: `_1` before arithmetic and `_2` during arithmetic. It predicts recording condition, **not** mental content.

## Protocol

- Download the EDF files into `data/eegmat/` (ignored by Git). No EDF data is committed. Exclude ECG/EOG/EMG/status channels and require exactly matching ordered channels across files; bandpass 1–30 Hz, resample to 128 Hz, use the first 60 seconds of each recording, and make 15 disjoint four-second windows per recording.
- Fixed subject split, seed 2026: 24 training, six validation, six held-out test subjects. Both recordings from any subject remain in one split; no windows from the same recording cross splits. Split and per-subject test metrics are written to JSON. Validation chooses neural checkpoints; test subjects do not choose epochs or hyperparameters.
- Fit channel standardization on training subjects only. For each of three training seeds, compare bandpower + standardized logistic regression, a compact EEGNet-inspired convolutional network, and a shared-core four-level recurrent EEG encoder (three fixed passes; no branches). Cross-subject held-out balanced accuracy and macro-F1, per-subject values, parameter counts, GPU allocated peak, and approximate batched latency are recorded.
- This is *cross-subject* evaluation. Within-subject evaluation would require a distinct recording-level protocol; there is only one recording per condition per subject, so random within-recording windows would inflate performance. Do not report these results as within-subject generalization.
- The dataset already underwent artifact removal by its authors. The rest condition is eyes-closed and precedes the arithmetic condition, so class differences may reflect eye state, order or recording context as well as arithmetic. It is a pipeline check, not evidence of thought decoding.

## On your Radeon RX 7700 XT

Use Ubuntu or WSL2 with a driver and ROCm PyTorch combination that AMD lists for the RX 7700 XT. AMD's current compatibility and installation pages are linked below. From the cloned branch:

```bash
python3 -m venv .venv
source .venv/bin/activate
# Install the current AMD-supported ROCm PyTorch build for your OS using AMD's instructions below.
bash scripts/run_eeg_experiment0.sh
```

The launcher deliberately checks for an existing ROCm-enabled PyTorch install. It installs the small EEG dependencies, prints the GPU name and runs the full three-seed study. It refuses to run the real study on CPU silently. For a fast pipeline check after dependencies install, run `bash scripts/run_eeg_experiment0.sh --smoke --epochs 1 --seeds 11` (synthetic and CPU-capable). To override Python during venv creation set `PYTHON_BIN=python3.11`.

Outputs: `runs/eeg_experiment0/metrics.json`; data and outputs are excluded from Git. Full dataset is about 175 MB. First run downloads it. Repeated runs reuse existing EDF files; if interrupted downloads are retried. Verify the printed GPU is your RX 7700 XT before starting the full run. The four-level modules use simulated quantization with full-precision master weights and ordinary GPU operations: theoretical two-bit weight storage is calculated separately, **not** achieved on disk or in VRAM.

## Gate to proceed

Run at least three seeds and inspect test subject distribution. The minimum gate is reproducibly above 50% balanced accuracy across seeds with no leakage or preprocessing errors. Research success means comparable accuracy to EEGNet with stable training. If a large gap or unexpectedly perfect performance appears, inspect channel selection and acquisition confounds before motor imagery.

Sources: https://physionet.org/content/eegmat/1.0.0/ ; https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html ; https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/pytorch-install.html
