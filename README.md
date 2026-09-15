# Ternary Recurrent LM

An experimental byte-level language model that reuses a compact ternary transformer core,
writes intermediate ideas to external memory, and can stop when its predictions stabilize.

The current experiment independently tests loop conditioning, low-rank per-loop adapters,
chunk-level memory, progressive supervision, staged memory, gradual ternarization, teacher
distillation, and prediction-stability halting. It also tests a combination of the mechanisms
that were promising in a one-seed pilot.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_wikitext2.py
pytest
trlm-ablate --config configs/improvements.json --output results/improvements.json
trlm-ablate --config configs/long_training.json --output results/long_training.json
trlm-ablate --config configs/size_matched.json --output results/size_matched.json
trlm-ablate --config configs/quantization_levels.json --output results/quantization_levels.json
```

The runner checkpoints after every variant. Add `--resume` to skip completed seed/variant
pairs after an interruption.

The focused long-training suite compares six controls and prior winners over exactly
1,048,576 sampled training tokens per model and three seeds.

The size-matched suite reallocates the large separate-block model's estimated packed
inference footprint to wider ternary reusable-core models. It compares several divisions of
that fixed byte budget between shared feed-forward capacity and pass-specific adapters.

The quantization-level suite holds the winning wide architecture fixed while comparing
3-, 4-, and 5-level weight alphabets over the same three seeds and training budget.

The completed three-seed results and interpretation are in
[`docs/results-improvements.md`](docs/results-improvements.md), with raw JSON and CSV in
`results/`.

The focused million-token follow-up is in
[`docs/results-long-training.md`](docs/results-long-training.md), with its raw data in
`results/long_training.json` and `results/long_training.csv`.

The packed-size-matched scaling experiment is in
[`docs/results-size-matched.md`](docs/results-size-matched.md), with recovered summary data
in `results/size_matched_summary.csv`.

The 3-, 4-, and 5-level comparison is in
[`docs/results-quantization-levels.md`](docs/results-quantization-levels.md), with the exact
surviving aggregate and seed-level loss values in
`results/quantization_levels_recovered_summary.csv`. The original per-run JSON/CSV must be
regenerated because the temporary workspace was cleared before those files were committed.

The experiment uses WikiText-2's predefined train and validation splits. The test split is
downloaded but deliberately left untouched. Ternary weights are simulated during training;
packed inference size is an analytical estimate rather than the current checkpoint size.
