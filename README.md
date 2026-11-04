# Phantom VFL

Phantom VFL is a framework for experiments with phantom individuals in vertical federated learning (VFL). For each seed it makes a set of vertically partitioned datasets for two or more clients:

- **the ground truth:** no phantom protocol, no removal; all individuals are real;
- **k\*:** the phantom rate that the protocol derives from the data with statistical checks (folder `k0`);
- **k = 5 % to 40 %:** imposed phantom rates.

It trains a vertical federated model with [Flower](https://flower.ai) on each of these datasets, and tests each phantom level against the ground truth of the same seed (paired tests in Python, a repeated-measures ANOVA with Tukey tests in R). One command runs the whole chain.

## Requirements

- Python 3.10 and the packages of `requirements.txt` (pinned to the tested versions).
- For the repeated-measures tests (Track B of the analysis): R with the R package emmeans. Without R, set `analysis.track_b` to false.
- The example datasets are downloaded from the Hugging Face Hub, so the first run needs network access.
- The training runs on the CPU.

```bash
conda create -n phantom-vfl python=3.10
conda activate phantom-vfl
python -m pip install -r requirements.txt
```

In R: `install.packages("emmeans")`.

## First steps

From the root of the repository, in this order:

1. **First test** (about 8 minutes). It checks that every part runs on your computer. Its numbers are not results. The results are in `runs/first_test_adult_income_ls/`.

   ```bash
   python framework/run_all.py --config examples/first_test.json
   ```

2. **Quick start** (about 40 minutes on the tested computer, Apple M4 Pro). It runs the same chain with the tuned 250 rounds. It is the check of a new installation (`RUNNING.md`, section 25).

   ```bash
   python framework/run_all.py --config examples/quickstart_adult_income_ls.json
   ```

   It runs the Adult Income dataset with label sharing, 3 seeds and 10 dataset levels (the ground truth, k\* and 8 imposed rates): 30 Flower runs. The results are in `runs/quickstart_adult_income_ls/`; start with `08_results_analysis/bce/split_80_20_onehot/report.md`.

3. **Full design** (two datasets, label sharing on and off, three losses, 10 seeds; about 13 hours): [`examples/README.md`](examples/README.md). It also tells how to cut the run time.

To write a run config of your own with questions:

```bash
python framework/run_all.py --new
```

## Documentation

- [`RUNNING.md`](RUNNING.md): the full guide: each stage, the settings, the files, the analysis and the known limits.
- [`docs/flower_reference.md`](docs/flower_reference.md): the source of the Flower app, every change to it, and its known limits.
- A new dataset: write a cleaned CSV and a `dataset.json` (`RUNNING.md`, sections 6 and 7). The scripts `datasets/<name>/prepare.py` are worked examples of the cleaning.

## Background

The method comes from the dissertation *Phantom Vertical Federated Learning: A Strategic Distribution-Preserving Privacy Approach* (James Mansueto, M.Sc. in Artificial Intelligence, University of Malta, 2026).

## Licence

Copyright 2026 James Mansueto. Licensed under the Apache License 2.0. The full text is in [`LICENSE`](LICENSE).

The Flower app in `flower_app/` is a modified version of the Flower example `examples/vertical-fl` (Apache License 2.0; see `flower_app/LICENSE`, `flower_app/ATTRIBUTION.md` and `flower_app/MODIFICATIONS.md`).
