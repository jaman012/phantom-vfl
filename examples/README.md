# Example run configs

Run every command from the root of the repository. The paths in the configs are relative to the folder of the config. Each config writes its data and results to `runs/` in the root of the repository. `RUNNING.md`, section 24, has the details.

| File | Content | Flower runs | Time (tested computer) |
|---|---|---|---|
| `first_test.json` | the quick start with 20 rounds instead of 250: a test that every part runs | 30 | about 8 minutes |
| `quickstart_adult_income_ls.json` | Adult Income, label sharing on, loss `bce`, seeds 42, 50 and 58 | 30 | about 40 minutes |
| `full_design/*.json` | the six combinations of dataset, label sharing and loss, 10 seeds each | 600 | about 13 hours |
| `full_design/summary_config.json` | the summary of the six analyses | – | under a minute |

All configs use the split 80-20 (no validation set) and the training values from the tuning studies (`RUNNING.md`, section 18). All configs except `first_test.json` use 250 rounds.

## First test

Start here.

```bash
python framework/run_all.py --config examples/first_test.json
```

It runs the same chain as the quick start, with 20 rounds instead of 250. It checks that every part runs on your computer. Its numbers are not results: the models train for less than a tenth of the tuned rounds.

## Quick start

```bash
python framework/run_all.py --config examples/quickstart_adult_income_ls.json
```

The quick start uses the tuned values. It is the check of a new installation (`RUNNING.md`, section 25).

## Full design

```bash
python framework/run_all.py \
  --config examples/full_design/adult_income_bce_ls.json \
  --config examples/full_design/adult_income_bce_nls.json \
  --config examples/full_design/bank_marketing_weighted_bce_ls.json \
  --config examples/full_design/bank_marketing_focal_ls.json \
  --config examples/full_design/bank_marketing_weighted_bce_nls.json \
  --config examples/full_design/bank_marketing_focal_nls.json \
  --summary examples/full_design/summary_config.json
```

The full design repeats the structure of the published experiments: the datasets, label sharing, the losses, the k levels, the seed numbers and the split. It does not repeat their numbers. This code seeds the phantom generation and the start weights of the models, and it has fixes that came after the published results (`RUNNING.md`, section 1).

## How to cut the time

- **Fewer seeds:** change `seeds` in `generation` and in `training`. The analysis needs at least 3 seeds.
- **Fewer rounds:** change `num-server-rounds` in `training.run_config`. The tuned value is 250.
- **Fewer k levels:** change `generation.k_levels`.
- **Some steps only:** `--steps` (for example `--steps generation,training`). After a stop, start the same command again with `--resume`.
