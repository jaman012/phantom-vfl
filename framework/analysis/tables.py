"""
Tables of the statistical tests: complete seeds only.

A seed is complete when it has a used run for every k level of the plan. The
tests compare the k levels within each seed, so a seed with a missing k level
is left out of every table. The rows are keyed by seed, never by position.
"""

import pandas as pd


def complete_seeds(selection, seeds, configs):
    """(complete seeds in plan order, {left-out seed: missing k levels})."""
    used = {seed: set() for seed in seeds}
    for row in selection.itertuples():
        used[int(row.seed)].add(row.config)
    complete = [seed for seed in seeds if set(configs) <= used[seed]]
    missing = {seed: [name for name in configs if name not in used[seed]]
               for seed in seeds if seed not in complete}
    return complete, missing


def wide_table(selection, metric, configs, seeds):
    """One row for each seed (in the given order), one column for each k level (in the given order)."""
    part = selection[selection["seed"].isin(seeds)]
    wide = part.pivot(index="seed", columns="config", values=metric)
    wide = wide.reindex(index=seeds, columns=configs)
    if wide.isna().any().any():
        raise ValueError(f"the wide table of {metric} has an empty cell")
    wide = wide.reset_index()
    wide.columns.name = None
    return wide


def long_table(wide, configs):
    """Columns seed, config, value; k levels in the given order, seeds in the order of the wide table."""
    rows = [{"seed": int(seed), "config": name, "value": wide.loc[index, name]}
            for name in configs for index, seed in wide["seed"].items()]
    return pd.DataFrame(rows, columns=["seed", "config", "value"])
