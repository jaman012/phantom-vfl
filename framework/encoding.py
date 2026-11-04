"""
Encoding: the integer form (internal) and the model form of the datasets.

Two forms exist:
- Integer form (internal). The split builds the maps from the training data
  and saves them as encoding_maps.json. The ground-truth stage and the phantom
  generation load these maps. The phantom protocol works on this form.
  A dataset with integer_encoding 'pre_encoded' is already integer-coded by its
  cleaning script; its maps are empty and its integer form is its raw form.
- Model form. Nominal features are one-hot encoded; continuous, binary and
  ordinal features keep their integer-form values. The ground-truth stage fits
  the one-hot columns on the ground truth and applies them to the test set.
  The phantom generation applies the same columns to each k level.

The features are saved unscaled. One StandardScaler is fitted for each client
on the ground truth and saved; the training code applies it. The setting
scaler_columns selects the columns that the scaler changes: 'continuous_ordinal'
(default) scales the continuous and ordinal columns, and the binary columns,
the one-hot columns and the integer codes of nominal features pass through
(mean 0, scale 1 in the scaler); 'all' scales every column.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

NEAR_ZERO_VARIANCE = 1e-3  # a column with a smaller standard deviation is reported before scaling
SCALER_COLUMNS = ("continuous_ordinal", "all")  # values of the setting scaler_columns
DEFAULT_SCALER_COLUMNS = "continuous_ordinal"
SCALED_TYPES = ("continuous", "ordinal")  # feature types that 'continuous_ordinal' scales


# ============================================================================
# INTEGER FORM
# ============================================================================

def build_integer_maps(train_df, cfg):
    """
    Build the category-to-code maps from the training data only.

    Order of the maps: binary features with a rule, then the other
    non-continuous features in config order, then the label.
    Rules: a binary rule gives {positive_name: 1, negative_name: 0}; an ordinal
    feature gets the codes of its configured order; every other feature gets
    the codes 0..n-1 in sorted order of its training values.

    Args:
        train_df: training rows with the original values
        cfg: DatasetConfig

    Returns:
        {column: {value as text: code}}; empty for a 'pre_encoded' dataset
    """
    if cfg.integer_encoding == "pre_encoded":
        return {}
    maps = {}
    for feature in cfg.features:
        rule = cfg.binary_rule(feature)
        if rule:
            maps[feature] = {rule[1]: 1, rule[2]: 0}
    for feature, spec in cfg.features.items():
        ftype = spec["type"]
        if ftype == "continuous" or cfg.binary_rule(feature):
            continue
        if ftype == "ordinal":
            levels = [str(value) for value in spec["order"]]
            unknown = sorted(set(train_df[feature].astype(str).unique()) - set(levels))
            if unknown:
                raise ValueError(f"Ordinal feature '{feature}' has training values that are not "
                                 f"in its configured order: {unknown}")
            maps[feature] = {level: code for code, level in enumerate(levels)}
            continue
        categories = sorted(train_df[feature].unique())
        if ftype == "binary" and len(categories) != 2:
            raise ValueError(f"Binary feature '{feature}' has {len(categories)} values in the "
                             f"training data; it needs exactly 2 (or a binary rule)")
        maps[feature] = {str(category): code for code, category in enumerate(categories)}
    maps[cfg.label_column] = dict(cfg.label_map)
    return maps


def apply_integer_maps(df, maps, cfg):
    """
    Return a copy of df in the integer form.

    Strict rule: a value that is not in the map of its feature stops the run.
    A binary rule needs no check, because every value that is not the
    positive value gets 0. The label column does not change.
    A 'pre_encoded' dataset is returned unchanged.
    """
    out = df.copy()
    if cfg.integer_encoding == "pre_encoded":
        return out
    rule_features = []
    for feature, spec in cfg.features.items():
        if spec["type"] == "continuous":
            continue
        if cfg.binary_rule(feature):
            rule_features.append(feature)
            continue
        column_map = maps[feature]
        values = out[feature].astype(str)
        unseen = sorted(set(values.unique()) - set(column_map))
        if unseen:
            raise ValueError(f"Column '{feature}' has values that are not in the training data: {unseen}")
        out[feature] = values.map(column_map).astype(int)
    for feature in rule_features:
        positive_value = cfg.binary_rule(feature)[0]
        out[feature] = (out[feature].astype(str).str.strip().str.upper() == positive_value).astype(int)
    return out


def save_integer_maps(maps, path):
    """Write the maps as JSON (indent 2, map order kept)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(maps, f, indent=2)


def load_integer_maps(path):
    """Read the maps that the split saved. Stop if the file is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Encoding maps not found: {path}\n"
                                f"Run split_train_test.py with the same seed first.")
    with open(path) as f:
        return json.load(f)


# ============================================================================
# MODEL FORM
# ============================================================================

def base_feature(column, cfg):
    """The feature that a model-form column comes from (a one-hot column is '<feature>_<level>')."""
    if column in cfg.features:
        return column
    matches = [f for f in cfg.nominal_in_model_order() if column.startswith(f"{f}_")]
    if not matches:
        raise KeyError(f"Model column '{column}' does not belong to a feature of the dataset config")
    return max(matches, key=len)


def _ordinal_codes_pre_encoded(series, order):
    """Codes 0..n-1 of a pre-encoded ordinal feature, by the numeric value of each level."""
    position = {float(level): code for code, level in enumerate(order)}
    codes = series.astype(float).map(position)
    if codes.isna().any():
        raise ValueError(f"Ordinal values not in the configured order: {sorted(series[codes.isna()].unique())}")
    return codes.astype(int)


def _non_nominal_model_values(int_df, cfg):
    """Integer-form values of the non-nominal features, as the model uses them."""
    out = int_df.copy()
    if cfg.integer_encoding == "pre_encoded":
        for feature in cfg.features_of_type("ordinal"):
            if feature in out.columns:  # the features of one client only (stage 3)
                out[feature] = _ordinal_codes_pre_encoded(out[feature], cfg.ordinal_order(feature))
    return out


def encode_model_from_raw(train_raw, test_raw, maps, cfg):
    """
    Fit the model form on the training rows and apply it to the test rows.

    One-hot columns are '<feature>_<value>' for each training value (pandas
    get_dummies, sorted values). A test value that the training data does not
    have gets no column; a training value that the test data does not have
    gives a column of zeros in the test set.

    Args:
        train_raw, test_raw: rows with the original values
        maps: integer maps (build_integer_maps or load_integer_maps)
        cfg: DatasetConfig

    Returns:
        (train_model, test_model): DataFrames in cfg.model_feature_order, one-hot
        columns expanded in place
    """
    nominal = cfg.nominal_in_model_order()
    train_dummies = pd.get_dummies(train_raw[nominal], columns=nominal, prefix=nominal).astype(int)
    test_dummies = pd.get_dummies(test_raw[nominal], columns=nominal, prefix=nominal).astype(int)
    test_dummies = test_dummies.reindex(columns=train_dummies.columns, fill_value=0)

    def assemble(raw, dummies):
        values = _non_nominal_model_values(apply_integer_maps(raw, maps, cfg), cfg)
        parts = []
        for feature in cfg.model_feature_order:
            if cfg.feature_type(feature) == "nominal":
                columns = [c for c in dummies.columns if base_feature(c, cfg) == feature]
                parts.append(dummies[columns].reset_index(drop=True))
            else:
                parts.append(values[[feature]].reset_index(drop=True))
        return pd.concat(parts, axis=1)

    return assemble(train_raw, train_dummies), assemble(test_raw, test_dummies)


def client_model_columns(model_columns, cfg):
    """Split the model-form columns by client; each list keeps the model order."""
    clients = [[] for _ in range(cfg.n_clients)]
    for column in model_columns:
        clients[cfg.client_of(base_feature(column, cfg))].append(column)
    return clients


def read_model_columns(model_dir, cfg):
    """Read the model-form columns of each client from the ground truth (file headers)."""
    columns = []
    for client_index in range(cfg.n_clients):
        path = Path(model_dir) / f"client_{client_index}" / "features.csv"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run the ground-truth stage for this seed first.")
        columns.append(pd.read_csv(path, nrows=0).columns.tolist())
    return columns


def encode_model_from_integer(int_df, client_index, model_columns, maps, cfg, context=""):
    """
    Encode the integer-form features of one client (for example a phantom k
    level) to the model columns of the ground truth.

    A one-hot column is 1 where the value equals the column's level: through the
    inverse integer map for a text-coded feature, or as a number for a
    'pre_encoded' feature. A value that the ground truth does not have is
    reported and gets no column.

    Args:
        int_df: integer-form features of one client
        client_index: index of the client in cfg.clients
        model_columns: the client's model columns (read_model_columns)
        maps: integer maps
        cfg: DatasetConfig
        context: text for the report, for example 'k20/client_0'

    Returns:
        DataFrame with exactly model_columns, in that order
    """
    client_features = cfg.clients[client_index]
    nominal_here = [f for f in client_features if cfg.feature_type(f) == "nominal"]
    encoded = _non_nominal_model_values(
        int_df[[f for f in client_features if f not in nominal_here]], cfg
    )
    for feature in nominal_here:
        prefix = f"{feature}_"
        level_columns = [c for c in model_columns if base_feature(c, cfg) == feature]
        if cfg.integer_encoding == "pre_encoded" or feature not in maps:
            values = int_df[feature].astype(float)
            levels = {c: float(c[len(prefix):]) for c in level_columns}
        else:
            inverse = {code: level for level, code in maps[feature].items()}
            values = int_df[feature].map(inverse)
            levels = {c: c[len(prefix):] for c in level_columns}
        unexpected = set(values.unique()) - set(levels.values())
        if unexpected:
            print(f"  [!] {context}: '{feature}' has value(s) {unexpected} that the ground truth "
                  f"does not have; they get no one-hot column.")
        for column, level in levels.items():
            encoded[column] = (values == level).astype(int)
    return encoded.reindex(columns=list(model_columns), fill_value=0)


# ============================================================================
# SCALERS
# ============================================================================

def low_variance_columns(client_df, threshold=NEAR_ZERO_VARIANCE):
    """{column: standard deviation} of the columns whose deviation is below threshold."""
    stds = client_df.std()
    return stds[stds < threshold].to_dict()


def check_scaler_columns(value):
    """Stop if value is not a value of the setting scaler_columns."""
    if not isinstance(value, str) or value not in SCALER_COLUMNS:
        raise ValueError(f"scaler_columns must be one of {list(SCALER_COLUMNS)}, got {value!r}")
    return value


def scaled_columns(columns, cfg, scaler_columns=DEFAULT_SCALER_COLUMNS):
    """
    The columns that the scaler changes, in the order of columns.

    'all': every column. 'continuous_ordinal': the columns of continuous and
    ordinal features. columns can be integer-form columns (feature names) or
    model-form columns (a one-hot column belongs to its nominal feature).
    """
    check_scaler_columns(scaler_columns)
    if scaler_columns == "all":
        return list(columns)
    return [c for c in columns if cfg.feature_type(base_feature(c, cfg)) in SCALED_TYPES]


def fit_scaler(client_df, scale_columns=None):
    """
    Fit a StandardScaler on the values of one client (no column names kept).

    scale_columns: the columns that the scaler changes; None: every column.
    For each other column the scaler holds mean 0, variance 1 and scale 1, so
    transform() returns its values unchanged. The column order and the number
    of columns of the scaler are the order and the number of client_df.
    """
    scaler = StandardScaler()
    scaler.fit(client_df.values)
    if scale_columns is not None:
        keep = set(scale_columns)
        unknown = sorted(keep - set(client_df.columns))
        if unknown:
            raise ValueError(f"scale_columns has columns that the client does not have: {unknown}")
        passthrough = [i for i, column in enumerate(client_df.columns) if column not in keep]
        if passthrough:
            scaler.mean_[passthrough] = 0.0
            scaler.var_[passthrough] = 1.0
            scaler.scale_[passthrough] = 1.0
    return scaler


def save_scaler(scaler, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
