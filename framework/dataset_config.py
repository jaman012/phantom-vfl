"""
Dataset configuration: load and check the facts about one dataset.

A dataset config (datasets/<name>/dataset.json) holds every per-dataset fact
that the framework needs: file names, the label, the clients and their
features, and one type for each feature. The framework scripts contain no
column names of a dataset; they read them from this config.

Feature types:
- continuous: a number; used as it is.
- binary: two values. With a rule (positive_value, positive_name,
  negative_name), a text value that equals positive_value after strip() and
  upper() gets 1 and every other value gets 0. Without a rule, the two
  training values get the codes 0 and 1 in sorted order.
- nominal: categories with no order; one-hot encoded for the model.
- ordinal: categories with a true order; the config gives the order, and the
  codes 0..n-1 follow it. Phase 2 also checks its rank correlation (Spearman)
  with ordinal and continuous features of the same client.

Optional output_text: the dataset-specific sentences that the ground-truth
stage writes into its config.json files. A missing key gets its text from
DEFAULT_OUTPUT_TEXT.

Usage:
    from dataset_config import load_dataset_config
    cfg = load_dataset_config("datasets/adult_income/dataset.json")
"""

import json
from pathlib import Path

SCHEMA_VERSION = 1
FEATURE_TYPES = ("continuous", "binary", "nominal", "ordinal")
INTEGER_ENCODINGS = ("fit_on_train", "pre_encoded")
TEST_LABEL_FORMATS = ("text", "numeric")
BINARY_RULE_KEYS = ("positive_value", "positive_name", "negative_name")
FEATURE_SPEC_KEYS = {"type", "order", "description"} | set(BINARY_RULE_KEYS)
OUTPUT_TEXT_KEYS = ("ground_truth_preprocessing", "model_purpose", "model_source_split")

# Text for the config.json files of the ground-truth stage when the dataset
# config does not give it. model_source_split can use {split_folder} and {seed}.
DEFAULT_OUTPUT_TEXT = {
    "ground_truth_preprocessing": {"note": "Preparation by the dataset adapter; encoding from the dataset config"},
    "model_purpose": "Model form: one-hot encoding of the nominal features",
    "model_source_split": "{split_folder}/rs{seed}/split_indices (identical rows to the integer-form ground truth)",
}

# Phase 2 of the protocol knows three types. Binary and nominal features are
# checked as categorical. An ordinal feature takes the categorical marginal
# checks (chi-square on the level counts, total variation) and the categorical
# bivariate check of its pair (Cramer's V or ANOVA F); in a pair with an
# ordinal or continuous feature it also takes the Spearman check, which ranks
# the values by the configured order (ordinal_ranks).
PROTOCOL_TYPE = {
    "continuous": "continuous",
    "binary": "categorical",
    "nominal": "categorical",
    "ordinal": "ordinal",
}


class DatasetConfigError(ValueError):
    """A dataset config has a missing field or a wrong value."""


def _check(data):
    """Return a list of error messages for a dataset config (empty if valid)."""
    errors = []
    if not isinstance(data, dict):
        return ["the config must be a JSON object"]
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(data.get("name"), str) or not data.get("name"):
        errors.append("name must be a non-empty text")

    files = data.get("files")
    if not isinstance(files, dict):
        errors.append("files must be an object with cleaned_csv and train_csv")
    else:
        for key in ("cleaned_csv", "train_csv"):
            if not isinstance(files.get(key), str) or not files.get(key):
                errors.append(f"files.{key} must be a non-empty text")

    label = data.get("label")
    if not isinstance(label, dict):
        errors.append("label must be an object with column, map and test_label_format")
        label = {}
    else:
        if not isinstance(label.get("column"), str) or not label.get("column"):
            errors.append("label.column must be a non-empty text")
        label_map = label.get("map")
        if not isinstance(label_map, dict) or sorted(label_map.values()) != [0, 1]:
            errors.append("label.map must map exactly two label values to 0 and 1")
        if label.get("test_label_format") not in TEST_LABEL_FORMATS:
            errors.append(f"label.test_label_format must be one of {list(TEST_LABEL_FORMATS)}")

    integer_encoding = data.get("integer_encoding")
    if integer_encoding not in INTEGER_ENCODINGS:
        errors.append(f"integer_encoding must be one of {list(INTEGER_ENCODINGS)}")

    features = data.get("features")
    if not isinstance(features, dict) or not features:
        errors.append("features must be a non-empty object {feature: {type: ...}}")
        features = {}
    for name, spec in features.items():
        if not isinstance(spec, dict):
            errors.append(f"feature '{name}': the spec must be an object")
            continue
        unknown = set(spec) - FEATURE_SPEC_KEYS
        if unknown:
            errors.append(f"feature '{name}': unknown keys {sorted(unknown)}")
        ftype = spec.get("type")
        if ftype not in FEATURE_TYPES:
            errors.append(f"feature '{name}': type must be one of {list(FEATURE_TYPES)}")
            continue
        rule_keys = [k for k in BINARY_RULE_KEYS if k in spec]
        if rule_keys and ftype != "binary":
            errors.append(f"feature '{name}': {rule_keys} are allowed for type binary only")
        if ftype == "binary" and rule_keys:
            if len(rule_keys) != len(BINARY_RULE_KEYS) or not all(
                isinstance(spec[k], str) and spec[k] for k in BINARY_RULE_KEYS
            ):
                errors.append(f"feature '{name}': a binary rule needs all of {list(BINARY_RULE_KEYS)} as text")
            elif integer_encoding == "pre_encoded":
                errors.append(f"feature '{name}': a binary rule needs text values, "
                              f"so it is not allowed with integer_encoding 'pre_encoded'")
        if "order" in spec and ftype != "ordinal":
            errors.append(f"feature '{name}': order is allowed for type ordinal only")
        if ftype == "ordinal":
            order = spec.get("order")
            if not isinstance(order, list) or len(order) < 2:
                errors.append(f"feature '{name}': type ordinal needs an order with 2 or more levels")
            elif len({str(v) for v in order}) != len(order):
                errors.append(f"feature '{name}': the order has a repeated level")

    clients = data.get("clients")
    if not isinstance(clients, list) or len(clients) < 2 or not all(
        isinstance(c, list) and c and all(isinstance(f, str) for f in c) for c in clients
    ):
        errors.append("clients must be a list of 2 or more non-empty lists of feature names")
        clients = []
    seen = {}
    for index, client in enumerate(clients):
        for feature in client:
            if feature in seen:
                errors.append(f"feature '{feature}' is in client {seen[feature]} and client {index}")
            seen[feature] = index
            if feature not in features:
                errors.append(f"feature '{feature}' of client {index} has no type in features")
    for feature in features:
        if clients and feature not in seen:
            errors.append(f"feature '{feature}' has a type but is in no client")
    if isinstance(label, dict) and label.get("column") in features:
        errors.append("the label column must not be a feature")

    order = data.get("model_feature_order")
    if order is not None:
        if not isinstance(order, list) or sorted(order) != sorted(features) or len(set(order)) != len(order):
            errors.append("model_feature_order must list every feature exactly once")

    output_text = data.get("output_text")
    if output_text is not None:
        errors.extend(_check_output_text(output_text))
    return errors


def _check_output_text(output_text):
    """Return a list of error messages for the optional output_text object."""
    if not isinstance(output_text, dict):
        return ["output_text must be an object"]
    errors = []
    unknown = set(output_text) - set(OUTPUT_TEXT_KEYS)
    if unknown:
        errors.append(f"output_text: unknown keys {sorted(unknown)}")
    preprocessing = output_text.get("ground_truth_preprocessing")
    if preprocessing is not None and not (
        isinstance(preprocessing, dict) and preprocessing
        and all(isinstance(k, str) and isinstance(v, str) for k, v in preprocessing.items())
    ):
        errors.append("output_text.ground_truth_preprocessing must be a non-empty object of texts")
    for key in ("model_purpose", "model_source_split"):
        value = output_text.get(key)
        if value is not None and (not isinstance(value, str) or not value):
            errors.append(f"output_text.{key} must be a non-empty text")
    source_split = output_text.get("model_source_split")
    if isinstance(source_split, str) and source_split:
        try:
            source_split.format(split_folder="split", seed=0)
        except (KeyError, IndexError, ValueError):
            errors.append("output_text.model_source_split can use only {split_folder} and {seed}")
    return errors


class DatasetConfig:
    """The checked facts of one dataset (see the module docstring)."""

    def __init__(self, data, source="<dict>"):
        self.source = str(source)
        errors = _check(data)
        if errors:
            raise DatasetConfigError(
                f"Dataset config {self.source} is not valid:\n  - " + "\n  - ".join(errors)
            )
        self.name = data["name"]
        self.output_name = data.get("output_name", data["name"])
        self.cleaned_csv = data["files"]["cleaned_csv"]
        self.train_csv = data["files"]["train_csv"]
        self.label_column = data["label"]["column"]
        self.label_map = dict(data["label"]["map"])
        self.test_label_format = data["label"]["test_label_format"]
        self.integer_encoding = data["integer_encoding"]
        self.features = dict(data["features"])
        self.clients = [list(c) for c in data["clients"]]
        self.model_feature_order = list(data.get("model_feature_order") or self._default_model_order())
        self.output_text = {**DEFAULT_OUTPUT_TEXT, **(data.get("output_text") or {})}

    def _default_model_order(self):
        """Continuous, then binary, then ordinal, then nominal, each in config order."""
        return [f for t in ("continuous", "binary", "ordinal", "nominal")
                for f, spec in self.features.items() if spec["type"] == t]

    @property
    def n_clients(self):
        return len(self.clients)

    def feature_type(self, feature):
        return self.features[feature]["type"]

    def features_of_type(self, ftype):
        """Features of one type, in config order."""
        return [f for f, spec in self.features.items() if spec["type"] == ftype]

    def binary_rule(self, feature):
        """(positive_value, positive_name, negative_name) of a binary feature, or None."""
        spec = self.features[feature]
        if spec["type"] == "binary" and "positive_value" in spec:
            return spec["positive_value"], spec["positive_name"], spec["negative_name"]
        return None

    def ordinal_order(self, feature):
        return list(self.features[feature]["order"])

    def ordinal_ranks(self, feature):
        """
        {integer-form value: rank} of an ordinal feature, as floats (the
        protocol reads the values as floats). The rank is the position in the
        configured order. 'fit_on_train': the integer code is the position;
        'pre_encoded': the integer form keeps the values of the cleaned data,
        so each configured level (a number) gets its position.
        """
        order = self.ordinal_order(feature)
        if self.integer_encoding == "pre_encoded":
            return {float(level): float(position) for position, level in enumerate(order)}
        return {float(position): float(position) for position in range(len(order))}

    def nominal_in_model_order(self):
        return [f for f in self.model_feature_order if self.feature_type(f) == "nominal"]

    def client_of(self, feature):
        for index, client in enumerate(self.clients):
            if feature in client:
                return index
        raise KeyError(feature)

    def protocol_feature_types(self):
        """{feature: 'continuous', 'categorical' or 'ordinal'} for Phase 2 of the protocol."""
        return {f: PROTOCOL_TYPE[spec["type"]] for f, spec in self.features.items()}


def load_dataset_config(path):
    """Read and check a dataset config file. Stop with an error if it is not valid."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset config not found: {path}")
    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as error:
            raise DatasetConfigError(f"Dataset config {path} is not valid JSON: {error}") from error
    return DatasetConfig(data, source=path)
