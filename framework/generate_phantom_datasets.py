"""
PHANTOM VFL STAGE 3: GENERATE THE PHANTOM DATASETS

For one dataset, one seed and one label-sharing setting, this script:
1. Loads the integer-form ground truth of the seed (stage 2) and checks it
   against the training split (data_io). A 'fit_on_train' dataset uses the
   integer maps that the split saved.
2. Writes a copy of the maps (encoding_maps.json) to the seed folder of the
   phantom datasets ({} for a 'pre_encoded' dataset).
3. Splits the features over the clients as the dataset config says.
4. Runs Phases 0 to 2 of the protocol once (the natural budget k*).
5. Runs Phases 3 to 4c for each k level: k* (folder --kstar-folder, default k0)
   and the imposed k levels (whole percentages, folders k<percent>).
6. Saves the datasets of each k level, ready for Flower VFL training, and a
   completion marker. With --resume, a complete k folder is skipped, and
   Phases 0 to 2 run only if at least one k level runs.
7. Writes the model form (one-hot) of every requested k level, with the model
   columns of the model-form ground truth (stage 2). This step runs every
   time, also for the k folders that --resume skipped.

Before step 1, the stage checks the k folders of the seed folder (one data
folder for each label-sharing setting): it stops before it
writes a file if a k folder was made with the other label-sharing setting,
and it warns about a k folder whose setting is unknown.

Usage:
    python generate_phantom_datasets.py --dataset-config <dataset.json> \
        --data-dir <data folder> --seed 42 --label-sharing on [--resume] \
        [--test-share 0.2] [--validation-share 0.0]

The shares give the split folder names; they must be the shares of stage 1.

The runner calls run() with the same settings (GenerationSettings).
When the script runs alone from its command line, its screen output also goes
to a log file and a warnings file in <data folder>/logs/ (run_log.py).
"""

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

FRAMEWORK_DIR = Path(__file__).resolve().parent
if str(FRAMEWORK_DIR) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_DIR))

import data_io  # noqa: E402
from dataset_config import DatasetConfigError, load_dataset_config  # noqa: E402
from run_log import run_log, set_context  # noqa: E402
from encoding import encode_model_from_integer, read_model_columns, save_integer_maps  # noqa: E402
from phantom_protocol import PhantomVFLConfig, prepare_protocol, run_k_level  # noqa: E402

KSTAR = "kstar"
DEFAULT_K_LEVELS = [KSTAR, 5, 10, 15, 20, 25, 30, 35, 40]
DEFAULT_KSTAR_FOLDER = "k0"
# Framework modules whose code decides the content of the k folders (with the
# protocol and this script); their checksums go into the completion marker
FRAMEWORK_MODULES = ("data_io.py", "encoding.py", "dataset_config.py")


DEFAULT_PHASE2_TRIALS = 1  # Phase 2: random removals for each tested budget (odd; majority vote)


def check_phase2_trials(value):
    """Stop if phase2_trials is not an odd whole number (1, 3, 5, ...)."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1 or value % 2 == 0:
        raise ValueError(f"phase2_trials must be an odd whole number (1, 3, 5, ...), got {value!r}")
    return value


@dataclass
class GenerationSettings:
    """Settings of one run of this stage (the runner fills them from run_config.json)."""
    dataset_config: str
    data_dir: str
    seed: int
    label_sharing: bool
    k_levels: list = field(default_factory=lambda: list(DEFAULT_K_LEVELS))
    kstar_folder: str = DEFAULT_KSTAR_FOLDER
    test_share: float = data_io.DEFAULT_TEST_SHARE
    validation_share: float = data_io.DEFAULT_VALIDATION_SHARE
    resume: bool = False
    cramers_v_yates: bool = False
    phase2_trials: int = DEFAULT_PHASE2_TRIALS


# ============================================================================
# K LEVELS
# ============================================================================

def k_levels_with_folders(k_levels, kstar_folder=DEFAULT_KSTAR_FOLDER):
    """
    [(k_value, folder name, is_kstar)] for the k levels, in the given order.

    'kstar' gives (0.0, kstar_folder, True): the natural budget from Phase 2.
    A whole percentage p from 1 to 99 gives (p / 100, 'k<p>', False).
    Any other value, or two levels with the same folder, stops the run.
    """
    if not kstar_folder or "/" in kstar_folder or "\\" in kstar_folder:
        raise ValueError(f"kstar_folder {kstar_folder!r} is not a folder name")
    levels = []
    for level in k_levels:
        text = str(level).strip()
        if text.lower() == KSTAR:
            levels.append((0.0, kstar_folder, True))
            continue
        if isinstance(level, bool) or not re.fullmatch(r"\d+", text):
            raise ValueError(f"k level {level!r} is not '{KSTAR}' or a whole percentage")
        percent = int(text)
        if not 1 <= percent <= 99:
            raise ValueError(f"k level {percent} is outside 1 to 99 (percent)")
        levels.append((percent / 100, f"k{round(percent)}", False))
    if not levels:
        raise ValueError("No k level given")
    folders = [folder for _, folder, _ in levels]
    duplicates = sorted({f for f in folders if folders.count(f) > 1})
    if duplicates:
        raise ValueError(f"Two k levels use the same folder: {duplicates}")
    return levels


# ============================================================================
# CLIENTS
# ============================================================================

def build_parties(df_int, cfg):
    """
    Split the integer-form training rows over the clients of the dataset config.

    Returns:
        vertical_splits: {client name: feature list} (Client_A, Client_B, ...)
        labels_dict: {person_<row index>: label code}
        client_data_dict: {client name: {person_<row index>: {feature: value}}}
    """
    vertical_splits = {f"Client_{chr(65 + i)}": list(features) for i, features in enumerate(cfg.clients)}

    # IDs from the row index; every value as a float
    ids = [f"person_{idx}" for idx in df_int.index]
    label_values = df_int[cfg.label_column].to_numpy(dtype=float)
    labels_dict = {id_: float(value) for id_, value in zip(ids, label_values)}

    client_data_dict = {}
    for party_id, features in vertical_splits.items():
        values = df_int[features].to_numpy(dtype=float)
        client_data = {id_: {feat: float(value) for feat, value in zip(features, row)}
                       for id_, row in zip(ids, values)}
        client_data_dict[party_id] = client_data
        print(f"  {party_id}: {len(client_data)} individuals x {len(features)} features: {features}")
    return vertical_splits, labels_dict, client_data_dict


# ============================================================================
# SAVE THE DATASETS OF ONE K LEVEL
# ============================================================================

def save_phantom_datasets(k_value, k_dir, results, vertical_splits, dataset_name,
                          phase2_trials=DEFAULT_PHASE2_TRIALS):
    """
    Save phantom datasets to disk in format ready for Flower VFL.
    
    
    Creates for each k-value:
    - training_order.json
    - labels.csv
    - phantom_rate_summary.json
    - ground_truth_analysis.csv
    - client_0/
        - features.csv
        - phantom_mapping.json
        - metadata.json
    - client_1/
        - features.csv
        - phantom_mapping.json
        - metadata.json
    - config.json
    - budget_computation_history.json
    - _complete.json (completion marker, written by run() as the last file)
    
    Args:
        k_value: Phantom proportion (0.0 for k*)
        k_dir: Output folder of this k level
        results: Protocol results of this k level (run_k_level)
        vertical_splits: {client name: feature list}
        dataset_name: Dataset name written to config.json
        phase2_trials: Phase 2 trials of the run; config.json gets the key
            phase2_trials only if the value is not 1 (with 1 the file does not
            change; the completion marker always holds the value)
    """
    print("\n" + "=" * 80)
    print(f"SAVING PHANTOM DATASETS (k={k_value})")
    print("=" * 80)
    
    # Create output directory
    k_dir = Path(k_dir)
    
    k_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {k_dir}")
    
    # Extract server and clients from results
    if results['mode'] == 'experimental' and k_value > 0.0:
        k_results = results['experimental_results'][0]
        server = k_results['server']
        clients = k_results['clients']
    else:
        server = results['server']
        clients = results['clients']
    
    # ========================================================================
    # Save training order
    # ========================================================================
    training_order_path = k_dir / "training_order.json"
    with open(training_order_path, 'w') as f:
        json.dump(server.training_order, f, indent=2)
    print(f"\n✓ Saved training order: {len(server.training_order)} IDs")
    
    # ========================================================================
    # Save labels (aligned to training order)
    # ========================================================================
    labels = [server.labels[id_] for id_ in server.training_order]
    labels_df = pd.DataFrame({'label': labels})
    labels_path = k_dir / "labels.csv"
    labels_df.to_csv(labels_path, index=False)
    print(f"✓ Saved labels: {len(labels)} samples")
    
    # ========================================================================
    # Save each client's data
    # ========================================================================
    client_ids = list(vertical_splits.keys())
    
    for i, (client, client_id) in enumerate(zip(clients, client_ids)):
        print(f"\n✓ Saving {client_id} data...")
        
        # Create client directory
        client_dir = k_dir / f"client_{i}"
        client_dir.mkdir(exist_ok=True)
        
        # ====================================================================
        # 1. Features (aligned to training order with phantom substitutions)
        # ====================================================================
        feature_data = []
        for position, real_id in enumerate(server.training_order):
            # Check if client uses phantom mapping at this position
            if position in client.phantom_mapping:
                actual_id = client.phantom_mapping[position]
            else:
                actual_id = real_id
            
            # Get feature values
            features = client.data[actual_id]
            feature_data.append(features)
        
        # Save features
        features_df = pd.DataFrame(feature_data)
        features_path = client_dir / "features.csv"
        features_df.to_csv(features_path, index=False)
        print(f"  - Features: {features_df.shape}")
        
        # ====================================================================
        # 2. Phantom mapping
        # ====================================================================
        mapping_str = {str(k): v for k, v in client.phantom_mapping.items()}
        mapping_path = client_dir / "phantom_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(mapping_str, f, indent=2)
        print(f"  - Phantom mapping: {len(client.phantom_mapping)} positions")

        # ====================================================================
        # 3. Metadata
        # ====================================================================
        metadata = {
            "client_id": client.client_id,
            "party_name": client_id,
            "num_features": len(client.features),
            "feature_names": client.features,
            "feature_types": client.feature_types,
            "num_training_samples": len(server.training_order),
            "num_phantom_positions": len(client.phantom_positions),
            "phantom_rate": len(client.phantom_positions) / len(server.training_order)
        }
        metadata_path = client_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  - Phantom rate: {100*metadata['phantom_rate']:.1f}%")
    
    # ========================================================================
    # Save overall configuration
    # ========================================================================
    overall_config = {
        "dataset": dataset_name,
        "phantom_proportion": k_value,
        "num_clients": len(clients),
        "num_training_samples": len(server.training_order),
        "psi_common_ids": results['psi']['common_ids'],
        "natural_phantom_budget": results['budget'].get('natural_budget', 0),
        "generation_seed": results.get('random_seed'),
        "vertical_splits": vertical_splits
    }
    if phase2_trials != 1:
        overall_config["phase2_trials"] = phase2_trials
    config_path = k_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(overall_config, f, indent=2)
    print(f"\n✓ Saved configuration")
    
    # ========================================================================
    # Save budget computation history (includes bivariate search history)
    # ========================================================================
    if 'budget' in results and 'histories' in results['budget']:
        # Convert any non-serializable objects
        histories_serializable = []
        for history in results['budget']['histories']:
            history_copy = {}
            for key, value in history.items():
                if key == 'bivariate_search_history':
                    # Convert tuple pairs to strings for JSON
                    search_history = []
                    for entry in value:
                        entry_copy = {}
                        for entry_key, entry_val in entry.items():
                            if entry_key == 'failed_pairs':
                                # Convert tuples like ('age', 'education') to strings
                                entry_copy[entry_key] = [f"{p[0][:3]}_{p[1][:3]}" for p in entry_val]
                            elif entry_key == 'all_valid':
                                # Convert numpy bool to Python bool
                                entry_copy[entry_key] = bool(entry_val)
                            elif isinstance(entry_val, (np.bool_, np.integer)):
                                # Convert any numpy types to Python types
                                entry_copy[entry_key] = entry_val.item()
                            else:
                                entry_copy[entry_key] = entry_val
                        search_history.append(entry_copy)
                    history_copy[key] = search_history
                elif key == 'bivariate_metrics':
                    # Convert tuple keys to strings and handle numpy types
                    if value:
                        metrics_serializable = {}
                        for k, v in value.items():
                            key_str = f"{k[0][:3]}_{k[1][:3]}"
                            # Recursively convert nested dict values
                            metric_dict = {}
                            for mk, mv in v.items():
                                if isinstance(mv, (np.bool_, bool)):
                                    metric_dict[mk] = bool(mv)
                                elif isinstance(mv, (np.integer, np.floating)):
                                    metric_dict[mk] = mv.item()
                                else:
                                    metric_dict[mk] = mv
                            metrics_serializable[key_str] = metric_dict
                        history_copy[key] = metrics_serializable
                elif key == 'search_histories':
                    # Handle marginal search histories
                    search_hist_serializable = {}
                    for feat, hist_list in value.items():
                        serialized_list = []
                        for hist_entry in hist_list:
                            hist_entry_copy = {}
                            for hk, hv in hist_entry.items():
                                if isinstance(hv, (np.bool_, bool)):
                                    hist_entry_copy[hk] = bool(hv)
                                elif isinstance(hv, (np.integer, np.floating)):
                                    hist_entry_copy[hk] = hv.item()
                                elif isinstance(hv, dict):
                                    # Handle nested metrics dict
                                    nested_dict = {}
                                    for nk, nv in hv.items():
                                        if isinstance(nv, (np.bool_, bool)):
                                            nested_dict[nk] = bool(nv)
                                        elif isinstance(nv, (np.integer, np.floating)):
                                            nested_dict[nk] = nv.item()
                                        else:
                                            nested_dict[nk] = nv
                                    hist_entry_copy[hk] = nested_dict
                                else:
                                    hist_entry_copy[hk] = hv
                            serialized_list.append(hist_entry_copy)
                        search_hist_serializable[feat] = serialized_list
                    history_copy[key] = search_hist_serializable
                elif isinstance(value, (np.bool_, bool)):
                    history_copy[key] = bool(value)
                elif isinstance(value, (np.integer, np.floating)):
                    history_copy[key] = value.item()
                else:
                    history_copy[key] = value
            histories_serializable.append(history_copy)
        
        budget_history_path = k_dir / "budget_computation_history.json"
        with open(budget_history_path, 'w') as f:
            json.dump(histories_serializable, f, indent=2)
        print(f"✓ Saved budget computation history with bivariate search details")
    
    print(f"\n✓ All datasets saved to: {k_dir}")


# ============================================================================
# CHECKPOINT: COMPLETION MARKER FOR EACH K FOLDER (--resume)
# ============================================================================
# After a k folder is saved, this stage writes MARKER_NAME as the last file of
# the folder. With --resume, a k level is skipped only if its marker matches
# the current run: same seed, same k, same label-sharing setting, same Phase 2
# settings (phase2_settings), same scripts
# and framework modules, same dataset config, same training data, same ground
# truth, and every listed file unchanged.
# The phantom generation uses one random stream for each stage (protocol,
# PART 1B), so a k level that runs again gives the same files as an
# uninterrupted run.

MARKER_NAME = "_complete.json"


def sha256_of_file(path):
    """SHA-256 checksum of one file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def folder_checksums(k_dir):
    """
    {relative path: SHA-256} for every file in the k folder.

    The marker itself and hidden files (for example .DS_Store, which macOS adds)
    are not included.
    """
    checksums = {}
    for path in sorted(Path(k_dir).rglob("*")):
        relative = path.relative_to(k_dir)
        if not path.is_file() or relative.name == MARKER_NAME:
            continue
        if any(part.startswith(".") for part in relative.parts):
            continue
        checksums[relative.as_posix()] = sha256_of_file(path)
    return checksums


def write_completion_marker(k_dir, k_value, inputs):
    """Write the completion marker as the last file of a saved k folder."""
    marker = {"k_value": k_value, **inputs, "files": folder_checksums(k_dir)}
    with open(Path(k_dir) / MARKER_NAME, "w") as f:
        json.dump(marker, f, indent=2)
    print(f"✓ Completion marker written: {Path(k_dir) / MARKER_NAME} ({len(marker['files'])} files)")


def marker_matches(k_dir, k_value, inputs):
    """
    Check the completion marker of one k folder against the current run.

    Returns:
        (True, "") if the folder is complete for this run;
        (False, reason) otherwise.
    """
    marker_path = Path(k_dir) / MARKER_NAME
    if not marker_path.exists():
        return False, "no completion marker"
    try:
        with open(marker_path) as f:
            marker = json.load(f)
    except (OSError, ValueError):
        return False, "completion marker cannot be read"
    if marker.get("k_value") != k_value:
        return False, "marker is for a different k"
    for key, value in inputs.items():
        if marker.get(key) != value:
            return False, f"{key} differs"
    if marker.get("files") != folder_checksums(k_dir):
        return False, "files in the folder differ from the marker"
    return True, ""


def phase2_settings(settings):
    """The settings of this run that change Phase 2 (recorded in each completion marker)."""
    return {"cramers_v_yates": settings.cramers_v_yates, "phase2_trials": settings.phase2_trials}


def run_inputs(seed, data_path, label_sharing, dataset_config_path, ground_truth_files, phase2=None):
    """
    The inputs that decide the content of every k folder of this run.

    ground_truth_files: {name: path} of the integer-form ground truth
    (data_io.load_integer_ground_truth).
    phase2: the Phase 2 settings (phase2_settings).
    """
    protocol_file = sys.modules[prepare_protocol.__module__].__file__
    return {
        "generation_seed": seed,
        "protocol_script_sha256": sha256_of_file(protocol_file),
        "generation_script_sha256": sha256_of_file(__file__),
        "training_data_sha256": sha256_of_file(data_path),
        "ground_truth_sha256": {name: sha256_of_file(p) for name, p in ground_truth_files.items()},
        "label_sharing": label_sharing,
        "phase2_settings": dict(phase2 or {}),
        "dataset_config_sha256": sha256_of_file(dataset_config_path),
        "framework_modules_sha256": {name: sha256_of_file(FRAMEWORK_DIR / name) for name in FRAMEWORK_MODULES},
    }


# ============================================================================
# MODEL FORM
# ============================================================================

def model_ground_truth_dir(data_dir, seed, model_split_folder):
    """Model-form ground truth of one seed (written by stage 2)."""
    return data_io.federated_seed_dir(data_dir, seed, model_split_folder) / "ground_truth"


def write_model_form_k_levels(cfg, maps, out_dir, model_out_dir, model_columns, folders):
    """
    Write the model form of each k folder in folders.

    The integer-form features of each client are encoded to the model columns
    of the model-form ground truth (encoding.encode_model_from_integer). The
    labels are copied through. No scaler is fitted here: the training code
    applies the scalers of the model-form ground truth.
    """
    for folder in folders:
        src, dst = out_dir / folder, model_out_dir / folder
        for client_index in range(cfg.n_clients):
            int_df = pd.read_csv(src / f"client_{client_index}" / "features.csv")
            model = encode_model_from_integer(int_df, client_index, model_columns[client_index], maps, cfg,
                                              f"{folder}/client_{client_index}")
            client_dir = dst / f"client_{client_index}"
            client_dir.mkdir(parents=True, exist_ok=True)
            model.to_csv(client_dir / "features.csv", index=False)
        pd.read_csv(src / "labels.csv").to_csv(dst / "labels.csv", index=False)
        print(f"  ✓ {folder}: model form written to {dst}")


# ============================================================================
# ONE DATA FOLDER FOR EACH LABEL-SHARING SETTING
# ============================================================================

def check_label_sharing_of_k_folders(seed_dir, label_sharing, kstar_folder=DEFAULT_KSTAR_FOLDER, warn=True):
    """
    Stop if a k folder of the seed folder was made with the other
    label-sharing setting.

    Every k folder is checked (k<number> and the k* folder), not only the
    requested k levels. The setting of a k folder is the key label_sharing of
    its completion marker. A k folder with no readable marker, or with a
    marker that has no label_sharing key, has an unknown setting (an
    interrupted run, or a folder from before the completion marker); with
    warn=True it gets a warning. A requested k level overwrites such a folder.

    Returns:
        the names of the k folders with an unknown setting
    """
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        return []
    k_dirs = sorted(p for p in seed_dir.iterdir()
                    if p.is_dir() and (re.fullmatch(r"k\d+", p.name) or p.name == kstar_folder))
    other, unknown = [], []
    for k_dir in k_dirs:
        try:
            with open(k_dir / MARKER_NAME) as f:
                stored = json.load(f).get("label_sharing")
        except (OSError, ValueError, AttributeError):
            stored = None
        if stored is True or stored is False:
            if stored != label_sharing:
                other.append(k_dir.name)
        else:
            unknown.append(k_dir.name)
    if other:
        raise ValueError(
            f"{seed_dir}: the k folders {other} were made with label sharing "
            f"{'off' if label_sharing else 'on'}; this run has label sharing {'on' if label_sharing else 'off'}. "
            f"Use a separate data folder for each label-sharing setting. No data file was written.")
    if warn:
        for name in unknown:
            print(f"  [!] k folder {name}: no completion marker with the label-sharing setting, so its "
                  f"setting is unknown. A requested k level overwrites this folder.")
    return unknown


# ============================================================================
# RUN
# ============================================================================

def run(settings):
    """
    Generate the phantom datasets of one seed.

    Returns:
        summary for the runner gate: output folder, natural budget, k folders
        that ran and that were skipped, reviewer check of each k folder that ran
    """
    if settings.label_sharing is not True and settings.label_sharing is not False:
        raise ValueError(f"label_sharing must be True or False, got {settings.label_sharing!r}")
    if settings.cramers_v_yates is not True and settings.cramers_v_yates is not False:
        raise ValueError(f"cramers_v_yates must be True or False, got {settings.cramers_v_yates!r}")
    check_phase2_trials(settings.phase2_trials)
    seed = int(settings.seed)
    if seed < 0:
        raise ValueError(f"seed must be 0 or more, got {seed}")
    set_context(f"seed {seed}, stage 3")
    levels = k_levels_with_folders(settings.k_levels, settings.kstar_folder)
    split_folder = data_io.split_folder_name(settings.test_share, settings.validation_share)
    model_split_folder = data_io.split_folder_name(settings.test_share, settings.validation_share, model=True)
    cfg = load_dataset_config(settings.dataset_config)
    data_dir = Path(settings.data_dir)

    print("\n" + "=" * 80)
    print(f"PHANTOM VFL STAGE 3: PHANTOM DATASETS - {cfg.name}")
    print("=" * 80)
    print(f"Seed: {seed}")
    print(f"Label sharing: {'ON' if settings.label_sharing else 'OFF'}")
    print(f"Cramér's V Yates correction: {'ON' if settings.cramers_v_yates else 'OFF'}")
    print(f"Phase 2 trials for each tested budget: {settings.phase2_trials}")
    print(f"k levels: {[folder for _, folder, _ in levels]}")
    print(f"Clients: {cfg.n_clients}")

    # One data folder for each label-sharing setting: stop before any file
    # is written if a k folder was made with the other setting
    unknown_k_folders = check_label_sharing_of_k_folders(
        data_io.federated_seed_dir(data_dir, seed, split_folder), settings.label_sharing, settings.kstar_folder)

    # Input: the integer-form ground truth of stage 2, checked against the
    # training split (row-set check; the run stops on any difference)
    df_int, maps, data_path, ground_truth_files = data_io.load_integer_ground_truth(
        data_dir, cfg, seed, split_folder)
    print(f"\nGround truth: {data_io.ground_truth_dir(data_dir, seed, split_folder)} "
          f"({len(df_int):,} rows; checked against the training data {data_path})")
    out_dir = data_io.federated_seed_dir(data_dir, seed, split_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")
    # The model form of the k levels uses the model columns of stage 2. Stop
    # here if they are missing, before the protocol runs.
    model_out_dir = data_io.federated_seed_dir(data_dir, seed, model_split_folder)
    model_columns = read_model_columns(model_ground_truth_dir(data_dir, seed, model_split_folder), cfg)

    # Copy of the integer maps of the split (shared by all k levels)
    save_integer_maps(maps, out_dir / data_io.MAPS_FILE)
    print(f"✓ Saved encoding maps: {out_dir / data_io.MAPS_FILE}")

    print("\nVertical split:")
    vertical_splits, labels_dict, client_data_dict = build_parties(df_int, cfg)

    config = PhantomVFLConfig()
    config.use_binary_search = True
    config.label_sharing = settings.label_sharing
    config.random_seed = seed  # seeds the phantom generation streams
    config.feature_types = cfg.protocol_feature_types()
    config.ordinal_ranks = {feature: cfg.ordinal_ranks(feature) for feature in cfg.features_of_type("ordinal")}
    config.cramers_v_yates = settings.cramers_v_yates
    config.validation_trials_per_k = settings.phase2_trials

    inputs = run_inputs(seed, data_path, settings.label_sharing, settings.dataset_config, ground_truth_files,
                        phase2_settings(settings))
    client_ids = list(vertical_splits.keys())
    client_data_list = [client_data_dict[cid] for cid in client_ids]

    summary = {
        "dataset": cfg.name,
        "seed": seed,
        "label_sharing": settings.label_sharing,
        "k_folders_unknown_label_sharing": unknown_k_folders,
        "output_dir": str(out_dir),
        "natural_budget": None,
        "phase2_ran": False,
        "ran": [],
        "skipped": [],
        "reviewer_check_passed": {},
        "model_form_written": [],
    }
    prepared = None
    for k_idx, (k_value, folder, is_kstar) in enumerate(levels):
        k_dir = out_dir / folder
        set_context(f"seed {seed}, {folder}")
        print("\n" + "=" * 80)
        print(f"[{k_idx + 1}/{len(levels)}] k level {'k*' if is_kstar else f'k = {k_value}'} (folder {folder})")
        print("=" * 80)
        if settings.resume:
            complete, reason = marker_matches(k_dir, k_value, inputs)
            if complete:
                print(f"✓ Skipped (--resume): {k_dir} is complete for this run")
                summary["skipped"].append(folder)
                continue
            print(f"  --resume: {folder} runs ({reason})")

        if prepared is None:
            # Phases 0 to 2 once per seed: the natural budget does not depend on the k level
            prepared = prepare_protocol(labels_dict, client_data_list, client_ids, config)
            summary["natural_budget"] = prepared["natural_budget"]
            summary["phase2_ran"] = True

        k_dir.mkdir(parents=True, exist_ok=True)
        results = run_k_level(prepared, None if is_kstar else k_value, str(k_dir))

        # Save datasets, then the completion marker as the last file
        save_phantom_datasets(k_value, k_dir, results, vertical_splits, cfg.output_name, settings.phase2_trials)
        write_completion_marker(k_dir, k_value, inputs)
        metrics = (results["substitution"] if is_kstar
                   else results["experimental_results"][0]["substitution_metrics"])
        summary["reviewer_check_passed"][folder] = bool(metrics["all_verified"])
        summary["ran"].append(folder)

    set_context(f"seed {seed}, model form")
    print("\nModel form (one-hot) of the k levels:")
    folders = [folder for _, folder, _ in levels]
    write_model_form_k_levels(cfg, maps, out_dir, model_out_dir, model_columns, folders)
    summary["model_form_written"] = folders

    print("\n" + "=" * 80)
    print("STAGE 3 COMPLETED")
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    print(f"Natural budget k*: {summary['natural_budget'] if summary['phase2_ran'] else 'not computed (all k levels skipped)'}")
    print(f"k folders written: {summary['ran'] or 'none'}")
    if settings.resume:
        print(f"k folders skipped (--resume): {summary['skipped'] or 'none'}")
    print(f"Reviewer check passed: {summary['reviewer_check_passed'] or 'no k level ran'}")
    print(f"Model-form k folders written: {summary['model_form_written']}")
    print("=" * 80 + "\n")
    return summary


def main(argv=None):
    """Command line of this stage."""
    parser = argparse.ArgumentParser(description="Stage 3: generate the phantom datasets of one seed")
    parser.add_argument("--dataset-config", required=True, help="Path of the dataset config (dataset.json)")
    parser.add_argument("--data-dir", default=".", help="Data folder that holds 03_train_test_splits/ (default: current folder)")
    parser.add_argument("--seed", type=int, required=True, help="Seed of the split and of the phantom generation")
    parser.add_argument("--label-sharing", choices=["on", "off"], required=True,
                        help="on: the clients get coded labels; off: no labels (no default)")
    parser.add_argument("--k-levels", default=",".join(str(k) for k in DEFAULT_K_LEVELS),
                        help="Comma-separated k levels: 'kstar' and whole percentages (default: %(default)s)")
    parser.add_argument("--kstar-folder", default=DEFAULT_KSTAR_FOLDER, help="Folder name of k* (default: %(default)s)")
    parser.add_argument("--test-share", type=float, default=data_io.DEFAULT_TEST_SHARE,
                        help="Test share of stage 1, a whole percentage; with the validation share it "
                             "gives the split folder names (default: %(default)s)")
    parser.add_argument("--validation-share", type=float, default=data_io.DEFAULT_VALIDATION_SHARE,
                        help="Validation share of stage 1, a whole percentage; 0: no validation set "
                             "(default: %(default)s)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip each k level whose _complete.json marker matches this run")
    parser.add_argument("--cramers-v-yates", action="store_true",
                        help="Phase 2: Cramer's V with the Yates correction of a 2 x 2 table, as in the "
                             "scripts of the published results (default: no correction)")
    parser.add_argument("--phase2-trials", type=int, default=DEFAULT_PHASE2_TRIALS,
                        help="Phase 2: random removals for each tested budget in both searches, an odd "
                             "number; majority vote (default: %(default)s)")
    args = parser.parse_args(argv)
    settings = GenerationSettings(
        dataset_config=args.dataset_config,
        data_dir=args.data_dir,
        seed=args.seed,
        label_sharing=(args.label_sharing == "on"),
        k_levels=[level for level in args.k_levels.split(",") if level.strip()],
        kstar_folder=args.kstar_folder,
        test_share=args.test_share,
        validation_share=args.validation_share,
        resume=args.resume,
        cramers_v_yates=args.cramers_v_yates,
        phase2_trials=args.phase2_trials,
    )
    with run_log(settings.data_dir, "stage3"):
        return run(settings)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, DatasetConfigError) as error:
        sys.exit(f"ERROR: {error}")
