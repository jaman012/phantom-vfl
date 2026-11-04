# Phantom VFL: phantom generation protocol (one module for both label-sharing settings)
#
# Input: the integer form of the training split only. The phantom generation
# stage (generate_phantom_datasets.py) reads it and calls this module. The test
# set stays apart for the evaluation, so no phantom individual enters it.
#
# History of changes: CHANGELOG_generation_code.md

"""
PHANTOM VFL PROTOCOL

ARCHITECTURE:
- Server: central orchestrator with LABELS ONLY (no features, no removal)
- Clients: passive parties with FEATURES ONLY; each client secretly removes individuals
- Any number of clients

LABEL SHARING (config.label_sharing; no default, True or False):
- True: the server sends coded labels to the clients (Phase 0); removal is
  stratified by coded label; each substitute has the same coded label when
  possible (label-aware substitution).
- False: the clients get no labels in any mode; removal is a simple random
  sample; each substitute is a uniform random draw from the pot.

PHASES:
Phase 0: Setup - encoded label sharing (label sharing on only)
Phase 1: Private Set Intersection (PSI)
Phase 2: Phantom budget computation (per client, then global negotiation)
  - Binary (or incremental) search for the marginal capacity of each feature
  - Binary search for the capacity that preserves all feature pairs (bivariate)
  - The natural budget k* does not depend on the k level: Phases 0 to 2 run
    once (prepare_protocol), and each k level uses the result (run_k_level)
Phase 3: Independent removal (secret, clients only)
Phase 4: Training order proposal (server selects at random from all matched IDs)
Phase 4b: Silent substitution (one-to-one mapping, reviewer check)
Phase 4c: Ground truth tracking (experimental validation; files in the k folder)
Phase 5: VFL training (Flower framework, separate scripts)
Phase 6: Privacy evaluation (after training, separate)

RANDOM STREAMS:
- Each stage (Phase 2 capacity, removal, server order, substitution) draws
  from its own numpy.random.RandomState, made from the run seed (PART 1B).
  The global NumPy random state is not used. A run can be repeated.

KEY DESIGN PRINCIPLES:
- Server never knows which positions are phantoms (privacy preserved)
- Clients silently substitute without revealing removals
- One-to-one mapping (no ID reuse), confirmed by verify_phantom_mapping
- Label sharing on: fallback to label mismatch when no same-label ID is left in the pot
"""

import numpy as np
import pandas as pd
import json
import time
import os
from scipy import stats
from collections import Counter, defaultdict, deque
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, asdict

# Convert NumPy types to Python native types
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)

# ============================================================================
# PART 1: DATA STRUCTURES AND CONFIGURATION
# ============================================================================

class Server:
    """
    Central server in Flower VFL architecture.
    - Has LABELS ONLY (no features)
    - Does NOT remove individuals (has labels for all)
    - Orchestrates phantom protocol
    - Creates training order by random selection
    """
    def __init__(self, labels: Dict[str, float]): # takes in labels as dictionary input e.g. `{"p00001": 0.0, "p00002": 1.0, "p00003": 0.0, ...}`
        self.labels = labels  # Original labels {id: label_value} i.e. original labels with alphabetic meaning
        self.encoded_labels = None  # Symbolically encoded labels {id: encoded_label} e.g. {'p00001': 'dog', 'p00002': 'ice_cream', ...}
        self.label_encoding = None  # Encoding map {original_value: encoded_value} e.g. {0.0: 'dog', 1.0: 'ice_cream'}
        self.matched_ids = None  # IDs after PSI
        self.training_order = []  # Training order (randomly selected from matched_ids)
        
    def set_matched_ids(self, ids: List[str]):
        """Set IDs after PSI protocol"""
        self.matched_ids = ids # store matched ids after PSI
        self.n_matched = len(ids)
    
    def create_label_encoding(self, encoding_type: str = 'symbolic') -> Dict:
        """
        Create encoding for labels (dogs & ice-cream analogy).
        Protects alphabetic meaning while enabling label-aware substitution.
        
        Args:
            encoding_type: 'symbolic' (dog/ice_cream) or 'alphabetic' (A/B)
        
        Returns:
            Encoding map {original_label: encoded_label}
        """
        unique_labels = sorted(set(self.labels.values()))
        
        if encoding_type == 'symbolic':
            # Dogs & ice-cream analogy
            symbols = ['dog', 'ice_cream', 'cat', 'pizza', 'book']
            self.label_encoding = {
                label: symbols[i % len(symbols)]
                for i, label in enumerate(unique_labels)
            }
        elif encoding_type == 'alphabetic':
            # Alphabetic encoding
            self.label_encoding = {
                label: f"class_{chr(65 + i)}"  # A, B, C, ...
                for i, label in enumerate(unique_labels)
            }
        else:
            # Numeric encoding
            self.label_encoding = {
                label: i
                for i, label in enumerate(unique_labels)
            }
        
        # Create encoded label dictionary
        self.encoded_labels = {
            id: self.label_encoding[label]
            for id, label in self.labels.items()
        }
        
        print(f"Label encoding created ({encoding_type}):")
        for orig, enc in sorted(self.label_encoding.items()):
            print(f"  {orig} → {enc}")
        print()
        
        return self.label_encoding
    
    def get_label_data(self, ids: Optional[List[str]] = None) -> Dict[str, float]:
        """Get original labels for specified IDs"""
        if ids is None:
            ids = self.matched_ids
        return {id: self.labels[id] for id in ids}
    
    def get_encoded_label_data(self, ids: Optional[List[str]] = None) -> Dict[str, any]:
        """Get encoded labels for specified IDs"""
        if ids is None:
            ids = self.matched_ids
        return {id: self.encoded_labels[id] for id in ids}


class PassiveClient:
    """
    Passive client party in Flower VFL.
    - Has FEATURES ONLY (no original labels)
    - Receives ENCODED LABELS from server (Phase 0, label sharing on only)
    - Participates in phantom budget computation
    - Independently removes individuals (secret)
    - Creates a SILENT one-to-one phantom mapping (label-aware if label sharing is on)
    """
    def __init__(self, client_id: str, features: List[str], data: Dict[str, Dict[str, float]]):
        self.client_id = client_id
        self.features = features # List of feature names
        self.data = data # {id: {feature_name: value}} for all individuals in client
        self.feature_types = {}  # {feature_name: 'continuous', 'categorical' or 'ordinal'}
        
        # PSI results
        self.matched_ids = None
        self.n_matched = 0
        
        # Phase 0: Encoded labels from server
        self.encoded_labels = None  # {id: encoded_label} from server
        
        # Phase 2: Budget computation
        self.removal_capacity = 0 # Maximum individuals that can be removed while preserving distribution
        
        # Phase 3: Independent removal (SECRET)
        self.removed_ids = None  # Set of removed IDs (SECRET)
        self.training_ids = None  # List of kept IDs
        
        # Phase 4b: Silent phantom mapping
        self.phantom_mapping = {}  # {position: real_id} - COMPLETE one-to-one mapping
        self.phantom_positions = []  # Positions where client uses substitutes
        
    def set_matched_ids(self, ids: List[str]):
        """Set IDs after PSI protocol"""
        self.matched_ids = ids
        self.n_matched = len(ids)
    
    def receive_encoded_labels(self, encoded_labels: Dict[str, any]):
        """
        Receive encoded labels from server (Phase 0).
        Enables label-aware substitution without knowing alphabetic meaning.
        """
        self.encoded_labels = encoded_labels
        print(f"{self.client_id} received {len(encoded_labels)} encoded labels")
    
    def set_removal_set(self, removed_ids: List[str]):
        """Set removed IDs after Phase 3 (kept secret from other parties)"""
        self.removed_ids = set(removed_ids)
        self.training_ids = [id for id in self.matched_ids 
                            if id not in self.removed_ids]
    
    def get_feature_data(self, ids: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Get feature data for specified IDs"""
        if ids is None:
            ids = self.training_ids
        return {id: self.data[id] for id in ids}


@dataclass
class PhantomRateStatistics:
    """
    Statistics for experimental phantom rate analysis and validation.
    Used by Ground Truth Tracker in Phase 4c.
    
    Also holds the execution timing of each phase.
    """
    total_ids_after_psi: int
    total_ids_after_removal: int
    
    # Per-client removal stats
    client_removal_counts: Dict[str, int]
    
    # Expected phantom rates (from Phase 2 budget)
    expected_phantom_rate_per_client: Dict[str, float]
    overall_expected_phantom_rate: float
    
    # Actual phantom rates (ground truth)
    actual_phantom_count: int
    actual_phantom_rate: float
    
    # Breakdown by client removal pattern
    removed_by_single_client: int
    removed_by_multiple_clients: int
    removed_by_all_clients: int
    
    # Execution timing
    time_total: float = 0.0
    time_phase0: float = 0.0
    time_phase1: float = 0.0
    time_phase2_total: float = 0.0
    time_phase3_total: float = 0.0
    time_phase4: float = 0.0
    time_phase4b_total: float = 0.0
    
    def to_dict(self):
        return asdict(self)


@dataclass
class TimingMetrics:
    """
    Execution timing metrics for performance analysis.
    
    Tracks phase-level timing and per-client timing for multi-client phases.
    Designed for Monte Carlo simulation analysis with mean, std, min, max aggregation.
    
    Precision: 3 decimal places (millisecond accuracy)
    
    Usage:
        timing = TimingMetrics()
        timing.phase0 = 0.053
        timing.phase2_per_client = {'client_1': 2.345, 'client_2': 2.187}
        timing.compute_totals()  # Computes phase2_total, phase3_total, etc.
    """
    # Phase-level timing (seconds, 3 decimal precision)
    phase0: float = 0.0  # Label encoding
    phase1: float = 0.0  # PSI
    phase2_total: float = 0.0  # Budget computation (all clients)
    phase3_total: float = 0.0  # Independent removal (all clients)
    phase4: float = 0.0  # Training order proposal
    phase4b_total: float = 0.0  # Phase 4b substitution (all clients)
    
    # Per-client timing for multi-client phases
    phase2_per_client: Dict[str, float] = None  # {client_id: duration}
    phase3_per_client: Dict[str, float] = None  # {client_id: duration}
    phase4b_per_client: Dict[str, float] = None  # {client_id: duration}
    
    # Total protocol time
    total: float = 0.0
    
    def __post_init__(self):
        """Initialize dictionaries if None"""
        if self.phase2_per_client is None:
            self.phase2_per_client = {}
        if self.phase3_per_client is None:
            self.phase3_per_client = {}
        if self.phase4b_per_client is None:
            self.phase4b_per_client = {}
    
    def compute_totals(self):
        """Compute total time and multi-client phase totals"""
        # Sum per-client times for multi-client phases
        self.phase2_total = sum(self.phase2_per_client.values()) if self.phase2_per_client else 0.0
        self.phase3_total = sum(self.phase3_per_client.values()) if self.phase3_per_client else 0.0
        self.phase4b_total = sum(self.phase4b_per_client.values()) if self.phase4b_per_client else 0.0
        
        # Total protocol time
        self.total = (
            self.phase0 + 
            self.phase1 + 
            self.phase2_total + 
            self.phase3_total + 
            self.phase4 + 
            self.phase4b_total
        )
    
    def get_summary_dict(self) -> Dict:
        """Get summary dictionary for results output (without per-client details)"""
        return {
            'total': round(self.total, 3),
            'phase0': round(self.phase0, 3),
            'phase1': round(self.phase1, 3),
            'phase2_total': round(self.phase2_total, 3),
            'phase3_total': round(self.phase3_total, 3),
            'phase4': round(self.phase4, 3),
            'phase4b_total': round(self.phase4b_total, 3)
        }
    
    def get_detailed_dict(self) -> Dict:
        """Get detailed dictionary including per-client timing"""
        summary = self.get_summary_dict()
        summary['phase2_per_client'] = {k: round(v, 3) for k, v in self.phase2_per_client.items()}
        summary['phase3_per_client'] = {k: round(v, 3) for k, v in self.phase3_per_client.items()}
        summary['phase4b_per_client'] = {k: round(v, 3) for k, v in self.phase4b_per_client.items()}
        return summary
    
    def to_dict(self):
        """Full serialization for dataclass"""
        return asdict(self)


class PhantomVFLConfig:
    """
    Configuration for Phantom VFL protocol.

    Main options:
    - label_sharing: True or False; no default (the protocol stops if it is not set)
    - use_binary_search: binary (O(log n)) or incremental (O(n)) capacity search
    - validation_trials_per_k: trials for each tested budget in the marginal and the bivariate
      search (an odd number; majority vote if more than 1); the run setting phase2_trials
    - feature_types: {feature: 'continuous', 'categorical' or 'ordinal'} for every feature
    - ordinal_ranks: {feature: {value: rank}} for every ordinal feature
    - cramers_v_yates: Yates correction of a 2 x 2 table in Cramér's V (default False)
    - enable_bivariate_validation: bivariate relationship checks in Phase 2
    - enable_ground_truth_tracking: ground truth CSV and summary JSON (Phase 4c)
    - random_seed: run seed of the phantom generation (PART 1B)
    """
    def __init__(self):
        """Initialize Phantom VFL configuration with the default settings."""

        # ====================================================================
        # LABEL SHARING
        # ====================================================================
        # True: Phase 0 runs (coded labels go to the clients), removal is
        # stratified by coded label and the substitution matches labels.
        # False: the clients get no labels in any mode.
        # No default: the protocol stops if this is not True or False.
        self.label_sharing = None
        
        # ====================================================================
        # PHASE 0: Label Encoding
        # ====================================================================
        self.label_encoding_type = 'alphabetic'  # 'symbolic', 'alphabetic', or 'numeric'
        
        # ====================================================================
        # PHASE 2: Distribution Validation Thresholds
        # ====================================================================
        # Continuous features (KS test + moment matching)
        self.epsilon_ks = 0.05  # KS statistic threshold
        self.epsilon_mom = 0.10  # Relative moment difference threshold
        
        # Categorical features (chi-square + TV distance)
        self.alpha = 0.05  # Significance level for chi-square
        self.epsilon_tv = 0.05  # Total variation distance threshold
        self.min_category_count = 5  # Minimum expected count per category
        
        # Bivariate validation thresholds
        self.enable_bivariate_validation = True  # Toggle bivariate checks
        self.delta_r_threshold = 0.05  # Pearson correlation tolerance
        self.delta_f_threshold = 0.20  # ANOVA F-statistic relative change tolerance
        self.delta_v_threshold = 0.05  # Cramér's V tolerance
        self.delta_rho_threshold = 0.05  # Spearman rank correlation tolerance (pairs of an ordinal feature with an ordinal or continuous feature)
        # Yates correction of chi2_contingency for a 2 x 2 table in Cramér's V.
        # False: no correction (the formula of the protocol); True: the scipy
        # default, as in the scripts of the published results
        self.cramers_v_yates = False
        
        # Maximum removal constraints
        self.max_removal_fraction = 0.25  # Max 25% of individuals can be removed
        
        # ====================================================================
        # PHASE 2: Search Strategy
        # ====================================================================
        # Controls search algorithm for finding removal capacity
        self.use_binary_search = True  # True: O(log n) iterations, False: O(n) iterations
        self.binary_search_tolerance = 1  # Acceptable error in capacity (individuals)
        # Trials for each tested budget, in the marginal and the bivariate search
        # (majority vote if more than 1; an odd number gives no tie). The
        # phantom generation stage sets it from the setting phase2_trials.
        self.validation_trials_per_k = 1
        
        # ====================================================================
        # PHASE 4b: Substitution Strategy
        # ====================================================================
        self.allow_label_mismatch_fallback = True  # Use mismatched labels when pot exhausted

        # ====================================================================
        # RANDOM STREAMS
        # ====================================================================
        # Run seed of the phantom generation. Each stage (Phase 2 capacity,
        # removal, server order, substitution) draws from a random stream derived
        # from this seed (see stage_rng). None: the protocol makes a seed,
        # prints it and records it in the results.
        self.random_seed = None
        
        # ====================================================================
        # PHASE 4c: GROUND TRUTH TRACKING
        # ====================================================================
        # For experimental validation only - creates CSV showing real vs phantom IDs
        # WARNING: Violates honest-but-curious model when enabled
        # The two files go to the output folder of each k level (run_k_level).
        self.enable_ground_truth_tracking = True  # Toggle on/off

        # ====================================================================
        # Feature types
        # ====================================================================
        # {feature_name: 'continuous', 'categorical' or 'ordinal'} for every
        # feature of every client. The run stops if a feature has no type or
        # another type. An ordinal feature takes the categorical marginal
        # checks and the categorical bivariate check of its pair; in a pair with
        # an ordinal or continuous feature it also takes the Spearman check.
        # The phantom generation stage sets this table from the dataset config.
        self.feature_types = {}
        # {feature_name: {value: rank}} for each ordinal feature: the rank of
        # each value in the configured order (Spearman check). The phantom
        # generation stage sets this table from the dataset config.
        self.ordinal_ranks = {}

# ============================================================================
# PART 1B: RANDOM STREAMS (REPEATABLE RUNS)
# ============================================================================
# Each stage of the phantom generation draws from its own numpy.random.RandomState.
# Its seed comes from the run seed, the stage, the k level and the client index. So two
# clients never share a stream, two k levels never share a stream, and the
# number of draws in one stage does not change the draws of a later stage.
# The Phase 2 stream has no k level: every k run of one seed finds the same
# natural budget.

STAGE_PHASE2_CAPACITY = 1   # Phase 2: trial removals of the capacity search
STAGE_REMOVAL = 2           # Phase 3: independent removal
STAGE_SERVER_ORDER = 3      # Phase 4: server training order
STAGE_SUBSTITUTION = 4      # Phase 4b: shuffle of the substitution pot
SERVER_STREAM_INDEX = 0     # Index of the server stream (one per k level)


def resolve_run_seed(config: PhantomVFLConfig) -> int:
    """
    Return the run seed of the phantom generation.

    If config.random_seed is None, make a seed from operating-system entropy,
    store it in config.random_seed and print it. The run can then be repeated
    with that seed.
    """
    if config.random_seed is None:
        config.random_seed = int(np.random.SeedSequence().entropy)
        print(f"No random seed given. Generated seed: {config.random_seed}")
    if int(config.random_seed) < 0:
        raise ValueError(f"random_seed must be 0 or more, got {config.random_seed}")
    return int(config.random_seed)


def k_stream_code(k_proportion: Optional[float]) -> int:
    """
    Code of the k level in a stream key.

    None (k* in natural mode, or a stage that does not depend on k) gives 0.
    An imposed k gives round(k * 10000) + 1, so an imposed k = 0 differs from k*.
    """
    if k_proportion is None:
        return 0
    return int(round(k_proportion * 10000)) + 1


def stage_seed(run_seed: int, stage: int, client_index: int,
               k_proportion: Optional[float] = None) -> int:
    """
    Derive the 32-bit NumPy seed of one stage with numpy.random.SeedSequence.

    Key: (run seed, stage, k level code, client index).
    """
    key = [int(run_seed), int(stage), k_stream_code(k_proportion), int(client_index)]
    return int(np.random.SeedSequence(key).generate_state(1)[0])


def stage_rng(config: PhantomVFLConfig, stage: int, client_index: int,
              k_proportion: Optional[float] = None) -> np.random.RandomState:
    """
    Return the random stream of one stage.

    The stream is a numpy.random.RandomState made from stage_seed. Its choice
    and permutation give the same draws as the global NumPy functions after
    seeding the global state with the same value. The global NumPy random
    state does not change.

    Args:
        config: PhantomVFLConfig with random_seed
        stage: One of the STAGE_* constants
        client_index: Position of the client in the client list
                      (SERVER_STREAM_INDEX for the server)
        k_proportion: Imposed k, or None for k* and for Phase 2

    Returns:
        numpy.random.RandomState of this stage
    """
    return np.random.RandomState(stage_seed(resolve_run_seed(config), stage, client_index, k_proportion))


def check_label_sharing(config: PhantomVFLConfig) -> bool:
    """Return config.label_sharing. Stop if it is not True or False (it has no default)."""
    if config.label_sharing is not True and config.label_sharing is not False:
        raise ValueError(
            f"config.label_sharing must be True or False, got {config.label_sharing!r}"
        )
    return config.label_sharing


# ============================================================================
# PART 2: PHASE 0 - SETUP (ENCODED LABEL SHARING)
# ============================================================================

def setup_encoded_label_sharing(
    server: Server,
    clients: List[PassiveClient],
    config: PhantomVFLConfig # Configuration including encoding type from PhantomVFLConfig class
) -> Dict[str, any]:
    """
    Phase 0: Server shares encoded labels with clients.
    
    Enables label-aware phantom substitution while protecting
    alphabetic meaning (dogs & ice-cream analogy).
    
    Args:
        server: Server with original labels
        clients: List of passive clients
        config: Configuration including encoding type
    
    Returns:
        setup_info: Dictionary with encoding details
    """
    print("=" * 70)
    print("PHASE 0: SETUP - ENCODED LABEL SHARING")
    print("=" * 70 + "\n")
    
    # Create encoding
    encoding = server.create_label_encoding(config.label_encoding_type) # creates encoding map (output) and encoded labels in server using the selected encoding type e.g. 'alphabetic'
    
    # Share encoded labels with all clients
    for client in clients:
        client.receive_encoded_labels(server.encoded_labels.copy())
    
    print("Clients can now perform label-aware substitution")
    print("Server's label meaning protected (labels are sent as codes, e.g. class_A / class_B)\n")
    
    setup_info = {
        'encoding_type': config.label_encoding_type,
        'encoding_map': encoding,
        'n_labels_shared': len(server.encoded_labels)
    }
    
    return setup_info


# ============================================================================
# PART 3: PHASE 1 - PRIVATE SET INTERSECTION
# ============================================================================

def private_set_intersection(
    server: Server,
    clients: List[PassiveClient]
) -> Tuple[List[str], Dict[str, int]]:
    """
    Phase 1: Compute intersection of IDs across server and all clients.
    
    In practice, this uses cryptographic PSI protocols (e.g., ECDH-PSI).
    Here we show the logical outcome.
    
    Args:
        server: Server with labels
        clients: List of passive client parties
    
    Returns:
        common_ids: List of IDs present in server AND all clients
        stats: Statistics about intersection
    """
    print("="*70)
    print("PHASE 1: PRIVATE SET INTERSECTION")
    print("="*70 + "\n")
    
    # Get ID sets
    server_ids = set(server.labels.keys()) # extracting IDs from server labels dictionary keys and convert to set for efficient intersection
    client_id_sets = [set(client.data.keys()) for client in clients] # list comprehension to get ID sets from all clients
    
    # Compute intersection
    common_ids = server_ids.intersection(*client_id_sets)  # intersecting IDs in all parties
    # Sorted list: the random draws pick positions in this list, and a set
    # has no fixed order between runs
    common_ids = sorted(common_ids)
    
    # Statistics
    print(f"Server IDs: {len(server_ids)}")
    for i, client in enumerate(clients):
        print(f"Client {i+1} IDs: {len(client_id_sets[i])}")
    print(f"Common IDs (intersection): {len(common_ids)}\n")
    
    # Update all parties
    server.set_matched_ids(common_ids)  # set_matched_ids defined in both Server and PassiveClient classes above
    for client in clients:              # each client stores the same matched IDs
        client.set_matched_ids(common_ids)
    
    stats = {
        'server_ids': len(server_ids),
        'client_ids': [len(s) for s in client_id_sets],
        'common_ids': len(common_ids) # note that common ids is dependent on the intersection of all parties i.e. intersection(*client_id_set)
    }
    
    return common_ids, stats  # used in Phase 1: Private Set Intersection further down below


# ============================================================================
# PART 4: DISTRIBUTION VALIDATION FUNCTIONS
# ============================================================================

def validate_marginal_distribution_continuous(
    original_values: np.ndarray,
    reduced_values: np.ndarray,
    config: PhantomVFLConfig # Config passed to access tolerances from PhantomVFLConfig class
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that reduced dataset preserves marginal distribution for continuous feature.
    Uses KS test + moment matching.
    """
    # Kolmogorov-Smirnov test for distributional similarity (# null hypothesis: distributions are the same)
    ks_stat, ks_pvalue = stats.ks_2samp(original_values, reduced_values) # ks_2samp from scipy (imported as stats above)
    ks_pass = ks_stat < config.epsilon_ks # Check if KS statistic is within tolerance
    
    # Moment preservation (mean, variance, skewness, kurtosis)
    moments_original = {
        'mean': np.mean(original_values), # first moment (central tendency)
        'var': np.var(original_values), # second moment (spread)
        'skew': stats.skew(original_values), # third moment (asymmetry)
        'kurt': stats.kurtosis(original_values) # fourth moment (tail heaviness)
    }
    
    moments_reduced = {
        'mean': np.mean(reduced_values),
        'var': np.var(reduced_values), # squared deviation from mean
        'skew': stats.skew(reduced_values), # 0: symmetric, >0: right-skewed, <0: left-skewed
        'kurt': stats.kurtosis(reduced_values) # 0: normal kurtosis, >0: heavy tails, <0: light tails
    }
    
    # Relative differences
    moment_diffs = {
        k: abs(moments_reduced[k] - moments_original[k]) / (abs(moments_original[k]) + 1e-10)
        for k in moments_original.keys()
    } # adding small constant (1e-10) to avoid division by zero
    
    moments_pass = all(diff < config.epsilon_mom for diff in moment_diffs.values())
    
    valid = ks_pass and moments_pass
    
    metrics = {
        'ks_stat': ks_stat,
        'ks_pvalue': ks_pvalue,
        'ks_pass': ks_pass,
        'moments_pass': moments_pass,
        'moment_diffs': moment_diffs,
        'valid': valid
    }
    
    return valid, metrics


def validate_marginal_distribution_categorical(
    original_values: np.ndarray,
    reduced_values: np.ndarray,
    config: PhantomVFLConfig # Config passed to access tolerances from PhantomVFLConfig class
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that reduced dataset preserves marginal distribution for categorical feature.
    Uses chi-square test + total variation distance.
    """
    # Get category counts
    orig_counts = Counter(original_values) # Counter from collections library (imported above) to count occurrences of each category
    red_counts = Counter(reduced_values)
    
    # Ensure same categories
    all_categories = set(orig_counts.keys()) | set(red_counts.keys()) # union of categories in original and reduced
    
    # Chi-square test (null hypothesis: the distributions are the same; fail to reject H0 when p-value > config.alpha)
    observed = np.array([red_counts.get(cat, 0) for cat in all_categories])
    expected_freq = np.array([orig_counts.get(cat, 0) / len(original_values) 
                             for cat in all_categories])
    expected = expected_freq * len(reduced_values)
    
    # Filter out categories with very low expected counts (chi-square needs enough observations per category)
    valid_mask = expected >= config.min_category_count # only consider categories with sufficient expected counts (config.min_category_count)
    if np.sum(valid_mask) < 2: # chi square needs at least 2 categories with sufficient counts to be valid
        chi2_pass = False
        chi2_stat = np.inf
        chi2_pvalue = 0.0
    else:
        # Normalize observed to the expected sum (scipy.stats.chisquare needs equal sums; prevents floating-point precision errors)
        observed_filtered = observed[valid_mask].astype(float)
        expected_filtered = expected[valid_mask].astype(float)
        
        # Ensure sums match exactly by normalizing observed to expected sum
        observed_sum = np.sum(observed_filtered)
        expected_sum = np.sum(expected_filtered)
        if observed_sum > 0 and abs(observed_sum - expected_sum) / expected_sum > 1e-10:
            # Normalize observed to match expected sum (fixes scipy chisquare precision requirement)
            observed_filtered = observed_filtered * (expected_sum / observed_sum)
        
        chi2_stat, chi2_pvalue = stats.chisquare(
            observed_filtered, 
            expected_filtered
        )
        chi2_pass = chi2_pvalue > config.alpha 
    
    # Total variation distance: half the sum of the absolute differences of the category proportions
    orig_dist = {cat: count / len(original_values) for cat, count in orig_counts.items()}
    red_dist = {cat: count / len(reduced_values) for cat, count in red_counts.items()}
    
    tv_dist = 0.5 * sum(abs(orig_dist.get(cat, 0) - red_dist.get(cat, 0)) 
                       for cat in all_categories)
    tv_pass = tv_dist < config.epsilon_tv # Check if TV distance is within tolerance (config.epsilon_tv)
    
    valid = chi2_pass and tv_pass
    
    metrics = {
        'chi2_stat': chi2_stat,
        'chi2_pvalue': chi2_pvalue,
        'chi2_pass': chi2_pass,
        'tv_distance': tv_dist,
        'tv_pass': tv_pass,
        'valid': valid
    }
    
    return valid, metrics


def validate_bivariate_relationship(
    original_data: Dict[str, np.ndarray],
    reduced_data: Dict[str, np.ndarray],
    feature1: str,
    feature2: str,
    type1: str,
    type2: str,
    config: PhantomVFLConfig
) -> Tuple[bool, Dict[str, float]]:
    """
    Validate that the relationship between two features is preserved after removal.
    
    Cases (an ordinal feature takes the test of a categorical feature):
    - continuous-continuous: Pearson correlation
    - continuous-categorical: ANOVA F-statistic
    - categorical-categorical: Cramér's V (Yates correction of a 2 x 2 table
      only if config.cramers_v_yates)
    A pair of an ordinal feature with an ordinal or continuous feature also
    takes the Spearman check (config.delta_rho_threshold); both checks must
    pass. The Spearman check ranks an ordinal feature by its configured order
    (config.ordinal_ranks) and uses average ranks for ties.
    """
    orig_f1 = original_data[feature1]
    orig_f2 = original_data[feature2]
    redu_f1 = reduced_data[feature1]
    redu_f2 = reduced_data[feature2]
    base1 = 'categorical' if type1 == 'ordinal' else type1
    base2 = 'categorical' if type2 == 'ordinal' else type2
    
    if base1 == 'continuous' and base2 == 'continuous': 
        # Pearson correlation
        orig_corr = np.corrcoef(orig_f1, orig_f2)[0, 1] # [0, 1] gives correlation coefficient between f1 and f2 by extracting the element at row 0, column 1 of the correlation matrix
        redu_corr = np.corrcoef(redu_f1, redu_f2)[0, 1]
        delta = abs(orig_corr - redu_corr)
        valid = delta <= config.delta_r_threshold # check if difference in correlation is within tolerance (config.delta_r_threshold)
        
        metrics = {
            'metric': 'pearson_correlation', # Pearson correlation ranges from -1 to 1
            'original': orig_corr,
            'reduced': redu_corr,
            'delta': delta,
            'threshold': config.delta_r_threshold,
            'valid': valid
        }
    
    elif (base1 == 'continuous' and base2 == 'categorical') or \
         (base1 == 'categorical' and base2 == 'continuous'): 
        # ANOVA F-statistic
        if base1 == 'categorical':
            orig_cat, orig_cont = orig_f1, orig_f2
            redu_cat, redu_cont = redu_f1, redu_f2
        else:
            orig_cat, orig_cont = orig_f2, orig_f1
            redu_cat, redu_cont = redu_f2, redu_f1
        
        # Group by category
        orig_groups = defaultdict(list)
        for cat, val in zip(orig_cat, orig_cont):
            orig_groups[cat].append(val)
        
        redu_groups = defaultdict(list)
        for cat, val in zip(redu_cat, redu_cont):
            redu_groups[cat].append(val)
        
        # Compute F-statistics
        orig_f_stat, _ = stats.f_oneway(*orig_groups.values()) # f_oneway from scipy for One-way ANOVA (imported as stats above)
        redu_f_stat, _ = stats.f_oneway(*redu_groups.values())
        
        rel_change = abs(orig_f_stat - redu_f_stat) / (orig_f_stat + 1e-10)
        valid = rel_change <= config.delta_f_threshold # check if relative change in F-statistic is within tolerance (config.delta_f_threshold)
        
        metrics = {
            'metric': 'anova_f_statistic',
            'original': orig_f_stat,
            'reduced': redu_f_stat,
            'rel_change': rel_change,
            'threshold': config.delta_f_threshold,
            'valid': valid
        }
    
    else:  # categorical-categorical
        # Cramér's V - validation ensures phantom mechanism doesn't introduce spurious (artificial) associations or destroy existing ones between categorical features
        orig_crosstab = pd.crosstab(orig_f1, orig_f2)
        redu_crosstab = pd.crosstab(redu_f1, redu_f2)
        
        # correction acts on a 2 x 2 table only (config.cramers_v_yates)
        orig_chi2, _, _, _ = stats.chi2_contingency(orig_crosstab, correction=config.cramers_v_yates)
        redu_chi2, _, _, _ = stats.chi2_contingency(redu_crosstab, correction=config.cramers_v_yates)
        
        orig_n = len(orig_f1)
        redu_n = len(redu_f1)
        
        orig_v = np.sqrt(orig_chi2 / (orig_n * (min(orig_crosstab.shape) - 1)))
        redu_v = np.sqrt(redu_chi2 / (redu_n * (min(redu_crosstab.shape) - 1)))
        
        delta = abs(orig_v - redu_v)
        valid = delta <= config.delta_v_threshold # check if difference in Cramér's V is within tolerance (config.delta_v_threshold)
        
        metrics = {
            'metric': 'cramers_v',
            'original': orig_v,
            'reduced': redu_v,
            'delta': delta,
            'threshold': config.delta_v_threshold,
            'valid': valid
        }

    # Spearman check: a pair of an ordinal feature with an ordinal or
    # continuous feature. A constant column gives no rank correlation (nan),
    # and the check fails, as the Pearson check does.
    if 'ordinal' in (type1, type2) and type1 in ('ordinal', 'continuous') and type2 in ('ordinal', 'continuous'):
        def ranked(values, feature, feature_type):
            if feature_type == 'ordinal':
                ranks = config.ordinal_ranks[feature]
                return np.array([ranks[value] for value in values], dtype=float)
            return np.asarray(values, dtype=float)

        orig_rho = stats.spearmanr(ranked(orig_f1, feature1, type1), ranked(orig_f2, feature2, type2)).statistic
        redu_rho = stats.spearmanr(ranked(redu_f1, feature1, type1), ranked(redu_f2, feature2, type2)).statistic
        rho_delta = abs(orig_rho - redu_rho)
        spearman_valid = bool(rho_delta <= config.delta_rho_threshold)
        valid = bool(valid) and spearman_valid
        metrics['spearman_original'] = float(orig_rho)
        metrics['spearman_reduced'] = float(redu_rho)
        metrics['spearman_delta'] = float(rho_delta)
        metrics['spearman_threshold'] = config.delta_rho_threshold
        metrics['spearman_valid'] = spearman_valid
        metrics['valid'] = valid
    
    return valid, metrics


# ============================================================================
# PART 4B: ROBUST VALIDATION HELPERS
# ============================================================================

def robust_validate_marginal(
    client: PassiveClient,
    feature: str,
    k: int,
    config: PhantomVFLConfig,
    rng: np.random.RandomState
) -> Tuple[bool, Dict]:
    """
    Validate marginal distribution with multiple trials to reduce sampling noise.
    
    Args:
        client: Client with data
        feature: Feature name to validate
        k: Number of individuals to remove
        config: Configuration with validation_trials_per_k
        rng: Random stream of the client's Phase 2 search

    Returns:
        is_valid: True if majority of trials pass
        metrics: Validation metrics from first trial
    """
    feature_type = client.feature_types.get(feature, 'continuous')
    orig_values = np.array([client.data[id][feature] for id in client.matched_ids])
    
    # Run multiple trials
    trial_results = []
    metrics_list = []
    
    for trial in range(config.validation_trials_per_k):
        # Different random sample each trial
        # A set gives a fast membership test; the draw is the same
        test_removal = set(rng.choice(client.matched_ids, size=k, replace=False).tolist())
        remaining_ids = [id for id in client.matched_ids if id not in test_removal]
        red_values = np.array([client.data[id][feature] for id in remaining_ids])
        
        # Validate
        if feature_type == 'continuous':
            valid, metrics = validate_marginal_distribution_continuous(orig_values, red_values, config)
        else:
            valid, metrics = validate_marginal_distribution_categorical(orig_values, red_values, config)
        
        trial_results.append(valid)
        metrics_list.append(metrics)
    
    # Majority vote (or all must pass for conservative)
    if config.validation_trials_per_k == 1:
        is_valid = trial_results[0]
    else:
        # Conservative: require majority to pass
        is_valid = sum(trial_results) > len(trial_results) // 2
    
    return is_valid, metrics_list[0]  # Return first trial's metrics for reporting


def robust_validate_bivariate(
    original_data: Dict[str, np.ndarray],
    remaining_ids: List[str],
    client: PassiveClient,
    feature_pairs: List[Tuple[str, str]],
    config: PhantomVFLConfig
) -> Tuple[bool, Dict]:
    """
    Validate all bivariate relationships for given remaining IDs.
    
    Args:
        original_data: Full dataset {feature: values}
        remaining_ids: IDs after removal
        client: Client with feature types
        feature_pairs: List of (feature1, feature2) tuples to validate
        config: Configuration
        
    Returns:
        all_valid: True if all pairs pass validation
        bivariate_metrics: Dict of metrics for each pair
    """
    # Prepare reduced data
    reduced_data = {
        feature: np.array([client.data[id][feature] for id in remaining_ids])
        for feature in client.features
    }
    
    all_valid = True
    bivariate_metrics = {}
    
    for f1, f2 in feature_pairs:
        type1 = client.feature_types.get(f1, 'continuous')
        type2 = client.feature_types.get(f2, 'continuous')
        
        valid, metrics = validate_bivariate_relationship(
            original_data, reduced_data, f1, f2, type1, type2, config
        )
        
        bivariate_metrics[f"{f1[:3]}_{f2[:3]}"] = metrics
        
        if not valid:
            all_valid = False
    
    return all_valid, bivariate_metrics


# ============================================================================
# PART 4C: BINARY SEARCH FUNCTIONS
# ============================================================================

def binary_search_marginal_capacity(
    client: PassiveClient,
    feature: str,
    max_budget: int,
    config: PhantomVFLConfig,
    rng: np.random.RandomState
) -> Tuple[int, List[Dict]]:
    """
    Use binary search to find maximum removal capacity for a single feature.
    
    Complexity: O(log n) iterations vs O(n) for incremental search.
    
    Args:
        client: Client with data
        feature: Feature name to validate
        max_budget: Maximum possible removals
        config: Configuration
        rng: Random stream of the client's Phase 2 search

    Returns:
        capacity: Maximum removals that preserve distribution
        search_history: List of tested k values and results
    """
    left, right = 0, max_budget
    capacity = 0
    search_history = []
    
    while left <= right:
        mid = (left + right) // 2
        
        if mid == 0:
            # Edge case: can't remove 0
            left = mid + 1
            continue
        
        # Test at mid with robust validation
        valid, metrics = robust_validate_marginal(client, feature, mid, config, rng)
        
        search_history.append({
            'k': mid,
            'valid': valid,
            'metrics': metrics
        })
        
        if valid:
            # This k works, try larger
            capacity = mid
            left = mid + 1
        else:
            # This k fails, try smaller
            right = mid - 1
    
    return capacity, search_history


def incremental_search_marginal_capacity(
    client: PassiveClient,
    feature: str,
    max_budget: int,
    config: PhantomVFLConfig,
    rng: np.random.RandomState
) -> Tuple[int, List[Dict]]:
    """
    Use incremental search to find maximum removal capacity for a single feature.
    
    Complexity: O(n) iterations - slower but more robust to sampling noise.
    Kept for comparison and validation purposes.
    
    Args:
        client: Client with data
        feature: Feature name to validate
        max_budget: Maximum possible removals
        config: Configuration
        rng: Random stream of the client's Phase 2 search

    Returns:
        capacity: Maximum removals that preserve distribution
        search_history: List of tested k values and results
    """
    capacity = 0
    search_history = []
    
    for k in range(1, max_budget + 1):
        valid, metrics = robust_validate_marginal(client, feature, k, config, rng)
        
        search_history.append({
            'k': k,
            'valid': valid,
            'metrics': metrics
        })
        
        if valid:
            capacity = k
        else:
            # First failure - stop
            break
    
    return capacity, search_history


def binary_search_bivariate_capacity(
    client: PassiveClient,
    max_budget: int,
    feature_pairs: List[Tuple[str, str]],
    config: PhantomVFLConfig,
    rng: np.random.RandomState
) -> Tuple[int, Dict, List[Dict]]:
    """
    Use binary search to find maximum removal capacity satisfying bivariate constraints.
    
    Args:
        client: Client with data
        max_budget: Maximum possible removals (typically min_capacity_marginal)
        feature_pairs: List of (feature1, feature2) tuples
        config: Configuration
        rng: Random stream of the client's Phase 2 search

    Returns:
        capacity: Maximum removals preserving all bivariate relationships
        bivariate_metrics: Metrics for feature pairs at final capacity
        search_history: List of all tested k values with validation results
    """
    # Prepare original data once
    original_data = {
        feature: np.array([client.data[id][feature] for id in client.matched_ids])
        for feature in client.features
    }
    
    left, right = 0, max_budget
    capacity = 0
    final_metrics = {}
    search_history = []
    
    print(f"    Binary search for bivariate capacity (max={max_budget})...")
    
    while left <= right:
        mid = (left + right) // 2
        
        if mid == 0:
            left = mid + 1
            continue
        
        # Test removal at mid: one random removal for each trial. A trial
        # passes if all pairs pass; mid passes if most trials pass (one trial:
        # its result). The metrics are those of the first trial.
        trial_results = []
        metrics = None
        for trial in range(config.validation_trials_per_k):
            # A set gives a fast membership test; the draw is the same
            test_removal = set(rng.choice(client.matched_ids, size=mid, replace=False).tolist())
            remaining_ids = [id for id in client.matched_ids if id not in test_removal]

            # Validate all bivariate relationships
            trial_valid, trial_metrics = robust_validate_bivariate(
                original_data, remaining_ids, client, feature_pairs, config
            )
            trial_results.append(trial_valid)
            if metrics is None:
                metrics = trial_metrics
        if config.validation_trials_per_k == 1:
            all_valid = trial_results[0]
        else:
            all_valid = sum(trial_results) > len(trial_results) // 2

        # Record this iteration in search history (trials_passed only with more than one trial)
        history_entry = {
            'k': mid,
            'all_valid': all_valid,
            'n_failed_pairs': len([p for p, m in metrics.items() if not m['valid']]),
            'failed_pairs': [pair for pair, m in metrics.items() if not m['valid']]
        }
        if config.validation_trials_per_k > 1:
            history_entry['trials_passed'] = int(sum(trial_results))
        search_history.append(history_entry)
        
        if all_valid:
            # All pairs valid, try larger
            capacity = mid
            final_metrics = metrics
            left = mid + 1
            if config.validation_trials_per_k == 1:
                print(f"      k={mid}: ✓ All {len(feature_pairs)} pairs valid")
            else:
                print(f"      k={mid}: ✓ passed in {sum(trial_results)} of {len(trial_results)} trials")
        else:
            # Some pair failed, try smaller
            right = mid - 1
            if config.validation_trials_per_k == 1:
                failed_pairs = [pair for pair, m in metrics.items() if not m['valid']]
                print(f"      k={mid}: ✗ {len(failed_pairs)} pair(s) failed")
            else:
                print(f"      k={mid}: ✗ passed in {sum(trial_results)} of {len(trial_results)} trials")
    
    return capacity, final_metrics, search_history


# ============================================================================
# PART 5: PHASE 2 - PHANTOM BUDGET COMPUTATION
# ============================================================================

def compute_removal_capacity_per_client(
    client: PassiveClient,
    config: PhantomVFLConfig,
    rng: np.random.RandomState
) -> Tuple[int, Dict]:
    """
    Phase 2: Compute maximum removal capacity for a single client.
    
    For each feature, determine maximum individuals that can be removed
    while preserving marginal distribution. Then search for maximum capacity
    satisfying all bivariate relationships.
    
    Args:
        client: Client with data and feature types
        config: Configuration
        rng: Random stream of this client's Phase 2 search (trial removals)

    Returns:
        capacity: Maximum number of individuals client can remove
        history: Validation results summary (includes bivariate metrics and search history)
    """
    trials = config.validation_trials_per_k
    if isinstance(trials, bool) or not isinstance(trials, int) or trials < 1 or trials % 2 == 0:
        raise ValueError(f"validation_trials_per_k (phase2_trials) must be an odd whole number "
                         f"(1, 3, 5, ...), got {trials!r}")
    n_matched = client.n_matched
    max_budget = int(config.max_removal_fraction * n_matched) # max_budget is the maximum number of individuals that can be removed (config.max_removal_fraction)
    
    print(f"\n{client.client_id}: Computing removal capacity...")
    print(f"  Matched IDs: {n_matched}")
    print(f"  Max removal fraction: {config.max_removal_fraction}")
    print(f"  Testing up to: {max_budget} removals")
    print(f"  Search method: {'BINARY' if config.use_binary_search else 'INCREMENTAL'}")
    print(f"  Trials for each tested budget (phase2_trials): {config.validation_trials_per_k}")
    print(f"  Bivariate validation: {'ENABLED' if config.enable_bivariate_validation else 'DISABLED'}")
    print(f"  Cramér's V Yates correction (2 x 2 tables): {'ON' if config.cramers_v_yates else 'OFF'}\n")
    
    feature_capacities = {} # a dictionary to store capacity for each feature
    search_histories = {}
    
    # STEP 1: Marginal distribution constraints (per feature)
    print(f"  STEP 1: Marginal Distribution Validation")
    for feature in client.features:
        print(f"    Searching capacity for '{feature}'...", end=" ")
        
        if config.use_binary_search:
            capacity, history = binary_search_marginal_capacity(
                client, feature, max_budget, config, rng
            )
        else:
            capacity, history = incremental_search_marginal_capacity(
                client, feature, max_budget, config, rng
            )
        
        feature_capacities[feature] = capacity
        search_histories[feature] = history
        
        search_iterations = len(history)
        print(f"→ {capacity} ({100*capacity/n_matched:.1f}%) [{search_iterations} iterations]")
    
    # Minimum capacity from marginal constraints
    min_capacity_marginal = min(feature_capacities.values())
    print(f"\n  Marginal-only capacity: {min_capacity_marginal} ({100*min_capacity_marginal/n_matched:.1f}%)")
    
    # STEP 2: Bivariate relationship validation
    bivariate_capacity = min_capacity_marginal  # Start with marginal capacity
    bivariate_metrics = {}
    bivariate_search_history = []
    
    if config.enable_bivariate_validation and len(client.features) >= 2:
        print(f"\n  STEP 2: Bivariate Relationship Validation")
        
        # Generate all feature pairs
        feature_pairs = []
        for i, f1 in enumerate(client.features):
            for f2 in client.features[i+1:]:
                feature_pairs.append((f1, f2))
        
        print(f"    Total feature pairs to validate: {len(feature_pairs)}")
        
        # Binary search for the bivariate capacity (at most the marginal capacity)
        bivariate_capacity, bivariate_metrics, bivariate_search_history = binary_search_bivariate_capacity(
            client=client,
            max_budget=min_capacity_marginal,  # Can't exceed marginal capacity
            feature_pairs=feature_pairs,
            config=config,
            rng=rng
        )
        
        # Report results
        print(f"\n  Bivariate capacity: {bivariate_capacity} ({100*bivariate_capacity/n_matched:.1f}%)")
        
        if bivariate_capacity < min_capacity_marginal:
            reduction = min_capacity_marginal - bivariate_capacity
            reduction_pct = 100 * reduction / min_capacity_marginal
            print(f"   Note: the bivariate checks reduced the capacity by {reduction} individuals ({reduction_pct:.1f}%)")
        else:
            print(f"    ✓ All bivariate relationships preserved at marginal capacity")
        
        # Show which pairs were most restrictive (if any failed at final capacity)
        violated_pairs = [
            pair for pair, metrics in bivariate_metrics.items()
            if not metrics.get('valid', True)
        ]
        if violated_pairs:
            print(f"    Most restrictive pairs: {violated_pairs[:3]}")
    
    # Final capacity
    final_capacity = bivariate_capacity if config.enable_bivariate_validation else min_capacity_marginal
    client.removal_capacity = final_capacity
    
    print(f"\n  {client.client_id} FINAL removal capacity: {final_capacity} ({100*final_capacity/n_matched:.1f}%)")
    
    # Compute total iterations across all searches
    total_iterations = sum(len(h) for h in search_histories.values())
    if config.enable_bivariate_validation:
        # Estimate bivariate iterations (binary search depth)
        bivariate_iterations = int(np.log2(min_capacity_marginal) + 1) if min_capacity_marginal > 0 else 0
        total_iterations += bivariate_iterations
    
    history = {
        'feature_capacities': feature_capacities,
        'search_histories': search_histories,
        'search_method': 'binary' if config.use_binary_search else 'incremental',
        'min_capacity_marginal': min_capacity_marginal,
        'bivariate_capacity': bivariate_capacity,
        'bivariate_search_history': bivariate_search_history if config.enable_bivariate_validation else [],
        'final_capacity': final_capacity,
        'removal_rate': final_capacity / n_matched,
        'bivariate_metrics': bivariate_metrics if config.enable_bivariate_validation else None,
        'total_iterations': total_iterations
    }
    
    return final_capacity, history


def negotiate_global_phantom_budget(
    clients: List[PassiveClient],
    config: PhantomVFLConfig # configuration (not read; the call sites pass it)
) -> int:
    """
    Negotiate global phantom budget across all clients.
    Takes minimum of all client removal capacities.
    
    This computes the NATURAL budget based on distribution
    preservation. For an imposed k level, the natural budget is only
    recorded (run_k_level).
    """
    print("\n" + "="*70)
    print("PHASE 2: GLOBAL PHANTOM BUDGET NEGOTIATION")
    print("="*70 + "\n")
    
    capacities = [client.removal_capacity for client in clients]
    natural_budget = min(capacities)
    
    print(f"Client removal capacities (natural constraints):")
    for i, (client, capacity) in enumerate(zip(clients, capacities)):
        print(f"  {client.client_id}: {capacity}")
    
    print(f"\nNatural global phantom budget: {natural_budget} (minimum)")
    
    print("   k*: this budget is used for removal; imposed k levels: recorded only\n")
    
    return natural_budget


# ============================================================================
# PART 6: PHASE 3 - INDEPENDENT REMOVAL (CLIENTS ONLY)
# ============================================================================

def independent_removal(
    client: PassiveClient,
    budget: int,
    config: PhantomVFLConfig, # label_sharing selects the sampling rule
    rng: np.random.RandomState
) -> Tuple[List[str], List[str]]:
    """
    Phase 3: Each client independently removes individuals (SECRET).
    
    Label sharing on: stratified random sampling by encoded label. Each label
    class loses its share of the budget, so the label distribution is
    preserved. Rounding is corrected so that exactly `budget` IDs are removed.
    Label sharing off: simple random sample of `budget` IDs. No labels are used.

    Budget:
    - Natural mode: budget = global budget from Phase 2 (k*)
    - Experimental mode: budget = int(k * n_matched)
    
    Server does NOT participate - it has labels for all individuals.
    
    Args:
        client: PassiveClient (with encoded_labels from Phase 0 if label sharing is on)
        budget: Number of individuals to remove
        config: Configuration (label_sharing)
        rng: Random stream of this client's removal
    
    Returns:
        training_ids: IDs kept for training
        removed_ids: IDs removed (SECRET - not shared with other parties)
    """
    if not check_label_sharing(config):
        # RANDOM REMOVAL: the client has no label information
        print(f"{client.client_id}: Removing {budget} individuals using RANDOM sampling (secret)...")
        removed_ids = rng.choice(
            client.matched_ids,
            size=budget,
            replace=False
        ).tolist()
        print(f"  Randomly removed {len(removed_ids)} individuals (no label stratification)")
        client.set_removal_set(removed_ids)
        # Random removal keeps the label distribution in expectation; this branch does not check it
        print(f"\n  Kept: {len(client.training_ids)} IDs")
        print(f"  Removed: {len(removed_ids)} IDs (SECRET)\n")
        return client.training_ids, removed_ids

    print(f"{client.client_id}: Removing {budget} individuals using STRATIFIED sampling (secret)...")
    
    # STRATIFIED REMOVAL: Preserve label distribution
    # Group matched IDs by their encoded label
    label_groups = defaultdict(list)
    for id_ in client.matched_ids:
        encoded_label = client.encoded_labels[id_]
        label_groups[encoded_label].append(id_)
    
    # Compute label distribution from full matched set
    total_matched = len(client.matched_ids)
    label_distribution = {
        label: len(ids) / total_matched 
        for label, ids in label_groups.items()
    }
    
    print(f"  Label distribution in matched set:")
    for label, proportion in label_distribution.items():
        print(f"    {label}: {len(label_groups[label])} ({100*proportion:.1f}%)")
    
    # Stratified removal: remove proportionally from each label class
    removed_ids = []
    removal_per_label = {}
    
    for label, ids in label_groups.items():
        # Calculate how many to remove from this label class
        n_to_remove_float = budget * label_distribution[label]
        n_to_remove = int(np.round(n_to_remove_float))
        
        # Ensure we don't exceed available IDs in this class
        n_to_remove = min(n_to_remove, len(ids))
        
        removal_per_label[label] = n_to_remove
        
        # Random sample from this label class
        removed_from_class = rng.choice(
            ids,
            size=n_to_remove,
            replace=False
        ).tolist()
        
        removed_ids.extend(removed_from_class)
    
    # Handle rounding errors: ensure total removal equals budget exactly
    total_removed = len(removed_ids)
    if total_removed < budget:
        # Need to remove a few more - take randomly from any class
        remaining_ids = [id_ for id_ in client.matched_ids if id_ not in removed_ids]
        additional_removal = rng.choice(
            remaining_ids,
            size=budget - total_removed,
            replace=False
        ).tolist()
        removed_ids.extend(additional_removal)
    elif total_removed > budget:
        # Removed too many - randomly keep some back
        removed_ids = rng.choice(
            removed_ids,
            size=budget,
            replace=False
        ).tolist()
    
    print(f"  Stratified removal breakdown:")
    for label, n_removed in removal_per_label.items():
        original_count = len(label_groups[label])
        remaining = original_count - n_removed
        print(f"    {label}: removed {n_removed}/{original_count} (kept {remaining})")
    
    # Set removal results
    client.set_removal_set(removed_ids)
    
    # Verify label distribution is preserved in kept set
    kept_label_counts = defaultdict(int)
    for id_ in client.training_ids:
        kept_label_counts[client.encoded_labels[id_]] += 1
    
    print(f"\n  Label distribution in KEPT set (after stratified removal):")
    for label in label_groups.keys():
        kept_count = kept_label_counts[label]
        kept_proportion = kept_count / len(client.training_ids)
        original_proportion = label_distribution[label]
        drift = abs(kept_proportion - original_proportion)
        print(f"    {label}: {kept_count} ({100*kept_proportion:.1f}%) [drift: {100*drift:.2f}%]")
    
    print(f"\n  Kept: {len(client.training_ids)} IDs")
    print(f"  Removed: {len(removed_ids)} IDs (SECRET)")
    print(f"  ✓ Label distribution preserved through stratified sampling\n")
    
    return client.training_ids, removed_ids


# ============================================================================
# PART 7: PHASE 4 - SERVER PROPOSES TRAINING ORDER (BLIND SELECTION)
# ============================================================================

def server_propose_training_order(
    server: Server,
    training_size: int,
    config: PhantomVFLConfig, # not used here (common signature of the phase functions)
    rng: np.random.RandomState
) -> List[str]:
    """
    Phase 4: Server randomly selects training instances from ALL matched IDs.
    
    CRITICAL: Server does NOT know which IDs clients removed.
    Server blindly selects training_size IDs from the full PSI result.
    
    Some selections will be IDs that clients removed → phantom positions.
    Clients will silently substitute without telling server.
    
    Args:
        server: Server with labels for all matched_ids
        training_size: Number of training instances to create
        config: Configuration
        rng: Random stream of the server order
    
    Returns:
        training_order: List of IDs for training (length = training_size)
    """
    print("=" * 70)
    print("PHASE 4: SERVER PROPOSES TRAINING ORDER (BLIND SELECTION)")
    print("=" * 70 + "\n")
    
    # Server randomly selects from ALL matched IDs
    # (Server doesn't know what clients removed)
    server.training_order = rng.choice(
        server.matched_ids, # full PSI result IDs
        size=training_size, # number of training instances
        replace=False # this ensures unique selection without replacement
    ).tolist() # converting to list for easier handling
    
    print(f"Server selected {training_size} training instances from {len(server.matched_ids)} total")
    print(f"Server has labels for ALL positions")
    print(f"Server DOES NOT KNOW which positions will be phantoms for clients\n")
    
    return server.training_order # list of IDs selected for training (length = training_size) some of which are phantoms


# ============================================================================
# PART 7B: PHASE 4b HELPERS (SUBSTITUTION POT AND REVIEWER CHECK)
# ============================================================================

def build_substitution_pot(client: PassiveClient, training_order: List[str],
                           rng: np.random.RandomState) -> List[str]:
    """
    Build the substitution pot of one client and shuffle it.

    Pot = the client's retained IDs that the server did not request.
    Each client removes as many IDs as the server leaves out of the training
    order, so the pot has exactly one ID for each phantom position of the client.

    The pot is sorted before the shuffle, so the result depends only on the
    random stream rng (the client's substitution stream).
    """
    requested = set(training_order)
    pot = sorted(id_ for id_ in client.training_ids if id_ not in requested)
    return [pot[i] for i in rng.permutation(len(pot))]


def find_phantom_positions(client: PassiveClient, training_order: List[str]) -> List[int]:
    """Return the positions of the training order whose ID the client removed."""
    return [
        position for position, requested_id in enumerate(training_order)
        if requested_id in client.removed_ids
    ]


def verify_phantom_mapping(client: PassiveClient, training_order: List[str]) -> Dict[str, any]:
    """
    Reviewer check: confirm that the client uses each ID once only.

    Rules:
      1. Every position of the training order has a mapping.
      2. Each ID appears at most once in the mapping.
      3. Each real position keeps the requested ID.
      4. Each phantom position gets a retained ID that the server did not request.
      5. No removed ID appears in the mapping.
    Also: client.phantom_positions equals the positions of the removed IDs.

    A failure stops the run. The error names the client and the position.
    The check prints one summary line for the client.

    Returns:
        {'positions', 'unique_ids', 'duplicate_ids', 'phantom_positions', 'verified'}
    """
    mapping = client.phantom_mapping
    n_positions = len(training_order)
    requested = set(training_order)
    retained = set(client.training_ids)

    if len(mapping) != n_positions:
        raise ValueError(
            f"{client.client_id}: mapping has {len(mapping)} positions, "
            f"training order has {n_positions}"
        )

    first_position_of_id = {}
    phantom_positions = []
    for position, requested_id in enumerate(training_order):
        if position not in mapping:
            raise ValueError(f"{client.client_id}: position {position} has no mapping")
        mapped_id = mapping[position]
        if mapped_id in first_position_of_id:
            raise ValueError(
                f"{client.client_id}: ID {mapped_id} appears at positions "
                f"{first_position_of_id[mapped_id]} and {position}"
            )
        first_position_of_id[mapped_id] = position
        if mapped_id in client.removed_ids:
            raise ValueError(f"{client.client_id}: removed ID {mapped_id} appears at position {position}")
        if requested_id in client.removed_ids:
            phantom_positions.append(position)
            if mapped_id not in retained or mapped_id in requested:
                raise ValueError(
                    f"{client.client_id}: phantom position {position} has ID {mapped_id}, "
                    f"which is not a retained ID outside the training order"
                )
        elif mapped_id != requested_id:
            raise ValueError(
                f"{client.client_id}: real position {position} has ID {mapped_id}, "
                f"not the requested ID {requested_id}"
            )

    if list(client.phantom_positions) != phantom_positions:
        raise ValueError(
            f"{client.client_id}: recorded phantom positions differ from the positions "
            f"of the removed IDs"
        )

    unique_ids = len(first_position_of_id)
    duplicate_ids = n_positions - unique_ids
    print(
        f"  Reviewer check {client.client_id}: positions {n_positions:,}, "
        f"unique IDs {unique_ids:,}, duplicates {duplicate_ids}, "
        f"phantom positions {len(phantom_positions):,}: passed"
    )
    return {
        'positions': n_positions,
        'unique_ids': unique_ids,
        'duplicate_ids': duplicate_ids,
        'phantom_positions': len(phantom_positions),
        'verified': True
    }


# ============================================================================
# PART 8: PHASE 4b - SILENT SUBSTITUTION (ONE-TO-ONE MAPPING)
# ============================================================================

def silent_substitution(
    server: Server, # server with the training order [from Phase 4 above]
    clients: List[PassiveClient],
    config: PhantomVFLConfig, # label_sharing selects the pick rule
    k_proportion: Optional[float] = None
) -> Tuple[Dict[str, Dict[int, str]], Dict[str, any]]:
    """
    Phase 4b: Each client creates COMPLETE one-to-one phantom mapping.

    For each position in server's training_order:
      - If ID was kept: map position → same ID (real)
      - If ID was removed: map position → different ID from pot (phantom)

    Pot (build_substitution_pot): the client's retained IDs that the server
    did not request, shuffled with the client's substitution stream. It has
    exactly one ID for each phantom position.

    Pick rule with label sharing on (label-aware): the phantom positions are
    handled in a shuffled order. Each position takes the next pot ID with the
    SAME encoded label. If no such ID is left, it takes the next pot ID with
    any label (label mismatch) when config.allow_label_mismatch_fallback is
    True; otherwise the run stops. The shuffled order spreads the mismatches
    over the training order.

    Pick rule with label sharing off (random): the phantom positions take the
    shuffled pot IDs in position order, so each substitute is a uniform random
    draw without reuse. No label information is used.

    CRITICAL PROPERTIES:
      - One-to-one mapping (no ID reuse), confirmed by verify_phantom_mapping
      - Silent (server doesn't know which positions are phantoms)

    Args:
        server: Server with training_order
        clients: List of clients with removed_ids (and encoded_labels if label sharing is on)
        config: Configuration (label_sharing, random_seed, allow_label_mismatch_fallback)
        k_proportion: Imposed k, or None for k* (selects the random stream)

    Returns:
        mappings: {client_id: {position: real_id}} for all clients
        metrics: Substitution statistics (phantom rates, reviewer check, and
                 label match rates if label sharing is on)
    """
    label_sharing = check_label_sharing(config)
    print("=" * 70)
    if label_sharing:
        print("PHASE 4b: LABEL-AWARE SILENT SUBSTITUTION")
        print("=" * 70 + "\n")
    else:
        print("PHASE 4b: RANDOM SILENT SUBSTITUTION")
        print("=" * 70 + "\n")
        print("Note: the substitution does not use labels\n")

    mappings = {}
    client_metrics = {}
    training_order = server.training_order

    for client_index, client in enumerate(clients):
        print(f"\n{client.client_id}: Creating phantom mapping...")

        rng = stage_rng(config, STAGE_SUBSTITUTION, client_index, k_proportion)
        pot = build_substitution_pot(client, training_order, rng)
        phantom_positions = find_phantom_positions(client, training_order)
        if len(pot) != len(phantom_positions):
            raise ValueError(
                f"{client.client_id}: pot has {len(pot)} IDs for "
                f"{len(phantom_positions)} phantom positions"
            )

        if label_sharing:
            # Real positions keep the requested ID
            position_mapping = {
                position: requested_id
                for position, requested_id in enumerate(training_order)
                if requested_id not in client.removed_ids
            }

            # Pot split by encoded label; each queue keeps the shuffled pot order
            pot_queue = deque(pot)
            label_queues = defaultdict(deque)
            for id_ in pot:
                label_queues[client.encoded_labels[id_]].append(id_)
            used_ids = set()

            def next_unused(queue):
                while queue and queue[0] in used_ids:
                    queue.popleft()
                return queue.popleft() if queue else None

            label_match_count = 0
            label_mismatch_count = 0
            for i in rng.permutation(len(phantom_positions)):
                position = phantom_positions[i]
                requested_label = client.encoded_labels[training_order[position]]
                substitute_id = next_unused(label_queues[requested_label])
                if substitute_id is not None:
                    label_match_count += 1
                elif config.allow_label_mismatch_fallback:
                    substitute_id = next_unused(pot_queue)
                    label_mismatch_count += 1
                else:
                    raise ValueError(
                        f"{client.client_id}: no pot ID with label {requested_label} left at "
                        f"position {position}, and allow_label_mismatch_fallback is False"
                    )
                used_ids.add(substitute_id)
                position_mapping[position] = substitute_id

            # Store results (mapping in position order)
            client.phantom_mapping = {
                position: position_mapping[position] for position in range(len(training_order))
            }
        else:
            # Each phantom position takes the next ID of the shuffled pot
            substitutes = dict(zip(phantom_positions, pot))
            client.phantom_mapping = {
                position: substitutes.get(position, requested_id)
                for position, requested_id in enumerate(training_order)
            }
        client.phantom_positions = phantom_positions
        mappings[client.client_id] = client.phantom_mapping

        check = verify_phantom_mapping(client, training_order)

        # Metrics (label matching: label sharing on only)
        phantom_count = len(phantom_positions)
        phantom_rate = phantom_count / len(training_order)
        entry = {'phantom_count': phantom_count, 'phantom_rate': phantom_rate}
        if label_sharing:
            label_match_rate = label_match_count / phantom_count if phantom_count > 0 else 0.0
            entry['label_match_count'] = label_match_count
            entry['label_mismatch_count'] = label_mismatch_count
            entry['label_match_rate'] = label_match_rate
        else:
            entry['label_match_rate'] = 'N/A - no label access'
        entry['pot_size'] = len(pot)
        entry['duplicate_ids'] = check['duplicate_ids']
        entry['verified'] = check['verified']
        client_metrics[client.client_id] = entry

        print(f"  Phantom positions: {phantom_count}/{len(training_order)} ({100*phantom_rate:.1f}%)")
        if label_sharing:
            print(f"  Label matches: {label_match_count}/{phantom_count} ({100*label_match_rate:.1f}%)")
            if label_mismatch_count > 0:
                print(f"  [!] {client.client_id}: label mismatches: {label_mismatch_count} (pot constraints)")

    # Overall metrics
    total_phantom = sum(m['phantom_count'] for m in client_metrics.values())
    total_positions = len(training_order) * len(clients)
    overall_phantom_rate = total_phantom / total_positions

    metrics = {
        'client_metrics': client_metrics,
        'overall_phantom_rate': overall_phantom_rate
    }
    if label_sharing:
        total_matches = sum(m['label_match_count'] for m in client_metrics.values())
        total_mismatches = sum(m['label_mismatch_count'] for m in client_metrics.values())
        overall_label_match_rate = total_matches / total_phantom if total_phantom > 0 else 0.0
        metrics['overall_label_match_rate'] = overall_label_match_rate
        metrics['total_phantom_positions'] = total_phantom
        metrics['total_label_matches'] = total_matches
        metrics['total_label_mismatches'] = total_mismatches
    else:
        metrics['overall_label_match_rate'] = 'N/A - no label access'
        metrics['total_phantom_positions'] = total_phantom
    metrics['all_verified'] = all(m['verified'] for m in client_metrics.values())

    print("\n" + "="*70)
    print("SUBSTITUTION SUMMARY")
    print("="*70)
    print(f"Overall phantom rate: {100*overall_phantom_rate:.1f}%")
    if label_sharing:
        print(f"Overall label match rate: {100*overall_label_match_rate:.1f}%")
        print(f"Total phantom positions: {total_phantom}")
        print(f"Total label matches: {total_matches}")
        print(f"Total label mismatches: {total_mismatches}")
    else:
        print(f"Label matching: N/A (no label access)")
        print(f"Total phantom positions: {total_phantom}")
    print(f"Reviewer check: each client uses each ID once only (all clients passed)\n")

    return mappings, metrics


# ============================================================================
# PART 9: PHASE 4c - GROUND TRUTH TRACKING (EXPERIMENTAL VALIDATION)
# ============================================================================

class GroundTruthTracker:
    """
    Tracks ground truth about real vs phantom individuals for
    experimental validation.
    
    WARNING: This module is for debugging/validation only. In production
    deployment, this should be DISABLED to maintain the honest-but-curious
    adversary model where the server does not know ground truth.
    
    Creates two outputs:
    1. ground_truth_analysis.csv - Detailed per-ID inspection
    2. phantom_rate_summary.json - Aggregate statistics
    """
    
    def __init__(self, config: PhantomVFLConfig, output_dir: str, enabled: bool = True):
        self.config = config
        self.output_dir = output_dir  # output folder of the k level
        self.enabled = enabled
        
    def track_post_phase4b(
        self,
        server: Server,
        clients: List[PassiveClient],
        natural_budget: int,
        actual_budget: int,
        expected_phantom_rate: float,
        timing: TimingMetrics = None
    ) -> Tuple[pd.DataFrame, PhantomRateStatistics]:
        """
        Track ground truth after Phase 4b completion.
        
        
        Args:
            server: Server with training_order
            clients: Clients with removed_ids
            natural_budget: Natural budget from Phase 2
            actual_budget: Actual budget used (may differ in experimental mode)
            expected_phantom_rate: Expected phantom rate from Phase 2
            timing: Execution timing metrics
            
        Returns:
            df_ground_truth: DataFrame with per-ID analysis
            stats: Aggregate statistics with timing
        """
        if not self.enabled:
            print("Ground truth tracking disabled\n")
            return None, None
        
        print("\n" + "="*70)
        print("PHASE 4c: GROUND TRUTH TRACKING (EXPERIMENTAL VALIDATION)")
        print("="*70 + "\n")
        print("Note: this module is for experimental validation only")
        print("   In production, this should be DISABLED\n")
        
        # Collect removal information from all clients
        all_removed_ids = set()
        client_removal_counts = {}
        
        for client in clients:
            all_removed_ids.update(client.removed_ids)
            client_removal_counts[client.client_id] = len(client.removed_ids)
        
        # Analyze each ID in the matched set
        records = []
        for id in server.matched_ids:
            # Check which clients removed this ID
            removed_by_clients = [
                client.client_id for client in clients
                if id in client.removed_ids
            ]
            
            is_phantom = len(removed_by_clients) > 0
            removed_by_count = len(removed_by_clients)
            removed_by_str = ','.join(removed_by_clients) if removed_by_clients else 'NONE'
            status = 'PHANTOM' if is_phantom else 'REAL'
            
            # Get label (original, not encoded)
            label = server.labels.get(id, None)
            
            records.append({
                'id': id,
                'is_phantom': is_phantom,
                'status': status,
                'removed_by_count': removed_by_count,
                'removed_by_clients': removed_by_str,
                'label': label
            })
        
        df_ground_truth = pd.DataFrame(records)
        
        # Compute statistics
        total_ids_after_psi = len(server.matched_ids)
        total_ids_after_removal = sum(len(client.training_ids) for client in clients) // len(clients)
        
        actual_phantom_count = df_ground_truth['is_phantom'].sum()
        actual_phantom_rate = actual_phantom_count / total_ids_after_psi
        
        # Breakdown by removal pattern
        removed_by_single = (df_ground_truth['removed_by_count'] == 1).sum()
        removed_by_multiple = (df_ground_truth['removed_by_count'] > 1).sum()
        removed_by_all = (df_ground_truth['removed_by_count'] == len(clients)).sum()
        
        # Expected phantom rates per client
        expected_rates = {
            client.client_id: len(client.removed_ids) / client.n_matched
            for client in clients
        }

        # Compute total time before saving
        if timing:
            timing.compute_totals()
        
        stats = PhantomRateStatistics(
            total_ids_after_psi=total_ids_after_psi,
            total_ids_after_removal=total_ids_after_removal,
            client_removal_counts=client_removal_counts,
            expected_phantom_rate_per_client=expected_rates,
            overall_expected_phantom_rate=expected_phantom_rate,
            actual_phantom_count=actual_phantom_count,
            actual_phantom_rate=actual_phantom_rate,
            removed_by_single_client=removed_by_single,
            removed_by_multiple_clients=removed_by_multiple,
            removed_by_all_clients=removed_by_all,
            # Timing metrics
            time_total=timing.total if timing else 0.0,
            time_phase0=timing.phase0 if timing else 0.0,
            time_phase1=timing.phase1 if timing else 0.0,
            time_phase2_total=timing.phase2_total if timing else 0.0,
            time_phase3_total=timing.phase3_total if timing else 0.0,
            time_phase4=timing.phase4 if timing else 0.0,
            time_phase4b_total=timing.phase4b_total if timing else 0.0
        )
        
        # Save outputs
        self._save_outputs(df_ground_truth, stats)
        
        # Print summary
        print("Ground Truth Summary:")
        print(f"  Total IDs after PSI: {total_ids_after_psi}")
        print(f"  Actual phantom count: {actual_phantom_count}")
        print(f"  Actual phantom rate: {100*actual_phantom_rate:.2f}%")
        print(f"  Expected phantom rate: {100*expected_phantom_rate:.2f}%")
        print(f"  Removed by single client: {removed_by_single}")
        print(f"  Removed by multiple clients: {removed_by_multiple}")
        print(f"  Removed by all clients: {removed_by_all}\n")
        
        return df_ground_truth, stats
    
    def _save_outputs(self, df_ground_truth: pd.DataFrame, stats: PhantomRateStatistics):
        """Save ground truth CSV and summary JSON"""
        
        # 1. Save detailed CSV for inspection
        csv_path = os.path.join(self.output_dir, "ground_truth_analysis.csv")
        df_ground_truth.to_csv(csv_path, index=False)
        print(f"✓ Ground truth CSV saved: {csv_path}")
        
        # 2. Save summary statistics JSON
        json_path = os.path.join(self.output_dir, "phantom_rate_summary.json")
        with open(json_path, 'w') as f:
            json.dump(stats.to_dict(), f, indent=2, default=convert_numpy)
        print(f"✓ Summary statistics saved: {json_path}\n")


# ============================================================================
# PART 10: EXPERIMENTAL K MODE EXECUTION
# ============================================================================

def run_experimental_k_configuration(
    k_proportion: float,
    server: Server,
    clients: List[PassiveClient],
    config: PhantomVFLConfig,
    output_dir: str,  # output folder of this k level (ground-truth files)
    main_timing: TimingMetrics = None  # timing of Phases 0 to 2 from the main protocol
) -> Dict[str, any]:
    """
    Run phantom protocol with specific experimental k proportion (with timing).
    
    This function executes Phases 3-4c with a controlled phantom proportion
    that may exceed natural distribution constraints. Used for systematic
    privacy-utility tradeoff analysis.
    
    Args:
        k_proportion: Phantom proportion (e.g., 0.20 for 20%)
        server: Server with matched_ids (and encoded_labels if label sharing is on)
        clients: Clients with matched_ids (and encoded_labels if label sharing is on)
        config: Configuration
        output_dir: Output folder of this k level (ground-truth files)
        
    Returns:
        results: Dictionary with phantom mappings, metrics, ground truth, and timing
    """
    # Timing for this k configuration; Phases 0 to 2 are copied from the main protocol
    k_timing = TimingMetrics()
    if main_timing:
        k_timing.phase0 = main_timing.phase0
        k_timing.phase1 = main_timing.phase1
        k_timing.phase2_total = main_timing.phase2_total
        k_timing.phase2_per_client = main_timing.phase2_per_client.copy()

    k_start = time.time()
    
    print("\n" + "="*70)
    print(f"EXPERIMENTAL K CONFIGURATION: k = {100*k_proportion:.0f}%")
    print("="*70 + "\n")
    
    n_matched = len(server.matched_ids)
    experimental_budget = int(k_proportion * n_matched)
    training_size = n_matched - experimental_budget
    
    print(f"Matched IDs: {n_matched}")
    print(f"Experimental budget: {experimental_budget} ({100*k_proportion:.1f}%)")
    print(f"Training size: {training_size}\n")
    
    # Phase 3: Independent removal with experimental budget
    for client_index, client in enumerate(clients):
        rng = stage_rng(config, STAGE_REMOVAL, client_index, k_proportion)
        phase3_client_start = time.time()
        independent_removal(client, experimental_budget, config, rng)
        k_timing.phase3_per_client[client.client_id] = time.time() - phase3_client_start
    
    # Phase 4: Server proposes training order
    rng = stage_rng(config, STAGE_SERVER_ORDER, SERVER_STREAM_INDEX, k_proportion)
    phase4_start = time.time()
    training_order = server_propose_training_order(server, training_size, config, rng)
    k_timing.phase4 = time.time() - phase4_start
    
    # Phase 4b: Silent substitution
    phase4b_start = time.time()
    mappings, substitution_metrics = silent_substitution(
        server, clients, config, k_proportion=k_proportion
    )
    k_timing.phase4b_total = time.time() - phase4b_start
    
    # Phase 4c: Ground truth tracking
    if config.enable_ground_truth_tracking:
        tracker = GroundTruthTracker(config, output_dir, enabled=True)
        df_ground_truth, phantom_stats = tracker.track_post_phase4b(
            server=server,
            clients=clients,
            natural_budget=None,  # Not applicable in experimental mode
            actual_budget=experimental_budget,
            expected_phantom_rate=k_proportion,
            timing=k_timing
        )
    else:
        df_ground_truth = None
        phantom_stats = None
    
    # Compute timing totals
    k_timing.compute_totals()
    
    print(f"\nTiming for k={100*k_proportion:.0f}%:")
    print(f"  Phase 3: {k_timing.phase3_total:.3f}s")
    print(f"  Phase 4: {k_timing.phase4:.3f}s")
    print(f"  Phase 4b: {k_timing.phase4b_total:.3f}s")
    print(f"  Total: {k_timing.total:.3f}s")
    
    results = {
        'k_proportion': k_proportion,
        'experimental_budget': experimental_budget,
        'training_size': training_size,
        'substitution_metrics': substitution_metrics,
        'ground_truth_df': df_ground_truth,
        'phantom_statistics': phantom_stats,
        'timing': k_timing.get_detailed_dict(),
        'server': server,    # the generation stage reads the training order from the server
        'clients': clients   # the generation stage reads the features and mappings from the clients
    }
    
    return results


# ============================================================================
# PART 11: COMPLETE PROTOCOL EXECUTION
# ============================================================================

def prepare_protocol(
    server_labels: Dict[str, float], # dictionary of {individual_id: label_value} for server
    client_data_list: List[Dict[str, Dict[str, float]]], # list of client data dictionaries [{individual_id: {feature: value}}] for each client
    client_ids: List[str], # list of client IDs ['client_1', 'client_2', ...]
    config: PhantomVFLConfig # Phantom VFL configuration
) -> Dict[str, any]:
    """
    Run Phases 0 to 2 once: parties and feature types, label sharing (Phase 0,
    label sharing on only), PSI (Phase 1) and the natural budget k* (Phase 2).

    The result is the input of run_k_level. The Phase 2 streams have no k
    level, so the natural budget is the same for every k level of one seed,
    and Phase 2 needs to run only once per seed.

    Args:
        server_labels: {individual_id: label_value}
        client_data_list: [{individual_id: {feature: value}}] for each client
        client_ids: ['client_1', 'client_2', ...]
        config: PhantomVFLConfig (label_sharing must be True or False)

    Returns:
        prepared: parties, common IDs, natural budget, timing of Phases 0 to 2,
                  and the shared part of the results (random_seed, setup, psi, budget)
    """
    label_sharing = check_label_sharing(config)

    # Initialize timing tracker
    timing = TimingMetrics()

    results = {} # shared part of the results of every k level
    results['random_seed'] = resolve_run_seed(config)
    print(f"Phantom generation seed: {results['random_seed']}\n")
    print(f"Label sharing: {'ON' if label_sharing else 'OFF'}\n")

    # Initialize parties
    server = Server(server_labels) # create server object with labels
    clients = [] # initialize empty list for clients
    
    for i, (client_id, client_data) in enumerate(zip(client_ids, client_data_list)): # iterate over client IDs and their corresponding data
        # Extract features from first individual's data
        sample_id = list(client_data.keys())[0]
        features = list(client_data[sample_id].keys())
        
        client = PassiveClient(client_id, features, client_data) # create PassiveClient object for each client (with ID, features and data)
        
        # FEATURE TYPES: one type for each feature, from config.feature_types
        missing = [feature for feature in features if feature not in config.feature_types]
        if missing:
            raise ValueError(f"{client_id}: no feature type for {missing}. "
                             f"config.feature_types needs a type for every feature.")
        wrong = {feature: config.feature_types[feature] for feature in features
                 if config.feature_types[feature] not in ("continuous", "categorical", "ordinal")}
        if wrong:
            raise ValueError(f"{client_id}: feature types must be 'continuous', 'categorical' or 'ordinal', got {wrong}")
        for feature in features:
            if config.feature_types[feature] != "ordinal":
                continue
            ranks = config.ordinal_ranks.get(feature)
            if not ranks:
                raise ValueError(f"{client_id}: ordinal feature '{feature}' has no ranks in config.ordinal_ranks")
            unknown = sorted({row[feature] for row in client_data.values()} - set(ranks))
            if unknown:
                raise ValueError(f"{client_id}: ordinal feature '{feature}' has values with no rank: {unknown[:10]}")
        print(f"\n{client_id}: feature types")
        for feature in features:
            client.feature_types[feature] = config.feature_types[feature]
            print(f"  {feature}: {client.feature_types[feature]}")
        
        clients.append(client)        

    # ========================================================================
    # PHASE 0: Setup - Encoded Label Sharing (label sharing on only)
    # ========================================================================
    if label_sharing:
        phase0_start = time.time()
        setup_info = setup_encoded_label_sharing(server, clients, config)
        timing.phase0 = time.time() - phase0_start
        results['setup'] = setup_info # store setup info in results
    else:
        print("Phase 0 (label sharing) does not run: the clients get no labels.\n")

    # ========================================================================
    # PHASE 1: Private Set Intersection
    # ========================================================================
    phase1_start = time.time()
    common_ids, psi_stats = private_set_intersection(server, clients)
    timing.phase1 = time.time() - phase1_start
    results['psi'] = psi_stats # store PSI stats in results
    
    # ========================================================================
    # PHASE 2: Phantom Budget Computation (Clients Only)
    # ========================================================================
    budget_histories = []
    for client_index, client in enumerate(clients): # loop over each client to compute removal capacity
        rng = stage_rng(config, STAGE_PHASE2_CAPACITY, client_index)
        phase2_client_start = time.time()
        capacity, history = compute_removal_capacity_per_client(client, config, rng) # compute capacity and history for each client
        timing.phase2_per_client[client.client_id] = time.time() - phase2_client_start
        budget_histories.append(history) # store history for later analysis
    
    natural_budget = negotiate_global_phantom_budget(clients, config) # compute natural global budget across clients
    natural_training_size = len(common_ids) - natural_budget
    natural_phantom_rate = natural_budget / len(common_ids)
    
    results['budget'] = {
        'client_capacities': [c.removal_capacity for c in clients], # individual client capacities
        'natural_budget': natural_budget, # natural phantom budget from distribution constraints
        'natural_training_size': natural_training_size, # training size if using natural budget
        'natural_phantom_rate': natural_phantom_rate, # expected phantom rate from natural budget
        'histories': budget_histories # detailed histories for each client
    }

    return {
        'config': config,
        'label_sharing': label_sharing,
        'server_labels': server_labels,
        'client_data_list': client_data_list,
        'client_ids': client_ids,
        'server': server,
        'clients': clients,
        'common_ids': common_ids,
        'natural_budget': natural_budget,
        'natural_training_size': natural_training_size,
        'natural_phantom_rate': natural_phantom_rate,
        'timing': timing,
        'results': results,
        'kstar_done': False
    }


def run_k_level(
    prepared: Dict[str, any], # result of prepare_protocol
    k_proportion: Optional[float], # None: k* (natural budget); a number: imposed k
    output_dir: str # output folder of this k level (ground-truth files)
) -> Dict[str, any]:
    """
    Run Phases 3 to 4c for one k level, on the result of prepare_protocol.

    k_proportion None: k* (natural mode). It uses the parties of `prepared` and
    changes them (removal), so k* can run once for each prepare_protocol call.
    k_proportion a number: an imposed k. It uses new copies of the parties, so
    the order of the k levels does not change any result.

    The ground-truth files (ground_truth_analysis.csv and
    phantom_rate_summary.json) go to output_dir.

    Returns:
        results in the form of phantom_vfl_complete_protocol for one k level:
        mode 'natural' for k*, mode 'experimental' with one entry otherwise
    """
    config = prepared['config']
    label_sharing = check_label_sharing(config)
    timing = prepared['timing']
    results = dict(prepared['results'])
    common_ids = prepared['common_ids']
    natural_budget = prepared['natural_budget']
    natural_training_size = prepared['natural_training_size']
    natural_phantom_rate = prepared['natural_phantom_rate']

    if k_proportion is None:
        if prepared['kstar_done']:
            raise ValueError("k* has already run on these parties; call prepare_protocol again")
        prepared['kstar_done'] = True
        server = prepared['server']
        clients = prepared['clients']

        # NATURAL MODE: Use computed natural budget
        print("\n" + "="*70)
        print("NATURAL MODE: Using distribution-preserving budget")
        print("="*70 + "\n")
        
        # Phase 3: Independent Removal
        print("=" * 70)
        print("PHASE 3: INDEPENDENT REMOVAL (SECRET, CLIENTS ONLY)")
        print("=" * 70 + "\n")
        print("Server does NOT remove (has labels for all individuals)\n")
        
        for client_index, client in enumerate(clients):
            rng = stage_rng(config, STAGE_REMOVAL, client_index)
            phase3_client_start = time.time()
            independent_removal(client, natural_budget, config, rng)
            timing.phase3_per_client[client.client_id] = time.time() - phase3_client_start
        
        # Phase 4: Server Proposes Training Order
        rng = stage_rng(config, STAGE_SERVER_ORDER, SERVER_STREAM_INDEX)
        phase4_start = time.time()
        training_order = server_propose_training_order(server, natural_training_size, config, rng)
        timing.phase4 = time.time() - phase4_start
        
        results['training_order'] = {
            'size': natural_training_size,
            'server_selection': 'random_from_all',
            'server_awareness': 'unaware_of_client_removals'
        }
        
        # Phase 4b: Silent Substitution
        phase4b_start = time.time()
        mappings, substitution_metrics = silent_substitution(
            server, clients, config, k_proportion=None
        )
        # Phase 4b is timed as a total only
        timing.phase4b_total = time.time() - phase4b_start
        
        results['substitution'] = substitution_metrics
        
        # Phase 4c: Ground Truth Tracking (if enabled)
        if config.enable_ground_truth_tracking:
            tracker = GroundTruthTracker(config, output_dir, enabled=True)
            df_ground_truth, phantom_stats = tracker.track_post_phase4b(
                server=server,
                clients=clients,
                natural_budget=natural_budget,
                actual_budget=natural_budget,
                expected_phantom_rate=natural_phantom_rate,
                timing=timing
            )
            results['ground_truth'] = {
                'df': df_ground_truth,
                'statistics': phantom_stats
            }
        
        # Summary
        print("\n" + "=" * 70)
        print("PHANTOM VFL PROTOCOL COMPLETED (Phases 0-4c)")
        print("=" * 70)
        print(f"\nResults Summary:")
        print(f"  Number of clients: {len(clients)}")
        print(f"  Common individuals (PSI): {results['psi']['common_ids']}")
        print(f"  Natural phantom budget: {results['budget']['natural_budget']}")
        print(f"  Training size: {results['budget']['natural_training_size']}")
        print(f"  Overall phantom rate: {100*substitution_metrics['overall_phantom_rate']:.1f}%")
        if label_sharing:
            print(f"  Label match rate: {100*substitution_metrics['overall_label_match_rate']:.1f}%")
        else:
            print(f"  Label match rate: {substitution_metrics['overall_label_match_rate']}")
        
        # Compute final timing metrics
        timing.compute_totals()
        
        print(f"\nExecution Timing:")
        print(f"  Phase 0 (Label Encoding):      {timing.phase0:.3f}s")
        print(f"  Phase 1 (PSI):                 {timing.phase1:.3f}s")
        print(f"  Phase 2 (Budget Computation):  {timing.phase2_total:.3f}s")
        for client_id, duration in timing.phase2_per_client.items():
            print(f"        {client_id}: {duration:.3f}s")
        print(f"  Phase 3 (Independent Removal): {timing.phase3_total:.3f}s")
        for client_id, duration in timing.phase3_per_client.items():
            print(f"        {client_id}: {duration:.3f}s")
        print(f"  Phase 4 (Training Order):      {timing.phase4:.3f}s")
        print(f"  Phase 4b (Substitution):       {timing.phase4b_total:.3f}s")
        print(f"  {'-' * 40}")
        print(f"  TOTAL:                         {timing.total:.3f}s")
        
        print(f"\nPrivacy Properties:")
        print(f"  ✓ Server unaware of phantom positions")
        print(f"  ✓ Clients never reveal removals")
        print(f"  ✓ Silent substitution preserves privacy")
        if label_sharing:
            print(f"  ✓ Label-aware substitution preserves utility")
        else:
            print(f"  ✓ Substitution uses no labels")
        
        results['mode'] = 'natural'
        results['server'] = server
        results['clients'] = clients

    else:
        # Create fresh copies of server and clients for this k
        # (needed because removal modifies state)
        server_k = Server(prepared['server_labels'].copy())
        server_k.set_matched_ids(common_ids)
        if label_sharing:
            server_k.create_label_encoding(config.label_encoding_type)

        clients_k = []
        for client_id, client_data in zip(prepared['client_ids'], prepared['client_data_list']):
            sample_id = list(client_data.keys())[0]
            features = list(client_data[sample_id].keys())
            client_k = PassiveClient(client_id, features, client_data)
            client_k.set_matched_ids(common_ids)
            if label_sharing:
                client_k.receive_encoded_labels(server_k.encoded_labels.copy())
            # Copy feature types from original client
            original_client = next(c for c in prepared['clients'] if c.client_id == client_id)
            client_k.feature_types = original_client.feature_types.copy()
            clients_k.append(client_k)

        # Run experimental configuration (with the timing of Phases 0 to 2)
        k_results = run_experimental_k_configuration(
            k_proportion=k_proportion,
            server=server_k,
            clients=clients_k,
            config=config,
            output_dir=output_dir,
            main_timing=timing  # Phase 0, 1, 2 timing of prepare_protocol
        )
        results['mode'] = 'experimental'
        results['experimental_results'] = [k_results]

    # Store timing metrics in results
    results['timing'] = timing.get_detailed_dict()

    return results


def phantom_vfl_complete_protocol(
    server_labels: Dict[str, float], # dictionary of {individual_id: label_value} for server
    client_data_list: List[Dict[str, Dict[str, float]]], # list of client data dictionaries [{individual_id: {feature: value}}] for each client
    client_ids: List[str], # list of client IDs ['client_1', 'client_2', ...]
    config: PhantomVFLConfig, # Phantom VFL configuration
    output_dir: str, # output folder of the k level (ground-truth files)
    k_proportion: Optional[float] = None # None: k* (natural budget); a number: imposed k
) -> Dict[str, any]:
    """
    Execute the Phantom VFL protocol (Phases 0-4c) for one k level:
    prepare_protocol, then run_k_level. Execution timing is recorded.

    To run several k levels of one seed, call prepare_protocol once and then
    run_k_level for each k level (Phase 2 then runs only once), as the phantom
    generation stage does.

    Phase 5 (VFL Training) and Phase 6 (Privacy Evaluation) are handled
    separately by the Flower framework.

    Returns:
        results of run_k_level: mode 'natural' for k*, mode 'experimental'
        with one entry for an imposed k
    """
    prepared = prepare_protocol(server_labels, client_data_list, client_ids, config)
    return run_k_level(prepared, k_proportion, output_dir)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("Phantom VFL protocol module.")
    print("This module defines the protocol; it does not generate datasets on its own.")
    print("Run generate_phantom_datasets.py to generate the phantom datasets.")
