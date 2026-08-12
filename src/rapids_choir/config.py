"""Validated, CUDA-independent runtime configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite

from ._enums import InputPolicy, Multiplicity, RuleSet
from .exceptions import InvalidConfigurationError


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidConfigurationError(f"{name} must be a positive integer")


def split_round_count(n_iterations: int) -> int:
    """Return CHOIR's ties-to-even ``round(n_iterations / 2)`` result."""

    _positive_int("n_iterations", n_iterations)
    return round(n_iterations / 2)


@dataclass(frozen=True, slots=True)
class RFProfile:
    """Every cuML random-forest parameter except fit-specific seed/tree count."""

    n_bins: int = 128
    max_depth: int | None = 16
    n_streams: int = 1
    split_criterion: str = "gini"
    max_features: str | float = "sqrt"
    bootstrap: bool = True
    min_samples_leaf: int = 1
    min_samples_split: int = 2
    min_impurity_decrease: float = 0.0
    max_leaves: int = -1
    max_samples: float = 1.0
    max_batch_size: int = 4096
    oob_score: bool = False
    verbose: bool = False
    output_type: str = "cupy"

    def __post_init__(self) -> None:
        if self.n_bins not in {128, 256, 512}:
            raise InvalidConfigurationError("n_bins must be one of 128, 256, 512")
        if self.max_depth not in {16, None}:
            raise InvalidConfigurationError("max_depth must be 16 or None")
        if self.n_streams not in {1, 4}:
            raise InvalidConfigurationError("n_streams must be 1 or 4")
        if self.split_criterion != "gini":
            raise InvalidConfigurationError("split_criterion is pinned to 'gini'")
        if self.max_features != "sqrt":
            raise InvalidConfigurationError("max_features is pinned to 'sqrt'")
        if not self.bootstrap:
            raise InvalidConfigurationError("bootstrap is pinned to True")
        if self.min_samples_leaf != 1 or self.min_samples_split != 2:
            raise InvalidConfigurationError("minimum split/leaf sizes are pinned")
        if self.min_impurity_decrease != 0.0:
            raise InvalidConfigurationError("min_impurity_decrease is pinned to 0.0")
        if self.max_leaves != -1 or self.max_samples != 1.0:
            raise InvalidConfigurationError("leaf/sample limits are pinned")
        if self.max_batch_size != 4096:
            raise InvalidConfigurationError("max_batch_size is pinned to 4096")
        if self.oob_score or self.verbose:
            raise InvalidConfigurationError("oob_score and verbose are pinned to False")
        if self.output_type != "cupy":
            raise InvalidConfigurationError("output_type is pinned to 'cupy'")


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    alpha: float = 0.05
    n_iterations: int = 100
    n_trees: int = 50
    use_variance: bool = True
    min_accuracy: float = 0.5
    min_connections: float = 1.0
    max_repeat_errors: int = 20
    sample_max: int | None = None
    downsampling_rate: float = 1.0
    min_reads: float | None = None
    rules: RuleSet | str = RuleSet.CORRECTED
    input_policy: InputPolicy | str = InputPolicy.TRANSFER_ONCE
    device: str = "cuda:0"
    vram_headroom: float = 0.20
    random_state: int = 1
    rf_profile: RFProfile = field(default_factory=RFProfile)

    def __post_init__(self) -> None:
        try:
            rules = RuleSet(self.rules)
            input_policy = InputPolicy(self.input_policy)
        except ValueError as error:
            raise InvalidConfigurationError(str(error)) from error
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "input_policy", input_policy)

        _positive_int("n_iterations", self.n_iterations)
        _positive_int("n_trees", self.n_trees)
        if rules is RuleSet.CORRECTED and (
            self.n_iterations < 4 or self.n_iterations % 2 != 0
        ):
            raise InvalidConfigurationError(
                "corrected rules require even n_iterations >= 4"
            )
        if not isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise InvalidConfigurationError("alpha must be finite and between 0 and 1")
        if not isfinite(self.min_accuracy) or not 0 <= self.min_accuracy <= 1:
            raise InvalidConfigurationError("min_accuracy must be between 0 and 1")
        if not isfinite(self.min_connections) or self.min_connections < 0:
            raise InvalidConfigurationError("min_connections must be finite and >= 0")
        if (
            isinstance(self.max_repeat_errors, bool)
            or not isinstance(self.max_repeat_errors, int)
            or self.max_repeat_errors < 0
        ):
            raise InvalidConfigurationError("max_repeat_errors must be an integer >= 0")
        if self.sample_max is not None:
            _positive_int("sample_max", self.sample_max)
        if not isfinite(self.downsampling_rate) or self.downsampling_rate <= 0:
            raise InvalidConfigurationError("downsampling_rate must be finite and > 0")
        if self.min_reads is not None and (
            not isfinite(self.min_reads) or self.min_reads <= 0
        ):
            raise InvalidConfigurationError("min_reads must be finite and > 0")
        if not isfinite(self.vram_headroom) or not 0 <= self.vram_headroom < 1:
            raise InvalidConfigurationError("vram_headroom must be in [0, 1)")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise InvalidConfigurationError("random_state must be an integer")
        if not self.device.startswith("cuda:") or not self.device[5:].isdigit():
            raise InvalidConfigurationError("device must have the form 'cuda:<index>'")

    @property
    def split_rounds(self) -> int:
        return split_round_count(self.n_iterations)

    @property
    def forest_fits(self) -> int:
        return 4 * self.split_rounds


@dataclass(frozen=True, slots=True)
class RNAConfig:
    """Explicit optional RNA preprocessing; no preprocessing is implicit."""

    counts_layer: str
    normalized_layer: str = "log1p"
    target_sum: float = 10_000.0
    hvg_flavor: str = "seurat_v3"
    n_top_genes: int = 2_000
    n_pcs: int = 50
    n_neighbors: int = 15

    def __post_init__(self) -> None:
        if not self.counts_layer:
            raise InvalidConfigurationError("counts_layer must be explicit")
        if not self.normalized_layer:
            raise InvalidConfigurationError("normalized_layer must be explicit")
        if self.target_sum <= 0:
            raise InvalidConfigurationError("target_sum must be > 0")
        if self.hvg_flavor not in {"cell_ranger", "seurat_v3", "pearson_residuals"}:
            raise InvalidConfigurationError("unsupported hvg_flavor")
        for name in ("n_top_genes", "n_pcs", "n_neighbors"):
            _positive_int(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ChoirConfig:
    key_added: str = "choir"
    layer: str = "log1p"
    use_rep: str = "X_pca"
    nn_graph_key: str = "choir_nn"
    snn_graph_key: str = "choir_snn"
    feature_mask_key: str = "highly_variable"
    preprocessing: RNAConfig | None = None
    rules: RuleSet | str = RuleSet.CORRECTED
    multiplicity: Multiplicity | str = Multiplicity.BONFERRONI
    input_policy: InputPolicy | str = InputPolicy.TRANSFER_ONCE
    records_path: str | None = None
    inplace: bool = True
    device: str = "cuda:0"
    vram_headroom: float = 0.20
    random_state: int = 1
    comparison: ComparisonConfig | None = None

    def __post_init__(self) -> None:
        if not self.key_added:
            raise InvalidConfigurationError("key_added must not be empty")
        for name in (
            "layer",
            "use_rep",
            "nn_graph_key",
            "snn_graph_key",
            "feature_mask_key",
        ):
            if not getattr(self, name):
                raise InvalidConfigurationError(f"{name} must be explicit")
        if self.nn_graph_key == self.snn_graph_key:
            raise InvalidConfigurationError("NN and SNN graph keys must be distinct")
        try:
            rules = RuleSet(self.rules)
            multiplicity = Multiplicity(self.multiplicity)
            input_policy = InputPolicy(self.input_policy)
        except ValueError as error:
            raise InvalidConfigurationError(str(error)) from error
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(self, "input_policy", input_policy)
        if self.comparison is not None and self.comparison.rules is not rules:
            raise InvalidConfigurationError("choir and comparison rule sets must agree")
