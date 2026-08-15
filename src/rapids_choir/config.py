"""Validated, CUDA-independent runtime configuration."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

from ._enums import InputPolicy, Multiplicity, RuleSet
from .exceptions import InvalidConfigurationError, MissingCalibrationError


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

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> RFProfile:
        """Build a profile while rejecting unknown calibration parameters."""

        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            names = ", ".join(sorted(unknown))
            raise InvalidConfigurationError(f"unknown RF profile fields: {names}")
        return cls(**dict(values))

    def to_record(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frozen_profile_path() -> Path:
    configured = os.environ.get("RAPIDS_CHOIR_RF_PROFILE")
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parent / "_data" / "rf_profile.json"


def _load_frozen_rf_profile(path: Path) -> tuple[RFProfile, str]:
    if not path.is_file():
        raise MissingCalibrationError(
            "no frozen RF profile is installed; pass an explicit RFProfile for "
            "development only, or install a calibrated release"
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "FROZEN":
            raise ValueError("profile status is not FROZEN")
        winner = payload["winner"]
        profile = winner.get("profile", winner)
        return RFProfile.from_mapping(profile), _sha256(path)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
        raise InvalidConfigurationError(
            f"invalid frozen RF profile at {path}: {error}"
        ) from error


def default_rf_profile() -> RFProfile:
    """Load the shipped frozen winner; never fall back to a seed profile."""

    profile, _ = _load_frozen_rf_profile(_frozen_profile_path())
    return profile


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
    rf_profile: RFProfile | None = None
    rf_profile_manifest: str | None = None
    rf_profile_source: str = field(init=False)
    rf_profile_evidence_eligible: bool = field(init=False)
    rf_profile_manifest_sha256: str | None = field(init=False)

    def __post_init__(self) -> None:
        try:
            rules = RuleSet(self.rules)
            input_policy = InputPolicy(self.input_policy)
        except ValueError as error:
            raise InvalidConfigurationError(str(error)) from error
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "input_policy", input_policy)

        profile = self.rf_profile
        manifest_hash: str | None = None
        if profile is None:
            profile_path = _frozen_profile_path()
            profile, manifest_hash = _load_frozen_rf_profile(profile_path)
            source = "frozen"
            object.__setattr__(self, "rf_profile_manifest", str(profile_path))
        elif self.rf_profile_manifest is not None:
            profile_path = Path(self.rf_profile_manifest).expanduser()
            frozen, manifest_hash = _load_frozen_rf_profile(profile_path)
            if profile != frozen:
                raise InvalidConfigurationError(
                    "explicit RF profile does not match its frozen manifest"
                )
            source = "frozen"
        else:
            source = "explicit_development"
        object.__setattr__(self, "rf_profile", profile)
        object.__setattr__(self, "rf_profile_source", source)
        object.__setattr__(self, "rf_profile_evidence_eligible", source == "frozen")
        object.__setattr__(self, "rf_profile_manifest_sha256", manifest_hash)

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

    @property
    def resolved_rf_profile(self) -> RFProfile:
        """Return the post-validation profile with a non-optional type."""

        if self.rf_profile is None:  # pragma: no cover - guarded in __post_init__
            raise MissingCalibrationError("RF profile resolution did not complete")
        return self.rf_profile


@dataclass(frozen=True, slots=True)
class RNAConfig:
    """Explicit optional RNA preprocessing; no preprocessing is implicit."""

    counts_layer: str = "counts"
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
    batch_key: str | None = None
    preprocessing: RNAConfig | None = None
    rules: RuleSet | str = RuleSet.CORRECTED
    multiplicity: Multiplicity | str = Multiplicity.BONFERRONI
    max_clusters: int | None = None
    min_cluster_depth: int = 2_000
    group_singletons: bool = True
    max_resolution_steps: int = 512
    input_policy: InputPolicy | str = InputPolicy.TRANSFER_ONCE
    records_path: str | None = None
    inplace: bool = True
    overwrite: bool = False
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
        if self.batch_key is not None and not self.batch_key:
            raise InvalidConfigurationError("batch_key must be non-empty when supplied")
        if self.preprocessing is not None and not isinstance(
            self.preprocessing, RNAConfig
        ):
            raise InvalidConfigurationError("preprocessing must be an RNAConfig")
        if (
            self.preprocessing is not None
            and self.layer != self.preprocessing.normalized_layer
        ):
            raise InvalidConfigurationError(
                "layer must match preprocessing.normalized_layer"
            )
        if self.comparison is not None and not isinstance(
            self.comparison, ComparisonConfig
        ):
            raise InvalidConfigurationError("comparison must be a ComparisonConfig")
        if self.max_clusters is not None:
            _positive_int("max_clusters", self.max_clusters)
        _positive_int("min_cluster_depth", self.min_cluster_depth)
        _positive_int("max_resolution_steps", self.max_resolution_steps)
        if not isinstance(self.group_singletons, bool):
            raise InvalidConfigurationError("group_singletons must be boolean")
        if not isinstance(self.inplace, bool) or not isinstance(self.overwrite, bool):
            raise InvalidConfigurationError("inplace and overwrite must be boolean")
        try:
            rules = RuleSet(self.rules)
            multiplicity = Multiplicity(
                "bh" if self.multiplicity == "fdr" else self.multiplicity
            )
            input_policy = InputPolicy(self.input_policy)
        except ValueError as error:
            raise InvalidConfigurationError(str(error)) from error
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(self, "input_policy", input_policy)
        if input_policy is not InputPolicy.TRANSFER_ONCE:
            raise InvalidConfigurationError(
                "AnnData CHOIR requires input_policy='transfer_once'"
            )
        if not self.device.startswith("cuda:") or not self.device[5:].isdigit():
            raise InvalidConfigurationError("device must have the form 'cuda:<index>'")
        if not isfinite(self.vram_headroom) or not 0 <= self.vram_headroom < 1:
            raise InvalidConfigurationError("vram_headroom must be in [0, 1)")
        if isinstance(self.random_state, bool) or not isinstance(
            self.random_state, int
        ):
            raise InvalidConfigurationError("random_state must be an integer")
        if self.records_path is not None and not self.records_path:
            raise InvalidConfigurationError("records_path must be non-empty")
        if self.comparison is not None:
            if self.comparison.rules is not rules:
                raise InvalidConfigurationError(
                    "choir and comparison rule sets must agree"
                )
            if self.comparison.device != self.device:
                raise InvalidConfigurationError(
                    "choir and comparison devices must agree"
                )
            if self.comparison.random_state != self.random_state:
                raise InvalidConfigurationError(
                    "choir and comparison random states must agree"
                )
            if self.comparison.vram_headroom != self.vram_headroom:
                raise InvalidConfigurationError(
                    "choir and comparison VRAM headroom must agree"
                )
