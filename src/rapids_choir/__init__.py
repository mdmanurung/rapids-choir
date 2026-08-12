"""CUDA-safe public package namespace."""

from __future__ import annotations

from . import get, tl
from ._enums import Decision, InputPolicy, Multiplicity, ReasonCode, RuleSet
from .config import ChoirConfig, ComparisonConfig, RFProfile, RNAConfig
from .results import ChoirResult, ComparisonResult

__version__ = "0.1.0.dev0"

__all__ = [
    "ChoirConfig",
    "ChoirResult",
    "ComparisonConfig",
    "ComparisonResult",
    "Decision",
    "InputPolicy",
    "Multiplicity",
    "RFProfile",
    "RNAConfig",
    "ReasonCode",
    "RuleSet",
    "get",
    "tl",
]
