from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "derive_comparator_margins",
    SCRIPTS / "derive_comparator_margins.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

SEED_PAIRS = [(100 + 2 * index, 101 + 2 * index) for index in range(10)]


def initialise_repository(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    marker = root / "marker.txt"
    marker.write_text("candidate\n", encoding="utf-8")
    subprocess.run(["git", "add", "marker.txt"], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Margin Test",
            "-c",
            "user.email=margin@example.invalid",
            "commit",
            "-qm",
            "candidate",
        ],
        cwd=root,
        check=True,
    )


def margin_inputs(
    root: Path, *, outlier_difference: float | None = None
) -> tuple[Path, list[tuple[str, int, Path, int, Path]]]:
    contract = {
        "roles": {"development": ["Dataset A"]},
        "reference_variability": {
            "seed_pairs": [list(pair) for pair in SEED_PAIRS],
            "deterministic_fields": ["decision", "reason"],
            "bounded_fields": ["mean_observed_accuracy"],
            "quantile": 0.975,
            "maximum_margin": 0.03,
        },
    }
    contract_path = root / "contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract), encoding="utf-8")
    candidate_record = candidate(root)
    reference_candidate = {
        "commit": candidate_record["candidate_commit"],
        "attempt": candidate_record["candidate_attempt"],
        "runtime_tree_sha256": candidate_record["runtime_tree_sha256"],
        "manifest_sha256": candidate_record["manifest_sha256"],
        "dirty_state": candidate_record["dirty_state"],
    }
    runs = []
    for index, (seed_a, seed_b) in enumerate(SEED_PAIRS):
        difference = (
            outlier_difference
            if index == len(SEED_PAIRS) - 1 and outlier_difference is not None
            else 0.01
        )
        paths = []
        for seed, accuracy in ((seed_a, 0.50), (seed_b, 0.50 + difference)):
            path = root / f"seed-{seed}.json"
            path.write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "dataset_id": "Dataset A",
                        "seed": seed,
                        "candidate": reference_candidate,
                        "h5ad_sha256": "h5ad",
                        "pair_manifest_sha256": "pairs",
                        "reference_source_sha256": "reference",
                        "results": [
                            {
                                "pair_id": "pair-1",
                                "decision": "merge",
                                "reason": "not_distinct",
                                "mean_observed_accuracy": accuracy,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            paths.append(path)
        runs.append(("Dataset A", seed_a, paths[0], seed_b, paths[1]))
    return contract_path, runs


def candidate(root: Path) -> dict[str, object]:
    return {
        "candidate_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "candidate_attempt": 1,
        "runtime_tree_sha256": "runtime",
        "manifest_sha256": "manifest",
        "dirty_state": {"tracked": False},
    }


def test_development_margin_uses_predeclared_975_quantile(tmp_path: Path) -> None:
    initialise_repository(tmp_path)
    contract, runs = margin_inputs(tmp_path)

    payload = MODULE.derive_margins(
        root=tmp_path,
        contract_path=contract,
        runs=runs,
        candidate=candidate(tmp_path),
    )

    assert payload["status"] == "FROZEN"
    assert payload["eligible"] is True
    assert payload["bounded_metric_margins"]["mean_observed_accuracy"][
        "margin"
    ] == pytest.approx(0.01)


def test_reference_variability_above_cap_makes_candidate_ineligible(
    tmp_path: Path,
) -> None:
    initialise_repository(tmp_path)
    contract, runs = margin_inputs(tmp_path, outlier_difference=0.10)

    payload = MODULE.derive_margins(
        root=tmp_path,
        contract_path=contract,
        runs=runs,
        candidate=candidate(tmp_path),
    )

    assert payload["status"] == "INELIGIBLE"
    assert payload["eligible"] is False
