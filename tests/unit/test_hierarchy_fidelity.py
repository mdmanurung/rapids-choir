from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from rapids_choir._semantic_trace import TraceRecorder, semantic_trace_reference
from rapids_choir.tree import hierarchy_from_partitions
from validation.scripts.aggregate_hierarchy_fidelity import aggregate
from validation.scripts.calibrate_hierarchy_pair import calibrate
from validation.scripts.derive_hierarchy_margins import derive
from validation.scripts.evaluate_hierarchy_fidelity import (
    _event_matches,
    _load_json,
    _mandatory_trace_difference,
    _pair_metrics,
    _trace,
    _validate_confirmation_opening,
    _validate_result,
)
from validation.scripts.verify_frozen_hierarchy_margins import (
    verify as verify_frozen_margins,
)

ROOT = Path(__file__).resolve().parents[2]


def test_hierarchy_endpoint_metrics_are_zero_for_label_permutation() -> None:
    metrics = _pair_metrics(["a", "a", "b", "b"], [2, 2, 1, 1])
    assert metrics == {
        "one_minus_ari": 0.0,
        "pairwise_false_split_rate": 0.0,
        "pairwise_false_merge_rate": 0.0,
    }


def test_hierarchy_result_schema_binds_lossless_tree_and_trace(
    tmp_path: Path,
) -> None:
    run_id = "run-" + "a" * 32
    recorder = TraceRecorder(run_id)
    recorder.append("provenance", "initialization", decision="validated")
    trace_path = tmp_path / "trace.jsonl"
    recorder.write_jsonl(trace_path)
    tree = hierarchy_from_partitions([[0, 0]], cell_ids=["c1", "c2"])
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "schema_version": "rapids_choir.hierarchy-result/v1",
                "candidate": {
                    "commit": "b" * 40,
                    "attempt": 1,
                    "runtime_tree_sha256": "c" * 64,
                    "manifest_sha256": "d" * 64,
                },
                "dataset_id": "fixture",
                "run_id": run_id,
                "cell_ids": ["c1", "c2"],
                "labels": [1, 1],
                "tree": tree.to_record(),
                "provenance": {
                    "schema": "rapids_choir.result/v1",
                    "run_id": run_id,
                    "input_sha256": "e" * 64,
                    "input_contract": {},
                    "configuration": {},
                    "rules": "choir_v0_3_0",
                    "multiplicity": "bonferroni",
                    "random_state": 1,
                },
                "semantic_trace": semantic_trace_reference(
                    trace_path,
                    format="jsonl",
                    relative_to=tmp_path,
                ),
            }
        ),
        encoding="utf-8",
    )
    value = _load_json(result_path)
    assert _validate_result(result_path, value) == tree
    assert _trace(result_path, value) == recorder.events


def test_hierarchy_endpoint_metrics_separate_false_splits_and_merges() -> None:
    split = _pair_metrics([0, 0, 1, 1], [0, 1, 2, 2])
    assert split["pairwise_false_split_rate"] == pytest.approx(0.5)
    assert split["pairwise_false_merge_rate"] == 0.0

    merge = _pair_metrics([0, 0, 1, 1], [0, 0, 0, 0])
    assert merge["pairwise_false_split_rate"] == 0.0
    assert merge["pairwise_false_merge_rate"] == pytest.approx(1.0)


def test_missing_bounded_metric_is_a_mandatory_trace_failure() -> None:
    recorder = TraceRecorder("run-" + "a" * 32)
    recorder.append("comparator_call", "pruning")
    assert _mandatory_trace_difference(
        "events[0].bounded_values", "exact_keys", recorder.events
    )


def test_role_event_requirement_uses_nested_subset_matching() -> None:
    recorder = TraceRecorder("run-" + "a" * 32)
    event = recorder.append(
        "comparator_call",
        "pruning",
        parameters={"pair_kind": "subtree_neighbor", "extra": 1},
        decision="split",
    )
    assert _event_matches(
        event,
        {
            "event_type": "comparator_call",
            "parameters": {"pair_kind": "subtree_neighbor"},
        },
    )
    assert not _event_matches(event, {"decision": "merge"})


def test_confirmation_evaluation_binds_open_once_sealed_artifacts(
    tmp_path: Path,
) -> None:
    candidate = {
        "commit": "a" * 40,
        "attempt": 1,
        "runtime_tree_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    authorization = tmp_path / "authorization.json"
    authorization.write_text("{}\n", encoding="utf-8")
    authorization_sha256 = hashlib.sha256(
        authorization.read_bytes()
    ).hexdigest()
    attempt = tmp_path / "opening.ATTEMPTED.json"
    attempt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "OPENING_STARTED",
                "scope": "one_time_hierarchy_confirmation_opening",
                "dataset_id": "Arigoni",
                "candidate": candidate,
                "authorization_sha256": authorization_sha256,
                "source_roles": [
                    "custody_record",
                    "h5ad",
                    "python_config",
                    "recipe",
                ],
            }
        ),
        encoding="utf-8",
    )
    artifacts = {}
    for name in ("h5ad", "recipe", "python_config", "custody_record"):
        sealed = tmp_path / f"sealed.{name}"
        sealed.write_bytes(name.encode("utf-8"))
        artifacts[name] = {
            "source_path": f"/restricted/{name}",
            "sealed_path": str(sealed),
            "bytes": sealed.stat().st_size,
            "sha256": hashlib.sha256(sealed.read_bytes()).hexdigest(),
        }
    opening = tmp_path / "opening.json"
    opening.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "SEALED",
                "scope": "one_time_hierarchy_confirmation_opening",
                "dataset_id": "Arigoni",
                "role": "computational_parity_confirmation_only",
                "opening_count": 1,
                "independent_hierarchy_validation": False,
                "candidate": candidate,
                "authorization": {
                    "path": str(authorization),
                    "sha256": authorization_sha256,
                    "authorization_id": "fixture",
                },
                "opening_attempt": {
                    "path": str(attempt),
                    "sha256": hashlib.sha256(attempt.read_bytes()).hexdigest(),
                },
                "artifacts": artifacts,
                "custody_deviation": "none",
            }
        ),
        encoding="utf-8",
    )
    result = _validate_confirmation_opening(
        opening,
        dataset_id="Arigoni",
        candidate=candidate,
    )
    assert result["opening_count"] == 1
    Path(artifacts["h5ad"]["sealed_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact is stale"):
        _validate_confirmation_opening(
            opening,
            dataset_id="Arigoni",
            candidate=candidate,
        )


def test_r_vs_r_calibration_normalizes_only_planned_seed_values(
    tmp_path: Path,
) -> None:
    candidate = {
        "commit": "b" * 40,
        "attempt": 1,
        "runtime_tree_sha256": "c" * 64,
        "manifest_sha256": "d" * 64,
    }
    tree = hierarchy_from_partitions([[0, 0]], cell_ids=["c1", "c2"])

    def write_result(seed: int, suffix: str) -> Path:
        run_id = f"run-{seed:032x}"
        recorder = TraceRecorder(run_id)
        recorder.append(
            "comparator_call",
            "pruning",
            parameters={"pair_kind": "sibling", "random_state": seed},
            seed=seed,
            decision="merge",
            reason="merge_accuracy",
            bounded_values={"mean_observed_accuracy": 0.5},
            payload={
                "round_seeds": [seed + 1, seed + 2],
                "fit_seeds": [seed + 3, seed + 4],
            },
        )
        trace_path = tmp_path / f"{suffix}.jsonl"
        recorder.write_jsonl(trace_path)
        result_path = tmp_path / f"{suffix}.json"
        result_path.write_text(
            json.dumps(
                {
                    "schema_version": "rapids_choir.hierarchy-result/v1",
                    "candidate": candidate,
                    "dataset_id": "calibration-fixture",
                    "run_id": run_id,
                    "cell_ids": ["c1", "c2"],
                    "labels": [1, 1],
                    "tree": tree.to_record(),
                    "provenance": {
                        "schema": "rapids_choir.result/v1",
                        "run_id": run_id,
                        "input_sha256": "e" * 64,
                        "input_contract": {},
                        "configuration": {},
                        "rules": "choir_v0_3_0",
                        "multiplicity": "bonferroni",
                        "random_state": seed,
                    },
                    "semantic_trace": semantic_trace_reference(
                        trace_path,
                        format="jsonl",
                        relative_to=tmp_path,
                    ),
                }
            ),
            encoding="utf-8",
        )
        return result_path

    report = calibrate(
        write_result(104729, "first"),
        write_result(104759, "second"),
        role="seed-normalization",
        repeat_pair="seed104729-vs-seed104759",
    )
    assert report["exact_fields_pass"] is True
    assert report["endpoint_differences"]["record_discordance"] == 0.0


def test_hierarchy_margin_derivation_requires_all_role_exclusive_objects(
    tmp_path: Path,
) -> None:
    contract = json.loads(
        (ROOT / "validation/contracts/hierarchy_validation_v1.json").read_text(
            encoding="utf-8"
        )
    )
    paths: list[Path] = []
    for index, record in enumerate(contract["development"]["objects"]):
        path = tmp_path / f"calibration-{index:02d}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": "rapids_choir.hierarchy-calibration/v1",
                    "dataset_id": record["dataset_id"],
                    "role": record["role"],
                    "repeat_pair": "seed104729-vs-seed104759",
                    "candidate": {
                        "commit": "a" * 40,
                        "attempt": 1,
                        "runtime_tree_sha256": "b" * 64,
                        "manifest_sha256": "c" * 64,
                    },
                    "exact_fields_pass": True,
                    "endpoint_differences": {
                        "one_minus_ari": 0.01,
                        "pairwise_false_split_rate": 0.01,
                        "pairwise_false_merge_rate": 0.01,
                        "record_discordance": 0.01,
                    },
                    "bounded_differences": {
                        "mean_observed_accuracy": 0.01
                    },
                }
            ),
            encoding="utf-8",
        )
        paths.append(path)
    result = derive(
        ROOT / "validation/contracts/hierarchy_validation_v1.json", paths
    )
    assert result["status"] == "FROZEN"
    assert result["eligible"] is True
    assert result["development_object_count"] == 21


def test_final_candidate_margins_must_exactly_reproduce_frozen_payload(
    tmp_path: Path,
) -> None:
    record = {
        "schema_version": 1,
        "contract_id": "hierarchy-fidelity-margins-v1",
        "status": "FROZEN",
        "eligible": True,
        "maximum_margin": 0.03,
        "quantile": 0.975,
        "quantile_method": "linear",
        "deterministic_float": {"rtol": 1e-7, "atol": 1e-10},
        "deterministic_bounded_fields": ["adjusted_alpha"],
        "stochastic_margins": {
            "accuracy": {"margin": 0.01, "eligible": True}
        },
        "hierarchy_endpoint_margins": {
            name: {"margin": 0.01, "eligible": True}
            for name in (
                "one_minus_ari",
                "pairwise_false_split_rate",
                "pairwise_false_merge_rate",
                "record_discordance",
            )
        },
        "exact_fields_pass": True,
        "development_object_count": 21,
        "calibration_record_count": 21,
    }
    frozen = tmp_path / "frozen.json"
    reproduced = tmp_path / "reproduced.json"
    frozen.write_text(json.dumps(record), encoding="utf-8")
    reproduced.write_text(
        json.dumps(
            record
            | {
                "candidate": {
                    "commit": "a" * 40,
                    "attempt": 1,
                    "runtime_tree_sha256": "b" * 64,
                    "manifest_sha256": "c" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    assert verify_frozen_margins(frozen, reproduced)["status"] == "REPRODUCED"
    drifted = json.loads(reproduced.read_text(encoding="utf-8"))
    drifted["hierarchy_endpoint_margins"]["one_minus_ari"]["margin"] = 0.02
    reproduced.write_text(json.dumps(drifted), encoding="utf-8")
    assert verify_frozen_margins(frozen, reproduced)["status"] == "DRIFTED"


def test_hierarchy_aggregate_requires_independent_complete_evidence(
    tmp_path: Path,
) -> None:
    design = ROOT / "validation/contracts/hierarchy_validation_v1.json"
    development_role_contract = (
        ROOT / "validation/contracts/hierarchy_development_objects_v1.json"
    )
    confirmation_role_contract = (
        ROOT / "validation/contracts/data_roles_v2.json"
    )
    contract = json.loads(design.read_text(encoding="utf-8"))
    margins = tmp_path / "margins.json"
    candidate = {
        "commit": "a" * 40,
        "attempt": 1,
        "runtime_tree_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    margins.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "FROZEN",
                "eligible": True,
                "maximum_margin": 0.03,
                "candidate": candidate,
                "hierarchy_endpoint_margins": {
                    name: {"margin": 0.01, "eligible": True}
                    for name in (
                        "one_minus_ari",
                        "pairwise_false_split_rate",
                        "pairwise_false_merge_rate",
                        "record_discordance",
                    )
                },
            }
        ),
        encoding="utf-8",
    )
    endpoint_values = {
        "one_minus_ari": 0.0,
        "pairwise_false_split_rate": 0.0,
        "pairwise_false_merge_rate": 0.0,
        "record_discordance": 0.0,
    }
    endpoint_margins = {name: 0.01 for name in endpoint_values}
    margin_sha256 = hashlib.sha256(margins.read_bytes()).hexdigest()

    def report(dataset_id: str, role: str, role_contract: Path) -> Path:
        reference_result = tmp_path / f"{dataset_id}.reference.json"
        candidate_result = tmp_path / f"{dataset_id}.candidate.json"
        reference_result.write_text("{}\n", encoding="utf-8")
        candidate_result.write_text("{}\n", encoding="utf-8")
        opening = None
        opening_evidence = None
        if role == "hierarchy_parity_confirmation":
            opening_path = tmp_path / f"{dataset_id}.opening.json"
            opening_path.write_text("{}\n", encoding="utf-8")
            opening = {
                "opening_count": 1,
                "attempt_sha256": "e" * 64,
                "authorization_sha256": "f" * 64,
            }
            opening_evidence = {
                "path": str(opening_path),
                "sha256": hashlib.sha256(
                    opening_path.read_bytes()
                ).hexdigest(),
            }
        path = tmp_path / f"{dataset_id}.json"
        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "PASS",
                    "dataset_id": dataset_id,
                    "role": role,
                    "opening": opening,
                    "run_id": "run-" + "a" * 32,
                    "candidate": candidate,
                    "checks": {
                        "candidate_context_exact": True,
                        "schemas_and_topology_invariants": True,
                        "provenance_contract_exact": True,
                        "semantic_trace_exact_requirements": True,
                        "endpoint_metrics_within_development_margins": True,
                        "role_contract_bound": True,
                        "role_requirements_satisfied": True,
                        "confirmation_opening_contract": True,
                    },
                    "metric_checks": {
                        name: True for name in endpoint_values
                    },
                    "metrics": endpoint_values,
                    "margins": endpoint_margins,
                    "evidence": {
                        "reference_result": {
                            "path": str(reference_result),
                            "sha256": hashlib.sha256(
                                reference_result.read_bytes()
                            ).hexdigest(),
                        },
                        "candidate_result": {
                            "path": str(candidate_result),
                            "sha256": hashlib.sha256(
                                candidate_result.read_bytes()
                            ).hexdigest(),
                        },
                        "margin_contract": {
                            "path": str(margins),
                            "sha256": margin_sha256,
                        },
                        "role_contract": {
                            "path": str(role_contract),
                            "sha256": hashlib.sha256(
                                role_contract.read_bytes()
                            ).hexdigest(),
                        },
                        **(
                            {"opening": opening_evidence}
                            if opening_evidence is not None
                            else {}
                        ),
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    development = [
        report(
            record["dataset_id"],
            record["role"],
            development_role_contract,
        )
        for record in contract["development"]["objects"]
    ]
    confirmation = [
        report(
            dataset_id,
            "hierarchy_parity_confirmation",
            confirmation_role_contract,
        )
        for dataset_id in ("Arigoni", "Chen")
    ]
    result = aggregate(
        design,
        margins,
        development_role_contract,
        confirmation_role_contract,
        development,
        confirmation,
    )
    assert result["status"] == "PASS"
    assert len(result["development"]) == 21
    assert [record["dataset_id"] for record in result["confirmation"]] == [
        "Arigoni",
        "Chen",
    ]
    assert result["acceptance_policy"]["pooled_acceptance"] is False

    with pytest.raises(ValueError, match="incomplete development"):
        aggregate(
            design,
            margins,
            development_role_contract,
            confirmation_role_contract,
            development[:-1],
            confirmation,
        )

    tampered = json.loads(development[0].read_text(encoding="utf-8"))
    tampered["evidence"]["margin_contract"]["sha256"] = "0" * 64
    tampered_path = tmp_path / "tampered-report.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match=r"stale|margin drift"):
        aggregate(
            design,
            margins,
            development_role_contract,
            confirmation_role_contract,
            [tampered_path, *development[1:]],
            confirmation,
        )
