#!/usr/bin/env python3
"""Evaluate the frozen paired L40S performance protocol and optional claim."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Any

from validation.scripts.gate2_common import file_hash, immutable_json


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"performance record must be an object: {path}")
    return value


def _positive_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"performance {field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"performance {field} must be finite and positive")
    return result


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"performance {field} must be a lowercase SHA-256")
    return value


def _preregistered_schedules(
    workloads: list[str], seed: int
) -> dict[str, list[str]]:
    generator = random.Random(seed)
    schedules: dict[str, list[str]] = {}
    for workload_id in workloads:
        schedule = ["r_then_python"] * 4 + ["python_then_r"] * 3
        generator.shuffle(schedule)
        schedules[workload_id] = schedule
    return schedules


def evaluate(contract_path: Path, workload_paths: list[Path]) -> dict[str, Any]:
    contract = _read(contract_path)
    expected = contract.get("workloads")
    interval = contract.get("interval", {})
    pair_order = contract.get("pair_order", {})
    schedules = pair_order.get("schedule", {})
    if (
        contract.get("schema_version") != 1
        or not isinstance(expected, list)
        or len(expected) != 6
        or contract.get("genes_per_workload") != 10000
        or contract.get("measured_pairs_per_workload") != 7
        or interval.get("method") != "one_sample_t"
        or interval.get("degrees_of_freedom") != 6
        or pair_order.get("method") != "preregistered_seeded_permutation"
        or pair_order.get("seed") != 104729
        or not isinstance(schedules, dict)
        or set(schedules) != set(expected)
        or schedules != _preregistered_schedules(expected, 104729)
        or contract.get("memory", {}).get("report_by_implementation") is not True
        or contract.get("correctness", {}).get(
            "per_pair_content_bound_report"
        )
        is not True
    ):
        raise ValueError("L40S performance contract has drifted")
    for workload_id, schedule in schedules.items():
        if (
            not isinstance(schedule, list)
            or len(schedule) != 7
            or schedule.count("r_then_python") != 4
            or schedule.count("python_then_r") != 3
        ):
            raise ValueError(
                f"invalid preregistered performance schedule: {workload_id}"
            )
    critical = float(interval["critical_value"])
    threshold = float(
        contract["claim_policy"]["every_workload_lower_bound_minimum"]
    )
    observed: dict[str, dict[str, Any]] = {}
    candidate: dict[str, Any] | None = None
    for path in workload_paths:
        record = _read(path)
        workload_id = record.get("workload_id")
        if workload_id not in expected or workload_id in observed:
            raise ValueError(f"unexpected or duplicate workload: {workload_id}")
        if (
            record.get("schema_version") != 1
            or record.get("hardware") != "NVIDIA L40S"
            or record.get("genes") != 10000
            or set(record.get("warmup", {})) != {"r", "python"}
            or record.get("timing_boundary")
            != "input_open_to_durable_output_fsync"
        ):
            raise ValueError(f"invalid performance workload: {workload_id}")
        warmup_processes: list[str] = []
        for implementation in ("r", "python"):
            warmup = record["warmup"][implementation]
            if (
                not isinstance(warmup, dict)
                or warmup.get("fresh_process") is not True
                or warmup.get("excluded_from_measurement") is not True
            ):
                raise ValueError(
                    f"invalid performance warm-up: {workload_id}/{implementation}"
                )
            _positive_number(
                warmup.get("wall_seconds"),
                f"{implementation} warm-up time",
            )
            process_id = str(warmup.get("process_id"))
            if process_id in {"", "None"}:
                raise ValueError(
                    f"invalid performance warm-up process: {workload_id}"
                )
            warmup_processes.append(process_id)
        if len(set(warmup_processes)) != 2:
            raise ValueError(f"performance warm-up process reuse: {workload_id}")
        record_candidate = record.get("candidate")
        if not isinstance(record_candidate, dict):
            raise ValueError(f"performance candidate is missing: {workload_id}")
        if candidate is None:
            candidate = record_candidate
        elif record_candidate != candidate:
            raise ValueError("performance candidate drift")
        pairs = record.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 7:
            raise ValueError(f"workload must contain seven pairs: {workload_id}")
        orders: list[str] = []
        process_ids: list[str] = []
        input_hashes: set[str] = set()
        settings_hashes: set[str] = set()
        log_speedups: list[float] = []
        correctness: list[bool] = []
        correctness_evidence: list[dict[str, Any]] = []
        memory_peaks: dict[str, list[int]] = {
            "r_host_peak_bytes": [],
            "r_device_peak_bytes": [],
            "python_host_peak_bytes": [],
            "python_device_peak_bytes": [],
        }
        for pair_index, pair in enumerate(pairs, start=1):
            if not isinstance(pair, dict):
                raise ValueError(f"malformed performance pair: {workload_id}")
            if pair.get("pair_index") != pair_index:
                raise ValueError(f"performance pair order is invalid: {workload_id}")
            orders.append(str(pair.get("order")))
            process_ids.extend(
                [str(pair.get("r_process_id")), str(pair.get("python_process_id"))]
            )
            input_hashes.add(_sha256(pair.get("input_sha256"), "input hash"))
            settings_hashes.add(
                _sha256(pair.get("settings_sha256"), "settings hash")
            )
            r_seconds = _positive_number(pair.get("r_wall_seconds"), "R time")
            python_seconds = _positive_number(
                pair.get("python_wall_seconds"), "Python time"
            )
            log_speedups.append(math.log(r_seconds / python_seconds))
            correctness_reference = pair.get("correctness_report")
            if (
                not isinstance(correctness_reference, dict)
                or set(correctness_reference) != {"path", "sha256"}
            ):
                raise ValueError(
                    f"performance correctness report is absent: {workload_id}"
                )
            correctness_path = Path(str(correctness_reference["path"]))
            correctness_hash = _sha256(
                correctness_reference["sha256"], "correctness report hash"
            )
            if (
                not correctness_path.is_file()
                or file_hash(correctness_path) != correctness_hash
            ):
                raise ValueError(
                    f"performance correctness report is stale: {workload_id}"
                )
            correctness_report = _read(correctness_path)
            checks = correctness_report.get("checks")
            pair_correct = (
                correctness_report.get("schema_version") == 1
                and correctness_report.get("status") == "PASS"
                and correctness_report.get("workload_id") == workload_id
                and correctness_report.get("pair_index") == pair_index
                and correctness_report.get("candidate") == record_candidate
                and isinstance(checks, dict)
                and bool(checks)
                and all(value is True for value in checks.values())
            )
            _sha256(correctness_report.get("r_output_sha256"), "R output hash")
            _sha256(
                correctness_report.get("python_output_sha256"),
                "Python output hash",
            )
            correctness.append(pair_correct)
            correctness_evidence.append(
                {
                    "pair_index": pair_index,
                    "path": str(correctness_path.resolve()),
                    "sha256": correctness_hash,
                    "passed": pair_correct,
                }
            )
            for field, destination in memory_peaks.items():
                raw = pair.get(field)
                if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                    raise ValueError(f"performance memory is invalid: {field}")
                destination.append(raw)
            if (
                pair.get("fresh_process") is not True
                or pair.get("full_io_included") is not True
                or pair.get("durable_output") is not True
            ):
                raise ValueError(f"performance boundary drift: {workload_id}")
        if (
            orders != schedules[workload_id]
            or len(set(process_ids)) != 14
            or any(value in {"", "None"} for value in process_ids)
            or not set(warmup_processes).isdisjoint(process_ids)
            or len(input_hashes) != 1
            or len(settings_hashes) != 1
        ):
            raise ValueError(f"paired performance design drift: {workload_id}")
        mean_log = statistics.fmean(log_speedups)
        standard_error = statistics.stdev(log_speedups) / math.sqrt(7)
        geometric_mean = math.exp(mean_log)
        lower_bound = math.exp(mean_log - critical * standard_error)
        all_correct = all(correctness)
        observed[str(workload_id)] = {
            "workload_id": workload_id,
            "all_outputs_correct": all_correct,
            "geometric_mean_speedup": geometric_mean,
            "one_sided_95_percent_lower_bound": lower_bound,
            "claim_threshold_pass": all_correct and lower_bound >= threshold,
            "absolute_peak_memory_bytes": {
                "r": {
                    "host": max(memory_peaks["r_host_peak_bytes"]),
                    "device": max(memory_peaks["r_device_peak_bytes"]),
                },
                "python": {
                    "host": max(memory_peaks["python_host_peak_bytes"]),
                    "device": max(memory_peaks["python_device_peak_bytes"]),
                },
            },
            "correctness_evidence": correctness_evidence,
            "path": str(path.resolve()),
            "sha256": file_hash(path),
        }
    if set(observed) != set(expected):
        raise ValueError(
            "performance evidence is incomplete: "
            f"missing={sorted(set(expected) - set(observed))}"
        )
    eligible = all(record["claim_threshold_pass"] for record in observed.values())
    return {
        "schema_version": 1,
        "status": "CLAIM_ELIGIBLE" if eligible else "CLAIM_WITHHELD",
        "candidate": candidate,
        "hardware_scope": "NVIDIA L40S only",
        "workloads": [observed[name] for name in expected],
        "cross_hardware_claim": False,
        "causal_gpu_or_rapids_claim": False,
        "contract": {
            "path": str(contract_path.resolve()),
            "sha256": file_hash(contract_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--workload", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    immutable_json(
        arguments.output,
        evaluate(arguments.contract, arguments.workload),
    )


if __name__ == "__main__":
    main()
