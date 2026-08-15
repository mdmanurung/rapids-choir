#!/usr/bin/env python3
"""Validate exhaustive R-symbol coverage and immutable parity-ledger evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_CLASSIFICATIONS = {
    "matched_compatibility",
    "intentional_corrected_divergence",
    "excluded_v0_1",
    "unavailable_input",
}
DEFINITION = re.compile(r"(?m)^([.`A-Za-z][A-Za-z0-9._]*)\s*<-\s*function\s*\(")


@dataclass(frozen=True, slots=True)
class RFunction:
    symbol: str
    path: str
    start_line: int
    end_line: int
    sha256: str

    def to_record(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "sha256": self.sha256,
        }


def _find_closing_brace(text: str, start: int) -> int:
    opening = text.find("{", start)
    if opening < 0:
        raise ValueError("R function has no opening brace")
    depth = 0
    quote: str | None = None
    escaped = False
    comment = False
    for index in range(opening, len(text)):
        character = text[index]
        if comment:
            if character == "\n":
                comment = False
            continue
        if quote is not None:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == quote:
                quote = None
            continue
        if character == "#":
            comment = True
        elif character in {"'", '"'}:
            quote = character
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
    raise ValueError("R function has no closing brace")


def inventory(reference_root: Path, repo_root: Path) -> dict[str, RFunction]:
    functions: dict[str, RFunction] = {}
    for path in sorted(reference_root.glob("*.R")):
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines(keepends=True)
        for match in DEFINITION.finditer(text):
            symbol = match.group(1)
            closing = _find_closing_brace(text, match.end())
            start_line = text.count("\n", 0, match.start()) + 1
            end_line = text.count("\n", 0, closing) + 1
            source = "".join(lines[start_line - 1 : end_line]).encode("utf-8")
            relative = str(path.relative_to(repo_root))
            if symbol in functions:
                raise ValueError(f"duplicate R function definition: {symbol}")
            functions[symbol] = RFunction(
                symbol=symbol,
                path=relative,
                start_line=start_line,
                end_line=end_line,
                sha256=hashlib.sha256(source).hexdigest(),
            )
    return functions


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ledger(
    ledger: dict[str, Any],
    *,
    functions: dict[str, RFunction],
    repo_root: Path,
) -> list[str]:
    errors: list[str] = []
    if ledger.get("schema_version") != 1:
        errors.append("unsupported parity ledger schema")
    if ledger.get("reference_version") != "CHOIR v0.3.0":
        errors.append("reference_version must be CHOIR v0.3.0")
    entries = ledger.get("entries")
    if not isinstance(entries, list):
        return [*errors, "entries must be a list"]
    by_symbol: dict[str, list[dict[str, Any]]] = {}
    entry_ids: set[str] = set()
    required = {
        "entry_id",
        "r_symbol",
        "r_source",
        "semantic_responsibility",
        "python_components",
        "classification",
        "expected_state_transition",
        "test_cases",
        "comparison_rule",
        "evidence",
        "meaningful_branches",
    }
    for index, raw_entry in enumerate(entries):
        prefix = f"entries[{index}]"
        if not isinstance(raw_entry, dict):
            errors.append(f"{prefix} must be an object")
            continue
        entry = raw_entry
        missing = required - set(entry)
        unknown = set(entry) - required
        if missing:
            errors.append(f"{prefix} missing fields: {sorted(missing)}")
        if unknown:
            errors.append(f"{prefix} has unknown fields: {sorted(unknown)}")
        entry_id = entry.get("entry_id")
        if not isinstance(entry_id, str) or not entry_id:
            errors.append(f"{prefix}.entry_id must be non-empty")
        elif entry_id in entry_ids:
            errors.append(f"duplicate entry_id: {entry_id}")
        else:
            entry_ids.add(entry_id)
        symbol = entry.get("r_symbol")
        if not isinstance(symbol, str) or symbol not in functions:
            errors.append(f"{prefix}.r_symbol is unknown: {symbol!r}")
            continue
        by_symbol.setdefault(symbol, []).append(entry)
        expected_source = functions[symbol].to_record()
        if entry.get("r_source") != expected_source:
            errors.append(f"{prefix}.r_source span or digest drift")
        classification = entry.get("classification")
        if classification not in ALLOWED_CLASSIFICATIONS:
            errors.append(f"{prefix}.classification is unclassified")
        for field in (
            "semantic_responsibility",
            "expected_state_transition",
            "comparison_rule",
        ):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                errors.append(f"{prefix}.{field} must be explained")
        components = entry.get("python_components")
        if not isinstance(components, list) or not components:
            errors.append(f"{prefix}.python_components must be non-empty")
        else:
            for component in components:
                component_path = str(component).split("::", 1)[0]
                if not (repo_root / component_path).is_file():
                    errors.append(
                        f"{prefix}.python_components path missing: {component_path}"
                    )
        tests = entry.get("test_cases")
        if not isinstance(tests, list) or not tests:
            errors.append(f"{prefix}.test_cases must be non-empty")
        else:
            for test in tests:
                test_path = str(test).split("::", 1)[0]
                if not (repo_root / test_path).is_file():
                    errors.append(f"{prefix}.test_cases path missing: {test_path}")
        evidence = entry.get("evidence")
        if not isinstance(evidence, dict) or set(evidence) != {
            "path",
            "sha256",
            "status",
        }:
            errors.append(f"{prefix}.evidence must contain path, sha256, status")
        else:
            evidence_path = repo_root / str(evidence["path"])
            if not evidence_path.is_file():
                errors.append(f"{prefix}.evidence path is missing")
            elif _file_sha256(evidence_path) != evidence["sha256"]:
                errors.append(f"{prefix}.evidence checksum drift")
            if evidence["status"] not in {"verified", "pending", "excluded"}:
                errors.append(f"{prefix}.evidence status is invalid")
        branches = entry.get("meaningful_branches")
        if not isinstance(branches, list) or not branches:
            errors.append(f"{prefix}.meaningful_branches must be non-empty")
        else:
            branch_ids: set[str] = set()
            for branch in branches:
                if not isinstance(branch, dict) or set(branch) != {
                    "branch_id",
                    "classification",
                    "explanation",
                }:
                    errors.append(f"{prefix} contains an invalid branch record")
                    continue
                branch_id = branch["branch_id"]
                if not isinstance(branch_id, str) or not branch_id:
                    errors.append(f"{prefix} branch ID must be non-empty")
                elif branch_id in branch_ids:
                    errors.append(f"{prefix} has duplicate branch {branch_id}")
                branch_ids.add(branch_id)
                if branch["classification"] not in ALLOWED_CLASSIFICATIONS:
                    errors.append(f"{prefix} branch {branch_id} is unclassified")
                if not isinstance(branch["explanation"], str) or not branch[
                    "explanation"
                ].strip():
                    errors.append(f"{prefix} branch {branch_id} is unexplained")
    missing_symbols = sorted(set(functions) - set(by_symbol))
    extra_symbols = sorted(set(by_symbol) - set(functions))
    if missing_symbols:
        errors.append(f"unmapped R functions: {missing_symbols}")
    if extra_symbols:
        errors.append(f"unknown mapped R functions: {extra_symbols}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ledger",
        type=Path,
        default=Path("validation/contracts/parity_ledger_v1.json"),
    )
    parser.add_argument("--inventory", action="store_true")
    arguments = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    reference_root = repo_root / "validation/reference/choir_v0_3_0/R"
    functions = inventory(reference_root, repo_root)
    if arguments.inventory:
        print(
            json.dumps(
                [functions[name].to_record() for name in sorted(functions)],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    ledger_path = (
        arguments.ledger
        if arguments.ledger.is_absolute()
        else repo_root / arguments.ledger
    )
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    errors = validate_ledger(ledger, functions=functions, repo_root=repo_root)
    report = {
        "schema_version": 1,
        "passed": not errors,
        "r_function_count": len(functions),
        "ledger_entry_count": len(ledger.get("entries", [])),
        "unclassified_count": sum("unclassified" in error for error in errors),
        "errors": errors,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
