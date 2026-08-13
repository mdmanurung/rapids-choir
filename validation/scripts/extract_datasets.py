#!/usr/bin/env python3
"""Extract only an explicitly frozen development or sealed holdout set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any

from check_profile_seal import verify_profile_seal
from gate2_common import atomic_json, dataset_number, dataset_slug

ARCHIVE_FILES = {
    "counts": ("RNA_matrix_raw.rds", ".rds"),
    "metadata": ("cell_metadata.csv", ".metadata.csv"),
}


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def member_matches(name: str, dataset_id: str, kind: str) -> bool:
    if kind not in ARCHIVE_FILES:
        raise ValueError(f"unknown dataset file kind: {kind}")
    number = dataset_number(dataset_id)
    targets = {
        f"simulateddataset{number}",
        f"dataset{number}",
        f"sim{number}",
    }
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return False
    components = [normalized(part) for part in path.parent.parts]
    expected_name, _ = ARCHIVE_FILES[kind]
    return path.name.casefold() == expected_name.casefold() and bool(
        targets & set(components)
    )


def select_members(
    inventory: dict[str, Any], dataset_ids: list[str], output_root: Path
) -> dict[str, dict[str, Any]]:
    """Resolve exactly one counts and metadata member for every dataset."""

    files = [
        member
        for member in inventory["members"]
        if member.get("type") == "file"
    ]
    selected: dict[str, dict[str, Any]] = {}
    for dataset_id in dataset_ids:
        for kind, (_, output_suffix) in ARCHIVE_FILES.items():
            matches = [
                member
                for member in files
                if member_matches(str(member["name"]), dataset_id, kind)
            ]
            if len(matches) != 1:
                names = [str(member["name"]) for member in matches]
                raise RuntimeError(
                    f"expected exactly one {kind} archive member for "
                    f"{dataset_id!r}; got {names}"
                )
            member = matches[0]
            member_name = str(member["name"])
            if member_name in selected:
                raise RuntimeError(f"archive member selected twice: {member_name}")
            selected[member_name] = {
                "dataset_id": dataset_id,
                "kind": kind,
                "expected_bytes": int(member["bytes"]),
                "output": output_root / f"{dataset_slug(dataset_id)}{output_suffix}",
            }
    return selected


def validate_holdout_seal(root: Path, seal_path: Path) -> dict[str, str | int]:
    """Revalidate a supplied seal against Git and its frozen profile."""

    supplied = json.loads(seal_path.read_text(encoding="utf-8"))
    if supplied.get("status") != "PASS" or not supplied.get("profile"):
        raise RuntimeError("holdout profile seal is not PASS")
    profile = root.resolve() / str(supplied["profile"])
    current = verify_profile_seal(root.resolve(), profile)
    protected_fields = (
        "status",
        "profile",
        "profile_sha256",
        "profile_commit",
        "checked_at_head",
    )
    if any(supplied.get(field) != current[field] for field in protected_fields):
        raise RuntimeError("holdout profile seal is stale or does not match Git state")
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--verification", required=True, type=Path)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--mode", choices=("development", "holdout"), required=True)
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--profile-seal", type=Path)
    parser.add_argument("--root", default=Path("."), type=Path)
    args = parser.parse_args()

    archive_path = args.archive.resolve()
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise RuntimeError("archive has not passed frozen MD5 verification")
    if Path(verification.get("archive", "")).resolve() != archive_path:
        raise RuntimeError("archive path does not match verification evidence")
    if args.mode == "holdout" and (
        args.profile_seal is None or not args.profile_seal.is_file()
    ):
        raise RuntimeError("holdout extraction requires committed RF profile seal")
    seal = (
        validate_holdout_seal(args.root, args.profile_seal)
        if args.mode == "holdout" and args.profile_seal is not None
        else None
    )
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    if Path(inventory.get("archive", "")).resolve() != archive_path:
        raise RuntimeError("inventory archive path does not match selected archive")
    if inventory.get("archive_md5") != verification.get("md5"):
        raise RuntimeError("inventory MD5 does not match verification evidence")
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = select_members(inventory, args.dataset, args.output_root)
    records = {
        dataset_id: {"dataset_id": dataset_id, "files": {}}
        for dataset_id in args.dataset
    }
    remaining = set(selected)
    # Stream the gzip archive once. Random seeks would repeatedly decompress the
    # 27.6 GB archive and make selective extraction needlessly expensive.
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            member_name = member.name
            if member_name not in remaining:
                continue
            selection = selected[member_name]
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(
                    f"unsafe/non-file selected archive member: {member_name}"
                )
            if member.size != selection["expected_bytes"]:
                raise RuntimeError(f"inventory size mismatch for {member_name}")
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read selected member: {member_name}")
            output = Path(selection["output"])
            temporary = Path(f"{output}.tmp")
            digest = hashlib.sha256()
            size = 0
            with source, temporary.open("wb") as handle:
                while True:
                    chunk = source.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != member.size:
                raise RuntimeError(f"extracted size mismatch for {member_name}")
            temporary.replace(output)
            dataset_id = str(selection["dataset_id"])
            kind = str(selection["kind"])
            records[dataset_id]["files"][kind] = {
                "archive_member": member_name,
                "path": str(output.resolve()),
                "bytes": size,
                "sha256": digest.hexdigest(),
            }
            remaining.remove(member_name)
            if not remaining:
                break
    if remaining:
        raise RuntimeError(
            "selected members were not found while streaming: "
            f"{sorted(remaining)}"
        )
    atomic_json(
        args.manifest,
        {
            "schema_version": 2,
            "mode": args.mode,
            "archive_md5": inventory["archive_md5"],
            "profile_seal": (
                str(args.profile_seal.resolve()) if args.profile_seal else None
            ),
            "profile_sha256": seal.get("profile_sha256") if seal else None,
            "profile_commit": seal.get("profile_commit") if seal else None,
            "datasets": sorted(
                records.values(),
                key=lambda record: dataset_number(str(record["dataset_id"])),
            ),
        },
    )


if __name__ == "__main__":
    main()
