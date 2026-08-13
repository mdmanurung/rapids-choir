#!/usr/bin/env python3
"""Extract only an explicitly frozen development or sealed holdout set."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
from pathlib import Path, PurePosixPath

from check_profile_seal import verify_profile_seal
from gate2_common import atomic_json, dataset_number, dataset_slug


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def member_matches(name: str, dataset_id: str) -> bool:
    number = dataset_number(dataset_id)
    targets = {
        f"simulateddataset{number}",
        f"dataset{number}",
        f"sim{number}",
    }
    path = PurePosixPath(name)
    components = [normalized(part.rsplit(".", 1)[0]) for part in path.parts]
    return path.suffix.lower() == ".rds" and bool(targets & set(components))


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
    file_names = [
        member["name"]
        for member in inventory["members"]
        if member["type"] == "file"
    ]
    selected: dict[str, str] = {}
    for dataset_id in args.dataset:
        matches = [name for name in file_names if member_matches(name, dataset_id)]
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one archive member for {dataset_id!r}; got {matches}"
            )
        selected[matches[0]] = dataset_id

    args.output_root.mkdir(parents=True, exist_ok=True)
    records = []
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member_name, dataset_id in selected.items():
            member = archive.getmember(member_name)
            if not member.isfile() or member.issym() or member.islnk():
                raise RuntimeError(
                    f"unsafe/non-file selected archive member: {member_name}"
                )
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"could not read selected member: {member_name}")
            output = args.output_root / f"{dataset_slug(dataset_id)}.rds"
            temporary = output.with_suffix(output.suffix + ".tmp")
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
            records.append(
                {
                    "dataset_id": dataset_id,
                    "archive_member": member_name,
                    "path": str(output.resolve()),
                    "bytes": size,
                    "sha256": digest.hexdigest(),
                }
            )
    atomic_json(
        args.manifest,
        {
            "schema_version": 1,
            "mode": args.mode,
            "archive_md5": inventory["archive_md5"],
            "profile_seal": (
                str(args.profile_seal.resolve()) if args.profile_seal else None
            ),
            "profile_sha256": seal.get("profile_sha256") if seal else None,
            "profile_commit": seal.get("profile_commit") if seal else None,
            "datasets": sorted(records, key=lambda record: record["dataset_id"]),
        },
    )


if __name__ == "__main__":
    main()
