#!/usr/bin/env python3
"""Run a non-Python producer and preserve immutable logs and failure evidence."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from validation.scripts.gate2_common import (
    immutable_text,
    preserve_failure_record,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--failure-anchor", required=True, type=Path)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise RuntimeError("preserved command cannot be empty")
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    immutable_text(
        arguments.log,
        (
            f"$ {' '.join(command)}\n"
            f"STDOUT\n{completed.stdout}\n"
            f"STDERR\n{completed.stderr}"
        ),
    )
    if completed.returncode == 0:
        return 0
    preserve_failure_record(
        arguments.failure_anchor,
        scope=arguments.scope,
        stage=arguments.stage,
        method_started=True,
        failure_classification="technical_outcome",
        evidence={
            "command": command,
            "returncode": completed.returncode,
            "log": str(arguments.log.resolve()),
        },
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
