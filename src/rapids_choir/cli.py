"""Command-line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from . import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rapids-choir")
    parser.add_argument("--version", action="version", version=__version__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor", help="validate the real GPU runtime")
    doctor.add_argument("--device", default="cuda:0")
    doctor.add_argument("--strict", action="store_true")
    doctor.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "doctor":
        from ._doctor import run_doctor

        report = run_doctor(args.device)
        if args.as_json:
            print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
        else:
            print(f"rapids-choir doctor: {report.status}")
            for check in report.checks:
                marker = "PASS" if check.passed else "FAIL"
                message = f"  {marker} {check.name}"
                if check.error:
                    message += f": {check.error}"
                print(message)
        return 1 if args.strict and report.status != "PASS" else 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
