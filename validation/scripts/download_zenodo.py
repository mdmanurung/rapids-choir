#!/usr/bin/env python3
"""Resumable, atomic download of the frozen Zenodo archive."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-bytes", required=True, type=int)
    args = parser.parse_args()

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        if output.stat().st_size != args.expected_bytes:
            raise RuntimeError(
                f"existing archive has {output.stat().st_size} bytes; "
                f"expected {args.expected_bytes}; refusing to overwrite"
            )
        return

    partial = output.with_suffix(output.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > args.expected_bytes:
        raise RuntimeError("partial download exceeds frozen archive size")
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(args.url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        status = getattr(response, "status", None)
        append = offset > 0 and status == 206
        if offset and not append:
            raise RuntimeError("server did not honor the resume Range request")
        mode = "ab" if append else "wb"
        with partial.open(mode) as handle:
            downloaded = offset
            while True:
                chunk = response.read(8 * 1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024 * 1024 * 1024) < len(chunk):
                    print(
                        f"downloaded {downloaded}/{args.expected_bytes} bytes",
                        file=sys.stderr,
                        flush=True,
                    )
    if partial.stat().st_size != args.expected_bytes:
        raise RuntimeError(
            f"download ended at {partial.stat().st_size} bytes; "
            f"expected {args.expected_bytes}"
        )
    partial.replace(output)


if __name__ == "__main__":
    main()
