from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path(__file__).parents[2]
SCRIPTS = ROOT / "validation/scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "build_pair_manifest", SCRIPTS / "build_pair_manifest.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_blake2b_halves_are_balanced_and_order_independent() -> None:
    cells = [f"cell{i}" for i in range(9)]
    first, second = MODULE.stable_halves(cells, 8675309)
    reversed_first, reversed_second = MODULE.stable_halves(
        list(reversed(cells)), 8675309
    )
    assert (first, second) == (reversed_first, reversed_second)
    assert len(first) == 4
    assert len(second) == 5
    assert set(first).isdisjoint(second)


def test_cross_label_pairs_require_an_nn_edge() -> None:
    graph = sparse.csr_matrix(
        np.asarray(
            [
                [0, 1, 0, 0],
                [1, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
    )
    labels = np.asarray(["a", "a", "b", "c"])
    assert MODULE.cross_label_edges(graph, labels) == {("a", "b"): 2}
