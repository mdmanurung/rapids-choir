from __future__ import annotations

import numpy as np

from rapids_choir._semantic_trace import TraceRecorder
from rapids_choir.tl._hierarchy import ClusteringFailure, find_starting_resolution


def test_resolution_one_failure_records_every_attempted_transition() -> None:
    def partition(resolution: float, members: np.ndarray, seed: int) -> np.ndarray:
        del seed
        if resolution == 1.0:
            raise ClusteringFailure("fixture failure")
        return np.zeros(members.size, dtype=np.int64)

    recorder = TraceRecorder("run-hierarchy-trace")
    result = find_starting_resolution(
        partition,
        np.arange(4, dtype=np.int64),
        random_state=7,
        trace=recorder,
    )
    assert result.resolution == 0.1
    assert [event.event_type for event in recorder.events] == [
        "resolution_transition",
        "resolution_transition",
        "resolution_search",
    ]
    assert [event.decision for event in recorder.events[:2]] == [
        "baseline",
        "failure",
    ]
