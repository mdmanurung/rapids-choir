"""Real-GPU runtime diagnostics."""

from __future__ import annotations

import importlib
import platform
from dataclasses import asdict, dataclass, field
from typing import Any

from ._cuda import initialize_cuda
from .config import RFProfile


@dataclass(slots=True)
class Check:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class DoctorReport:
    schema_version: int
    status: str
    device: str
    checks: list[Check]
    environment: dict[str, str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _run_check(name: str, function: Any) -> Check:
    try:
        details = function()
    except Exception as error:
        return Check(name=name, passed=False, error=f"{type(error).__name__}: {error}")
    return Check(name=name, passed=True, details=details)


def run_doctor(device: str = "cuda:0") -> DoctorReport:
    context_holder: dict[str, Any] = {}

    def device_check() -> dict[str, Any]:
        context = initialize_cuda(device)
        context_holder["context"] = context
        free_bytes, total_bytes = context.memory_info()
        return {
            "name": context.device_name,
            "index": context.device_index,
            "compute_capability": ".".join(map(str, context.compute_capability)),
            "free_bytes": free_bytes,
            "total_bytes": total_bytes,
        }

    checks = [_run_check("cuda_device_and_rmm", device_check)]

    def versions_check() -> dict[str, Any]:
        context = context_holder["context"]
        cp = context.cupy
        driver = int(cp.cuda.runtime.driverGetVersion())
        runtime = int(cp.cuda.runtime.runtimeGetVersion())
        if driver < runtime:
            raise RuntimeError(f"CUDA driver {driver} is older than runtime {runtime}")
        versions: dict[str, str | int] = {
            "driver": driver,
            "runtime": runtime,
            "cupy": importlib.import_module("cupy").__version__,
            "rmm": importlib.import_module("rmm").__version__,
        }
        for name in ("cuml", "cugraph", "rapids_singlecell"):
            versions[name] = importlib.import_module(name).__version__
        return versions

    checks.append(_run_check("driver_runtime_and_versions", versions_check))

    def allocation_check() -> dict[str, Any]:
        cp = context_holder["context"].cupy
        before_free, _ = context_holder["context"].memory_info()
        allocation = cp.arange(4096, dtype=cp.float64)
        checksum = float(cp.sum(allocation, dtype=cp.float64).item())
        pointer = int(allocation.data.ptr)
        del allocation
        after_free, _ = context_holder["context"].memory_info()
        return {
            "allocator": "rmm_cupy_allocator",
            "pointer_nonzero": pointer > 0,
            "checksum": checksum,
            "free_bytes_before": before_free,
            "free_bytes_after": after_free,
        }

    checks.append(_run_check("rmm_cupy_allocation", allocation_check))

    def random_forest_check() -> dict[str, Any]:
        cp = context_holder["context"].cupy
        classifier_type = importlib.import_module(
            "cuml.ensemble"
        ).RandomForestClassifier
        profile = RFProfile()
        features = cp.ascontiguousarray(
            cp.asarray(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] * 8,
                dtype=cp.float32,
            )
        )
        labels = cp.ascontiguousarray(cp.asarray([0, 0, 1, 1] * 8, dtype=cp.int32))
        classifier = classifier_type(
            n_estimators=4,
            max_depth=4,
            split_criterion=profile.split_criterion,
            max_features=profile.max_features,
            bootstrap=profile.bootstrap,
            min_samples_leaf=profile.min_samples_leaf,
            min_samples_split=profile.min_samples_split,
            max_leaves=profile.max_leaves,
            max_samples=profile.max_samples,
            n_bins=profile.n_bins,
            n_streams=profile.n_streams,
            random_state=1,
            output_type="cupy",
        )
        classifier.fit(features, labels, convert_dtype=False)
        predicted = classifier.predict(features, convert_dtype=False)
        accuracy = float(cp.mean(predicted.astype(cp.int32) == labels).item())
        if accuracy < 0.5:
            raise RuntimeError(f"tiny cuML RF accuracy is unexpectedly low: {accuracy}")
        return {
            "feature_dtype": str(features.dtype),
            "label_dtype": str(labels.dtype),
            "convert_dtype": False,
            "accuracy": accuracy,
        }

    checks.append(_run_check("cuml_random_forest", random_forest_check))

    def louvain_check() -> dict[str, Any]:
        cudf = importlib.import_module("cudf")
        cugraph = importlib.import_module("cugraph")
        edges = cudf.DataFrame(
            {
                "source": [0, 1, 2, 3, 0, 2],
                "destination": [1, 0, 3, 2, 2, 0],
                "weight": [1.0, 1.0, 1.0, 1.0, 0.01, 0.01],
            }
        )
        graph = cugraph.Graph(directed=False)
        graph.from_cudf_edgelist(
            edges,
            source="source",
            destination="destination",
            edge_attr="weight",
            renumber=False,
        )
        partitions, modularity = cugraph.louvain(graph, resolution=1.0)
        if len(partitions) != 4:
            raise RuntimeError("tiny Louvain did not return every vertex")
        return {
            "vertices": len(partitions),
            "clusters": int(partitions["partition"].nunique()),
            "modularity": float(modularity),
        }

    checks.append(_run_check("cugraph_louvain", louvain_check))
    status = "PASS" if all(check.passed for check in checks) else "FAIL"
    return DoctorReport(
        schema_version=1,
        status=status,
        device=device,
        checks=checks,
        environment={
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
