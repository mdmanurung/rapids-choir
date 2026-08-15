from pathlib import Path
from itertools import product
from datetime import UTC, datetime
import json
import os
import shlex
import subprocess
import sys
import yaml

from workflow.guards import enforce_executor_context

enforce_executor_context(sys.argv, os.environ, base_dir=Path.cwd())

configfile: "workflow/config.yaml"

CANDIDATE_COMMIT = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
CANDIDATE_ATTEMPT = config.get("candidate_attempt", 1)
if (
    isinstance(CANDIDATE_ATTEMPT, bool)
    or not isinstance(CANDIDATE_ATTEMPT, int)
    or CANDIDATE_ATTEMPT < 1
):
    raise ValueError("candidate_attempt must be a positive integer")
CANDIDATE_KEY = f"candidate-{CANDIDATE_COMMIT}/attempt-{CANDIDATE_ATTEMPT}"
EVIDENCE_ROOT = f"validation/evidence/candidates/{CANDIDATE_KEY}"
LOG_ROOT = f"validation/logs/candidates/{CANDIDATE_KEY}"
GATE0_EVIDENCE = f"{EVIDENCE_ROOT}/gate0.json"
GATE1_STATIC_EVIDENCE = f"{EVIDENCE_ROOT}/gate1_static.json"
GATE1_GPU_EVIDENCE = f"{EVIDENCE_ROOT}/gate1_gpu.json"
GATE2_EVIDENCE = f"{EVIDENCE_ROOT}/gate2.json"
GATE2_PASS_SEAL = f"{EVIDENCE_ROOT}/gate2.PASS.json"
CANDIDATE_MANIFEST = f"{EVIDENCE_ROOT}/candidate.json"
PARITY_FRAMEWORK_SEAL = f"{EVIDENCE_ROOT}/parity/framework.READY.json"
HIERARCHY_MARGIN_EVIDENCE = f"{EVIDENCE_ROOT}/hierarchy/margins.FROZEN.json"
HIERARCHY_FROZEN_MARGIN_CONTRACT = (
    "validation/contracts/hierarchy_fidelity_margins_v1.json"
)
HIERARCHY_MARGIN_REPRODUCTION = (
    f"{EVIDENCE_ROOT}/hierarchy/margins.REPRODUCED.json"
)
HIERARCHY_AGGREGATE_EVIDENCE = f"{EVIDENCE_ROOT}/hierarchy/fidelity.json"
PARITY_PROMOTION_EVIDENCE = f"{EVIDENCE_ROOT}/parity/promotion.READY.json"

FIXTURE_NAMES = [
    "batch_eligibility",
    "batch_positional_metrics",
    "decision_rules",
    "disconnected_graph",
    "kde_bootstrap",
    "multiplicity",
    "odd_iterations",
    "random_state",
    "repeat_error_identity",
    "session",
    "source_assertions",
    "subtree_distance_source",
]
FIXTURE_FILES = [f"validation/fixtures/generated/{name}.json" for name in FIXTURE_NAMES]
REFERENCE_FILES = sorted(
    str(path)
    for path in Path("validation/reference/choir_v0_3_0").glob("**/*")
    if path.is_file()
)
STATIC_PACKAGE_FILES = sorted(
    str(path)
    for directory in ("src", "tests", "validation/scripts")
    for path in Path(directory).glob("**/*")
    if path.is_file()
    and path.suffix in {".py", ".R", ".json"}
    and "__pycache__" not in path.parts
)
ASSURANCE_CONTRACT_FILES = [
    "validation/contracts/data_roles_v2.json",
    "validation/contracts/hierarchy_development_objects_v1.json",
    "validation/contracts/hierarchy_fidelity_margins_v1.json",
    "validation/contracts/hierarchy_result_v1.schema.json",
    "validation/contracts/hierarchy_validation_v1.json",
    "validation/contracts/l40s_performance_v1.json",
    "validation/contracts/parity_ledger_v1.json",
    "validation/contracts/parity_mutations_v1.json",
    "validation/contracts/publication_claims_v2.json",
    "validation/contracts/release_candidate_v1.json",
    "validation/contracts/sant_reproduction_v1.json",
    "validation/contracts/secondary_evidence_v1.json",
    "validation/contracts/semantic_trace_v1.schema.json",
    "validation/contracts/v0_1_scope.json",
]
HIERARCHY_DESIGN_PATH = "validation/contracts/hierarchy_validation_v1.json"
HIERARCHY_DEVELOPMENT_CONTRACT_PATH = (
    "validation/contracts/hierarchy_development_objects_v1.json"
)
PUBLICATION_DATA_ROLES_PATH = "validation/contracts/data_roles_v2.json"
REFERENCE_SOURCE_ROOT = "validation/reference/choir_v0_3_0"
SECONDARY_EVIDENCE_CONTRACT = (
    "validation/contracts/secondary_evidence_v1.json"
)
SANT_REPRODUCTION_CONTRACT = (
    "validation/contracts/sant_reproduction_v1.json"
)
PERFORMANCE_CONTRACT = "validation/contracts/l40s_performance_v1.json"
RELEASE_CANDIDATE_CONTRACT = (
    "validation/contracts/release_candidate_v1.json"
)
with open(HIERARCHY_DESIGN_PATH, encoding="utf-8") as handle:
    HIERARCHY_DESIGN = json.load(handle)
with open(HIERARCHY_DEVELOPMENT_CONTRACT_PATH, encoding="utf-8") as handle:
    HIERARCHY_DEVELOPMENT_CONTRACT = json.load(handle)
HIERARCHY_DEVELOPMENT_IDS = [
    record["dataset_id"]
    for record in HIERARCHY_DESIGN["development"]["objects"]
]
HIERARCHY_CONFIRMATION_IDS = [
    record["dataset_id"]
    for record in HIERARCHY_DESIGN["confirmation"]["datasets"]
]
HIERARCHY_ROLE_BY_ID = {
    record["dataset_id"]: record["role"]
    for record in HIERARCHY_DEVELOPMENT_CONTRACT["objects"]
}
if list(HIERARCHY_ROLE_BY_ID) != HIERARCHY_DEVELOPMENT_IDS:
    raise ValueError("hierarchy producer roles drifted from the frozen design")
HIERARCHY_REFERENCE_SEEDS = tuple(
    int(value) for value in HIERARCHY_DEVELOPMENT_CONTRACT["reference_seeds"]
)
if HIERARCHY_REFERENCE_SEEDS != (104729, 104759):
    raise ValueError("hierarchy reference seed pair has drifted")
HIERARCHY_CALIBRATION_REPORTS = [
    f"{EVIDENCE_ROOT}/hierarchy/calibration/{dataset_id}.json"
    for dataset_id in HIERARCHY_DEVELOPMENT_IDS
]
HIERARCHY_DEVELOPMENT_REPORTS = [
    f"{EVIDENCE_ROOT}/hierarchy/development/{dataset_id}.json"
    for dataset_id in HIERARCHY_DEVELOPMENT_IDS
]
HIERARCHY_CONFIRMATION_REPORTS = [
    f"{EVIDENCE_ROOT}/hierarchy/confirmation/{dataset_id}.json"
    for dataset_id in HIERARCHY_CONFIRMATION_IDS
]
SECONDARY_DATASET_IDS = ("Stoeckius", "Kang", "pancreas")
SANT_ANALYSIS_IDS = (
    "221_non_comparator_simulations",
    "Kinker_RNA",
    "Hao_RNA_only",
    "Srivatsan_RNA_only",
)
PERFORMANCE_WORKLOAD_IDS = tuple(
    f"workload_{index:02d}" for index in range(1, 7)
)
SECONDARY_REPORTS = [
    f"{EVIDENCE_ROOT}/publication/secondary/{dataset_id}.json"
    for dataset_id in SECONDARY_DATASET_IDS
]
SANT_RECORDS = [
    f"{EVIDENCE_ROOT}/publication/sant/{analysis_id}.json"
    for analysis_id in SANT_ANALYSIS_IDS
]
PERFORMANCE_RECORDS = [
    f"{EVIDENCE_ROOT}/publication/performance/{workload_id}.json"
    for workload_id in PERFORMANCE_WORKLOAD_IDS
]
SECONDARY_AGGREGATE = f"{EVIDENCE_ROOT}/publication/secondary.json"
SANT_AGGREGATE = f"{EVIDENCE_ROOT}/publication/sant.json"
PERFORMANCE_AGGREGATE = f"{EVIDENCE_ROOT}/publication/performance.json"
PUBLICATION_EVIDENCE = f"{EVIDENCE_ROOT}/publication/evidence.READY.json"
RELEASE_CANDIDATE_EVIDENCE = f"{EVIDENCE_ROOT}/release/candidate.READY.json"
TRACKED_REPOSITORY_FILES = subprocess.run(
    ["git", "ls-files"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
RUNTIME_CONTROL_FILES = sorted(
    set(TRACKED_REPOSITORY_FILES)
    | set(STATIC_PACKAGE_FILES)
    | set(REFERENCE_FILES)
    | set(ASSURANCE_CONTRACT_FILES)
    | {
        ".gitignore",
        "CITATION.cff",
        "LICENSE",
        "NOTICE",
        "README.md",
        "Snakefile",
        "docs/architecture/architecture.md",
        "docs/decisions/README.md",
        "docs/validation/parity-assurance.md",
        "docs/validation/source-audit.md",
        "pyproject.toml",
        "renv.lock",
        "validation/contracts/comparator_candidate_v2.yaml",
        SECONDARY_EVIDENCE_CONTRACT,
        SANT_REPRODUCTION_CONTRACT,
        PERFORMANCE_CONTRACT,
        RELEASE_CANDIDATE_CONTRACT,
        "validation/fixtures/specs/datasets.yaml",
        "validation/governance/resource_budget.yaml",
        "validation/governance/README.md",
        "validation/governance/RF_PROFILE_PROMOTION.md",
        "workflow/config.yaml",
        "workflow/envs/rapids-cu128-linux-64.lock",
        "workflow/envs/rapids-cu128-pip.lock",
        "workflow/envs/rapids-cu128.yml",
        "workflow/profiles/slurm/config.yaml",
        "workflow/authorization/README.md",
        "workflow/guards.py",
    }
)

DATASET_SPEC_PATH = "validation/fixtures/specs/datasets.yaml"
with open(DATASET_SPEC_PATH, encoding="utf-8") as handle:
    DATASET_SPEC = yaml.safe_load(handle)
COMPARATOR_CONTRACT_PATH = "validation/contracts/comparator_candidate_v2.yaml"
with open(COMPARATOR_CONTRACT_PATH, encoding="utf-8") as handle:
    COMPARATOR_CONTRACT = yaml.safe_load(handle)
RESOURCE_BUDGET_PATH = "validation/governance/resource_budget.yaml"
with open(RESOURCE_BUDGET_PATH, encoding="utf-8") as handle:
    RESOURCE_BUDGET = yaml.safe_load(handle)
if RESOURCE_BUDGET["limits"] != {
    "l40s_hours": 250.0,
    "scratch_bytes": 2000000000000,
    "cloud_spend_usd": 0.0,
}:
    raise ValueError("v0.1 resource limits drifted")
RESOURCE_LEDGER = "validation/evidence/governance/resource_ledger.jsonl"


def dataset_slug(dataset_id):
    return f"simulated_dataset_{int(dataset_id.rsplit(' ', 1)[1]):03d}"


DEVELOPMENT = DATASET_SPEC["development"]["datasets"]
HOLDOUT = DATASET_SPEC["holdout"]["datasets"]
if COMPARATOR_CONTRACT["roles"]["development"] != [
    record["id"] for record in DEVELOPMENT
]:
    raise ValueError("comparator development roles drifted from the v2 contract")
if COMPARATOR_CONTRACT["roles"]["sealed_holdout"] != [
    record["id"] for record in HOLDOUT
]:
    raise ValueError("comparator holdout roles drifted from the v2 contract")
REFERENCE_SEED_PAIRS = tuple(
    tuple(int(value) for value in pair)
    for pair in COMPARATOR_CONTRACT["reference_variability"]["seed_pairs"]
)
REFERENCE_SEEDS = tuple(sorted({value for pair in REFERENCE_SEED_PAIRS for value in pair}))
DEV_BY_SLUG = {dataset_slug(record["id"]): record for record in DEVELOPMENT}
HOLD_BY_SLUG = {dataset_slug(record["id"]): record for record in HOLDOUT}
RAW_ROOT = config["data"]["raw_root"]
WORK_ROOT = config["data"]["work_root"]
CANDIDATE_WORK_ROOT = f"{WORK_ROOT}/candidates/{CANDIDATE_KEY}"
ARCHIVE = f"{RAW_ROOT}/{DATASET_SPEC['dataset_source']['archive']['name']}"
ARCHIVE_VERIFY = f"{CANDIDATE_WORK_ROOT}/inputs/Simulated_Datasets.md5.json"
ARCHIVE_INVENTORY = (
    f"{CANDIDATE_WORK_ROOT}/inputs/Simulated_Datasets.members.json"
)
DEV_RAW_ROOT = f"{CANDIDATE_WORK_ROOT}/development/raw"
HOLD_RAW_ROOT = f"{CANDIDATE_WORK_ROOT}/holdout/raw"
DEV_EXTRACT_MANIFEST = f"{CANDIDATE_WORK_ROOT}/development/extraction.json"
HOLD_EXTRACT_MANIFEST = f"{CANDIDATE_WORK_ROOT}/holdout/extraction.json"
TRUTH_CONTRACT = f"{CANDIDATE_WORK_ROOT}/development/truth_contract.json"
PROFILE_SEAL = f"{CANDIDATE_WORK_ROOT}/holdout/rf_profile.sealed.json"
# The winning profile is the one tracked promotion input. Its internal hashes
# bind it to candidate-specific immutable calibration evidence.
RF_PROFILE = "validation/evidence/rf_profile.json"
CALIBRATION_PROFILE = f"{EVIDENCE_ROOT}/gate2/rf_profile.candidate.json"
RF_PROFILE_REPRODUCTION = (
    f"{EVIDENCE_ROOT}/gate2/rf_profile.REPRODUCED.json"
)
AUTHORIZATION_CONFIG = config["authorization"]
COMPARATOR_AUTHORIZATION = AUTHORIZATION_CONFIG["comparator_confirm"]
HIERARCHY_AUTHORIZATIONS = AUTHORIZATION_CONFIG.get("hierarchy_confirm", {})
COMPARATOR_AUTHORIZATION_SEAL = (
    f"{CANDIDATE_WORK_ROOT}/authorization/comparator_confirm.json"
)
GATE1_RESERVATION = f"{EVIDENCE_ROOT}/reservations/gate1_gpu.json"
DEVELOPMENT_RESERVATION = (
    f"{EVIDENCE_ROOT}/reservations/comparator_development.json"
)
CONFIRM_RESERVATION = f"{EVIDENCE_ROOT}/reservations/comparator_confirm.json"
HIERARCHY_DEVELOPMENT_RESERVATION = (
    f"{EVIDENCE_ROOT}/reservations/hierarchy_development.json"
)
SCHEDULER_SMOKE = f"{CANDIDATE_WORK_ROOT}/smoke/scheduler.json"
WORST_SHARD_CANARY = f"{CANDIDATE_WORK_ROOT}/smoke/worst_shard_canary.json"


def hierarchy_development_path(dataset_id, kind):
    return f"{CANDIDATE_WORK_ROOT}/hierarchy/development/{dataset_id}.{kind}"


def hierarchy_python_trace_path(dataset_id):
    return (
        f"{EVIDENCE_ROOT}/hierarchy/runs/development/{dataset_id}/"
        "python.seed104729.trace.jsonl"
    )


def hierarchy_python_result_path(dataset_id):
    return (
        f"{EVIDENCE_ROOT}/hierarchy/runs/development/{dataset_id}/"
        "python.seed104729.result.json"
    )


def hierarchy_r_path(dataset_id, seed, kind, phase="development"):
    return (
        f"{EVIDENCE_ROOT}/hierarchy/runs/{phase}/{dataset_id}/"
        f"r.seed{seed}.{kind}"
    )


def hierarchy_confirmation_work_path(dataset_id, kind):
    return f"{CANDIDATE_WORK_ROOT}/hierarchy/confirmation/{dataset_id}.{kind}"


def hierarchy_confirmation_python_path(dataset_id, kind):
    return (
        f"{EVIDENCE_ROOT}/hierarchy/runs/confirmation/{dataset_id}/"
        f"python.seed104729.{kind}"
    )


def hierarchy_confirmation_reservation_path(dataset_id):
    return (
        f"{EVIDENCE_ROOT}/reservations/"
        f"hierarchy_confirm_{dataset_id}.json"
    )


def current_comparator_authorization(_wildcards):
    path = Path(COMPARATOR_AUTHORIZATION_SEAL)
    if not path.is_file():
        return path.as_posix()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "AUTHORIZED":
        raise ValueError("comparator authorization seal is not AUTHORIZED")
    if payload.get("target") != "comparator_confirm":
        raise ValueError("comparator authorization target drift")
    if payload.get("candidate_commit") != CANDIDATE_COMMIT:
        raise ValueError("comparator authorization candidate drift")
    if payload.get("attempt") != CANDIDATE_ATTEMPT:
        raise ValueError("comparator authorization attempt drift")
    expires = datetime.fromisoformat(payload["expires_at"])
    if expires.astimezone(UTC) <= datetime.now(UTC):
        raise ValueError(
            "comparator authorization expired; increment candidate_attempt "
            "only under the frozen rerun policy"
        )
    return path.as_posix()


def hierarchy_confirmation_config(dataset_id, field):
    record = config.get("hierarchy_confirmation", {}).get(dataset_id)
    if not isinstance(record, dict) or not record.get(field):
        raise ValueError(
            f"hierarchy confirmation {dataset_id} requires configured {field}"
        )
    return str(record[field])


def hierarchy_authorization_config(dataset_id, field):
    record = HIERARCHY_AUTHORIZATIONS.get(dataset_id)
    if not isinstance(record, dict) or not record.get(field):
        raise ValueError(
            f"hierarchy confirmation {dataset_id} requires signed {field}"
        )
    return str(record[field])


def current_hierarchy_authorization(wildcards):
    dataset_id = wildcards.dataset
    path = Path(hierarchy_confirmation_work_path(dataset_id, "authorization.json"))
    if not path.is_file():
        return path.as_posix()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "AUTHORIZED"
        or payload.get("target") != f"hierarchy_confirm_{dataset_id}"
        or payload.get("dataset_roles")
        != [f"hierarchy_parity_confirmation:{dataset_id}"]
        or payload.get("candidate_commit") != CANDIDATE_COMMIT
        or payload.get("attempt") != CANDIDATE_ATTEMPT
    ):
        raise ValueError(f"hierarchy authorization drift: {dataset_id}")
    expires = datetime.fromisoformat(payload["expires_at"])
    if expires.astimezone(UTC) <= datetime.now(UTC):
        raise ValueError(f"hierarchy authorization expired: {dataset_id}")
    return path.as_posix()


def configured_artifact(section, role):
    value = config.get(section, {}).get(role)
    if not value:
        raise ValueError(
            f"guarded target requires workflow config {section}.{role}"
        )
    return str(value)


def configured_artifact_arguments(section, roles):
    return " ".join(
        "--artifact "
        + shlex.quote(f"{role}={configured_artifact(section, role)}")
        for role in roles
    )


def raw_path(mode, record):
    root = DEV_RAW_ROOT if mode == "development" else HOLD_RAW_ROOT
    return f"{root}/{dataset_slug(record['id'])}.rds"


def metadata_path(mode, record):
    root = DEV_RAW_ROOT if mode == "development" else HOLD_RAW_ROOT
    return f"{root}/{dataset_slug(record['id'])}.metadata.csv"


def h5ad_path(mode, record):
    return f"{CANDIDATE_WORK_ROOT}/{mode}/h5ad/{dataset_slug(record['id'])}.h5ad"


def data_manifest_path(mode, record):
    return (
        f"{CANDIDATE_WORK_ROOT}/{mode}/manifests/"
        f"{dataset_slug(record['id'])}.json"
    )


def pair_manifest_path(mode, record):
    return (
        f"{CANDIDATE_WORK_ROOT}/{mode}/pairs/"
        f"{dataset_slug(record['id'])}.json"
    )


def reference_path(mode, record):
    return (
        f"{CANDIDATE_WORK_ROOT}/{mode}/reference/"
        f"{dataset_slug(record['id'])}.json"
    )


def reference_variability_path(record, seed):
    return (
        f"{CANDIDATE_WORK_ROOT}/development/reference_variability/"
        f"{dataset_slug(record['id'])}.seed{seed}.json"
    )


DEV_RDS = [raw_path("development", record) for record in DEVELOPMENT]
HOLD_RDS = [raw_path("holdout", record) for record in HOLDOUT]
DEV_METADATA = [metadata_path("development", record) for record in DEVELOPMENT]
HOLD_METADATA = [metadata_path("holdout", record) for record in HOLDOUT]
DEV_H5AD = [h5ad_path("development", record) for record in DEVELOPMENT]
HOLD_H5AD = [h5ad_path("holdout", record) for record in HOLDOUT]
DEV_DATA_MANIFESTS = [
    data_manifest_path("development", record) for record in DEVELOPMENT
]
HOLD_DATA_MANIFESTS = [data_manifest_path("holdout", record) for record in HOLDOUT]
DEV_PAIR_MANIFESTS = [pair_manifest_path("development", record) for record in DEVELOPMENT]
HOLD_PAIR_MANIFESTS = [pair_manifest_path("holdout", record) for record in HOLDOUT]
DEV_REFERENCES = [reference_path("development", record) for record in DEVELOPMENT]
HOLD_REFERENCES = [reference_path("holdout", record) for record in HOLDOUT]
DEV_REFERENCE_VARIABILITY = [
    reference_variability_path(record, seed)
    for record in DEVELOPMENT
    for seed in REFERENCE_SEEDS
]
DEV_MARGIN_CONTRACT = (
    f"{CANDIDATE_WORK_ROOT}/development/margins/reference_variability.json"
)
DEV_MARGIN_ELIGIBILITY = (
    f"{CANDIDATE_WORK_ROOT}/development/margins/ELIGIBLE.json"
)

RF_GRID = [
    {
        "id": f"bins{n_bins}_depth{'none' if depth is None else depth}_streams{streams}",
        "n_bins": n_bins,
        "max_depth": "none" if depth is None else str(depth),
        "n_streams": streams,
    }
    for n_bins, depth, streams in product(
        COMPARATOR_CONTRACT["profile_grid"]["n_bins"],
        COMPARATOR_CONTRACT["profile_grid"]["max_depth"],
        COMPARATOR_CONTRACT["profile_grid"]["n_streams"],
    )
]
RF_BY_ID = {record["id"]: record for record in RF_GRID}
GPU_REPEAT_INDICES = tuple(
    range(1, int(COMPARATOR_CONTRACT["profile_grid"]["gpu_repeats"]) + 1)
)
if len(RF_GRID) != 12 or GPU_REPEAT_INDICES != (1, 2, 3, 4, 5):
    raise ValueError("comparator profile grid or repeat count drifted")
CALIBRATION_SUMMARIES = [
    f"{EVIDENCE_ROOT}/gate2/calibration/{record['id']}.json"
    for record in RF_GRID
]
CALIBRATION_PARQUETS = [
    f"{EVIDENCE_ROOT}/gate2/calibration/{record['id']}.parquet"
    for record in RF_GRID
]


def calibration_chunk_path(profile, dataset, repeat, suffix):
    return (
        f"{EVIDENCE_ROOT}/gate2/calibration/chunks/"
        f"{profile}/{dataset}.repeat{repeat}.{suffix}"
    )


def holdout_chunk_path(dataset, repeat, suffix):
    return (
        f"{EVIDENCE_ROOT}/gate2/holdout/chunks/"
        f"{dataset}.repeat{repeat}.{suffix}"
    )


def dataset_arguments(records, mode, include_reference=False):
    values = []
    for record in records:
        fields = [
            record["id"],
            h5ad_path(mode, record),
            data_manifest_path(mode, record),
            pair_manifest_path(mode, record),
        ]
        if include_reference:
            fields.append(reference_path(mode, record))
        values.append("--dataset " + shlex.quote("\t".join(fields)))
    return " ".join(values)


def extraction_arguments(records):
    return " ".join(
        "--dataset " + shlex.quote(record["id"]) for record in records
    )


def truth_arguments(records):
    return " ".join(
        "--dataset "
        + shlex.quote(
            "\t".join(
                (
                    record["id"],
                    str(record["ground_truth_groups"]),
                    str(record["cells"]),
                    raw_path("development", record),
                    metadata_path("development", record),
                )
            )
        )
        for record in records
    )


def calibration_run_arguments():
    return " ".join(
        "--run " + shlex.quote(f"{summary}\t{parquet}")
        for summary, parquet in zip(
            CALIBRATION_SUMMARIES, CALIBRATION_PARQUETS, strict=True
        )
    )


def repeated_path_arguments(flag, paths):
    return " ".join(f"{flag} {shlex.quote(str(path))}" for path in paths)


def chunk_run_arguments(records, mode, profile=None):
    values = []
    for record in records:
        dataset = dataset_slug(record["id"])
        for repeat in GPU_REPEAT_INDICES:
            if mode == "development":
                summary = calibration_chunk_path(profile, dataset, repeat, "json")
                parquet = calibration_chunk_path(
                    profile, dataset, repeat, "parquet"
                )
            else:
                summary = holdout_chunk_path(dataset, repeat, "json")
                parquet = holdout_chunk_path(dataset, repeat, "parquet")
            values.append("--run " + shlex.quote(f"{summary}\t{parquet}"))
    return " ".join(values)


def expected_dataset_arguments(records):
    return " ".join(
        "--expected-dataset " + shlex.quote(record["id"]) for record in records
    )


def controlled_file_arguments():
    return " ".join(
        "--controlled-file " + shlex.quote(path)
        for path in RUNTIME_CONTROL_FILES
    )


def reference_variability_arguments():
    values = []
    for record in DEVELOPMENT:
        for first, second in REFERENCE_SEED_PAIRS:
            fields = (
                record["id"],
                str(first),
                reference_variability_path(record, first),
                str(second),
                reference_variability_path(record, second),
            )
            values.append("--run " + shlex.quote("\t".join(fields)))
    return " ".join(values)
rule all:
    input:
        GATE0_EVIDENCE,
        GATE1_STATIC_EVIDENCE,


rule development_ready:
    input:
        gate0=GATE0_EVIDENCE,
        gate1_static=GATE1_STATIC_EVIDENCE,
        gate1_gpu=GATE1_GPU_EVIDENCE,
        resource_canary=WORST_SHARD_CANARY,
        pairs=DEV_PAIR_MANIFESTS,
        references=DEV_REFERENCES,
        margins=DEV_MARGIN_CONTRACT,
        margin_eligibility=DEV_MARGIN_ELIGIBILITY,


rule gate2_development:
    input:
        DEV_PAIR_MANIFESTS,
        DEV_REFERENCES,
        DEV_MARGIN_CONTRACT,


rule candidate_requalification:
    input:
        scheduler_smoke=SCHEDULER_SMOKE,
        resource_canary=WORST_SHARD_CANARY,
        gate0=GATE0_EVIDENCE,
        gate1_static=GATE1_STATIC_EVIDENCE,
        gate1_gpu=GATE1_GPU_EVIDENCE,


rule gate2_calibration:
    input:
        CALIBRATION_PROFILE,


rule hierarchy_development:
    input:
        HIERARCHY_MARGIN_EVIDENCE,


rule final_candidate_reproduction:
    input:
        gate0=GATE0_EVIDENCE,
        gate1_static=GATE1_STATIC_EVIDENCE,
        gate1_gpu=GATE1_GPU_EVIDENCE,
        rf_profile=RF_PROFILE_REPRODUCTION,
        hierarchy_margins=HIERARCHY_MARGIN_REPRODUCTION,
        hierarchy_development=HIERARCHY_DEVELOPMENT_REPORTS,


rule gate2_holdout:
    input:
        GATE2_EVIDENCE,


rule comparator_confirm:
    input:
        GATE2_PASS_SEAL,


rule hierarchy_confirmation:
    input:
        HIERARCHY_CONFIRMATION_REPORTS,


rule parity_promotion:
    input:
        PARITY_PROMOTION_EVIDENCE,


rule r_logic_fixtures:
    input:
        script="validation/scripts/generate_logic_fixtures.R",
        lock="renv.lock",
        reference=REFERENCE_FILES,
    output:
        FIXTURE_FILES,
    log:
        f"{LOG_ROOT}/r_logic_fixtures.log",
    shell:
        "{config[rscript]} --vanilla {input.script} validation/fixtures/generated > {log} 2>&1"


rule fixture_manifest:
    input:
        script="validation/scripts/write_manifest.py",
        fixtures=FIXTURE_FILES,
    output:
        "validation/fixtures/generated/MANIFEST.json",
    log:
        f"{LOG_ROOT}/fixture_manifest.log",
    shell:
        "{config[python]} {input.script} --root . --output {output} {input.fixtures} > {log} 2>&1"


rule freeze_candidate_runtime:
    input:
        fixtures="validation/fixtures/generated/MANIFEST.json",
        runtime=RUNTIME_CONTROL_FILES,
        script="validation/scripts/freeze_candidate.py",
    output:
        CANDIDATE_MANIFEST,
    params:
        controlled=controlled_file_arguments(),
    log:
        f"{LOG_ROOT}/freeze_candidate.log",
    shell:
        "PYTHONPATH=validation/scripts {config[python]} {input.script} --root . "
        "--expected-commit {CANDIDATE_COMMIT:q} --attempt {CANDIDATE_ATTEMPT} "
        "{params.controlled} --output {output:q} > {log:q} 2>&1"


rule reserve_gate1_gpu_resources:
    input:
        candidate=CANDIDATE_MANIFEST,
        budget=RESOURCE_BUDGET_PATH,
        script="validation/scripts/resource_ledger.py",
    output:
        GATE1_RESERVATION,
    params:
        hours=RESOURCE_BUDGET["frozen_reservations"]["gate1_gpu"][
            "l40s_hours"
        ],
        scratch=RESOURCE_BUDGET["frozen_reservations"]["gate1_gpu"][
            "scratch_bytes"
        ],
    log:
        f"{LOG_ROOT}/reservations/gate1_gpu.log",
    shell:
        "{config[python]} {input.script} reserve --ledger {RESOURCE_LEDGER:q} "
        "--budget {input.budget:q} --phase gate1_gpu "
        "--candidate-commit {CANDIDATE_COMMIT:q} "
        "--candidate-attempt {CANDIDATE_ATTEMPT} "
        "--l40s-hours {params.hours} --scratch-bytes {params.scratch} "
        "--output {output:q} > {log:q} 2>&1"


rule reserve_comparator_development_resources:
    input:
        candidate=CANDIDATE_MANIFEST,
        gate1=GATE1_GPU_EVIDENCE,
        margin_eligibility=DEV_MARGIN_ELIGIBILITY,
        budget=RESOURCE_BUDGET_PATH,
        script="validation/scripts/resource_ledger.py",
    output:
        DEVELOPMENT_RESERVATION,
    params:
        hours=RESOURCE_BUDGET["frozen_reservations"][
            "comparator_development"
        ]["l40s_hours"],
        scratch=RESOURCE_BUDGET["frozen_reservations"][
            "comparator_development"
        ]["scratch_bytes"],
    log:
        f"{LOG_ROOT}/reservations/comparator_development.log",
    shell:
        "{config[python]} {input.script} reserve --ledger {RESOURCE_LEDGER:q} "
        "--budget {input.budget:q} --phase comparator_development "
        "--candidate-commit {CANDIDATE_COMMIT:q} "
        "--candidate-attempt {CANDIDATE_ATTEMPT} "
        "--l40s-hours {params.hours} --scratch-bytes {params.scratch} "
        "--output {output:q} > {log:q} 2>&1"


rule scheduler_smoke:
    input:
        candidate=CANDIDATE_MANIFEST,
        reservation=GATE1_RESERVATION,
        script="validation/scripts/scheduler_smoke.py",
    output:
        SCHEDULER_SMOKE,
    log:
        f"{LOG_ROOT}/smoke/scheduler.log",
    threads: 4
    resources:
        mem_mb=32000,
        runtime=30,
        slurm_partition="gpu-short",
        gres="gpu:L40S:1",
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} "
        "PYTHONPATH=src:. {config[gpu_python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--output {output:q} > {log:q} 2>&1"


rule worst_shard_resource_canary:
    input:
        scheduler_smoke=SCHEDULER_SMOKE,
        candidate=CANDIDATE_MANIFEST,
        reservation=GATE1_RESERVATION,
        script="validation/scripts/resource_canary.py",
    output:
        WORST_SHARD_CANARY,
    log:
        f"{LOG_ROOT}/smoke/worst_shard_canary.log",
    threads: config["gpu_chunk"]["cpus_per_task"]
    resources:
        mem_mb=config["gpu_chunk"]["mem_mb"],
        runtime=config["gpu_chunk"]["runtime_minutes"],
        slurm_partition=config["gpu_chunk"]["partition"],
        gres=config["gpu_chunk"]["gres"],
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} "
        "PYTHONPATH=src:validation/scripts {config[gpu_python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--cells 23810 --features 10000 --output {output:q} "
        "> {log:q} 2>&1"


rule reserve_comparator_confirm_resources:
    input:
        candidate=CANDIDATE_MANIFEST,
        authorization=current_comparator_authorization,
        budget=RESOURCE_BUDGET_PATH,
        script="validation/scripts/resource_ledger.py",
    output:
        CONFIRM_RESERVATION,
    params:
        hours=RESOURCE_BUDGET["frozen_reservations"]["comparator_confirm"][
            "l40s_hours"
        ],
        scratch=RESOURCE_BUDGET["frozen_reservations"][
            "comparator_confirm"
        ]["scratch_bytes"],
    log:
        f"{LOG_ROOT}/reservations/comparator_confirm.log",
    shell:
        "{config[python]} {input.script} reserve --ledger {RESOURCE_LEDGER:q} "
        "--budget {input.budget:q} --phase comparator_confirm "
        "--candidate-commit {CANDIDATE_COMMIT:q} "
        "--candidate-attempt {CANDIDATE_ATTEMPT} "
        "--l40s-hours {params.hours} --scratch-bytes {params.scratch} "
        "--output {output:q} > {log:q} 2>&1"


rule gate0:
    input:
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/check_gate0.py",
        manifest="validation/fixtures/generated/MANIFEST.json",
        fixtures=FIXTURE_FILES,
        reference=REFERENCE_FILES,
        lock="renv.lock",
        datasets="validation/fixtures/specs/datasets.yaml",
        audit="docs/validation/source-audit.md",
        parity_assurance="docs/validation/parity-assurance.md",
        architecture="docs/architecture/architecture.md",
        decisions="docs/decisions/README.md",
        license="LICENSE",
        notice="NOTICE",
        citation="CITATION.cff",
        assurance=ASSURANCE_CONTRACT_FILES,
    output:
        GATE0_EVIDENCE,
    log:
        f"{LOG_ROOT}/gate0.log",
    shell:
        "PYTHONPATH=validation/scripts {config[python]} {input.script} --root . "
        "--fixtures validation/fixtures/generated "
        "--candidate-manifest {input.candidate:q} --output {output:q} "
        "> {log:q} 2>&1"


rule gate1_static:
    input:
        gate0=GATE0_EVIDENCE,
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/run_gate1_static.py",
        package=STATIC_PACKAGE_FILES,
        assurance=ASSURANCE_CONTRACT_FILES,
        project="pyproject.toml",
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        GATE1_STATIC_EVIDENCE,
    log:
        f"{LOG_ROOT}/gate1_static.log",
    shell:
        "PYTHONPATH=src:. {config[gpu_python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--rscript {config[rscript]:q} "
        "--output {output:q} --log {log:q}"


rule require_parity_framework:
    input:
        candidate=CANDIDATE_MANIFEST,
        gate0=GATE0_EVIDENCE,
        gate1=GATE1_STATIC_EVIDENCE,
        gate1_gpu=GATE1_GPU_EVIDENCE,
        script="validation/scripts/require_parity_framework.py",
        assurance=ASSURANCE_CONTRACT_FILES,
    output:
        PARITY_FRAMEWORK_SEAL,
    log:
        f"{LOG_ROOT}/parity/require_framework.log",
    shell:
        "PYTHONPATH=validation/scripts {config[python]} {input.script} --root . "
        "--candidate-manifest {input.candidate:q} --gate0 {input.gate0:q} "
        "--gate1-static {input.gate1:q} --gate1-gpu {input.gate1_gpu:q} "
        "--output {output:q} > {log:q} 2>&1"


rule synthetic_comparator_fixture:
    input:
        script="validation/scripts/generate_comparator_fixture.R",
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        lock="renv.lock",
    output:
        "validation/fixtures/comparator_synthetic.json",
    log:
        f"{LOG_ROOT}/comparator_synthetic.log",
    shell:
        "{config[rscript]} --vanilla {input.script} {input.source:q} {output:q} 1 "
        "> {log:q} 2>&1"


rule gate1_gpu:
    input:
        static=GATE1_STATIC_EVIDENCE,
        candidate=CANDIDATE_MANIFEST,
        reservation=GATE1_RESERVATION,
        scheduler_smoke=SCHEDULER_SMOKE,
        resource_canary=WORST_SHARD_CANARY,
        synthetic="validation/fixtures/comparator_synthetic.json",
        script="validation/scripts/run_gate1_gpu.py",
        package=STATIC_PACKAGE_FILES,
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        GATE1_GPU_EVIDENCE,
    log:
        f"{LOG_ROOT}/gate1_gpu.log",
    threads: 4
    resources:
        mem_mb=32000,
        runtime=30,
        slurm_partition="gpu-short",
        gres="gpu:L40S:1",
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} "
        "PYTHONPATH=validation/scripts {config[gpu_python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--output {output:q} --log {log:q}"


rule download_dataset_archive:
    input:
        script="validation/scripts/download_zenodo.py",
    output:
        ARCHIVE,
    params:
        url=config["data"]["zenodo_archive_url"],
        expected_bytes=DATASET_SPEC["dataset_source"]["archive"]["bytes"],
    log:
        f"{LOG_ROOT}/gate2/download_archive.log",
    resources:
        mem_mb=2000,
        runtime=1440,
    shell:
        "{config[python]} {input.script} --url {params.url:q} --output {output:q} "
        "--expected-bytes {params.expected_bytes} > {log:q} 2>&1"


rule verify_dataset_archive:
    input:
        archive=ARCHIVE,
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/verify_dataset_archive.py",
    output:
        ARCHIVE_VERIFY,
    params:
        expected_bytes=DATASET_SPEC["dataset_source"]["archive"]["bytes"],
        expected_md5=DATASET_SPEC["dataset_source"]["archive"]["md5"],
    log:
        f"{LOG_ROOT}/gate2/verify_archive.log",
    resources:
        mem_mb=2000,
        runtime=240,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--expected-bytes {params.expected_bytes} --expected-md5 {params.expected_md5} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--output {output:q} > {log:q} 2>&1"


rule inventory_dataset_archive:
    input:
        archive=ARCHIVE,
        verification=ARCHIVE_VERIFY,
        script="validation/scripts/inventory_dataset_archive.py",
    output:
        ARCHIVE_INVENTORY,
    log:
        f"{LOG_ROOT}/gate2/inventory_archive.log",
    resources:
        mem_mb=4000,
        runtime=240,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--verification {input.verification:q} --output {output:q} "
        "> {log:q} 2>&1"


rule extract_development_datasets:
    input:
        archive=ARCHIVE,
        verification=ARCHIVE_VERIFY,
        inventory=ARCHIVE_INVENTORY,
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/extract_datasets.py",
    output:
        manifest=DEV_EXTRACT_MANIFEST,
        counts=DEV_RDS,
        metadata=DEV_METADATA,
    params:
        datasets=extraction_arguments(DEVELOPMENT),
        output_root=DEV_RAW_ROOT,
    log:
        f"{LOG_ROOT}/gate2/extract_development.log",
    resources:
        mem_mb=4000,
        runtime=480,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--verification {input.verification:q} --inventory {input.inventory:q} "
        "--mode development {params.datasets} --output-root {params.output_root:q} "
        "--manifest {output.manifest:q} --candidate-manifest {input.candidate:q} "
        "--root . > {log:q} 2>&1"


rule discover_development_truth_label:
    input:
        counts=DEV_RDS,
        metadata=DEV_METADATA,
        extraction=DEV_EXTRACT_MANIFEST,
        script="validation/scripts/discover_truth_label.R",
    output:
        TRUTH_CONTRACT,
    params:
        datasets=truth_arguments(DEVELOPMENT),
    log:
        f"{LOG_ROOT}/gate2/discover_truth_label.log",
    resources:
        mem_mb=64000,
        runtime=240,
    shell:
        "{config[rscript]} --vanilla {input.script} {params.datasets} "
        "--extraction-manifest {input.extraction:q} --output {output:q} "
        "> {log:q} 2>&1"


rule preprocess_development_dataset:
    input:
        counts=lambda wildcards: raw_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        metadata=lambda wildcards: metadata_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        truth=TRUTH_CONTRACT,
        script="validation/scripts/preprocess_seurat.R",
    output:
        f"{CANDIDATE_WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: DEV_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        f"{LOG_ROOT}/gate2/development/{{dataset}}.preprocess.log",
    resources:
        mem_mb=64000,
        runtime=240,
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --counts {input.counts:q} "
        "--metadata {input.metadata:q} "
        "--output {output:q} --truth-contract {input.truth:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} > {log:q} 2>&1"


rule verify_development_h5ad:
    input:
        h5ad=f"{CANDIDATE_WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/verify_h5ad_contract.py",
    output:
        f"{CANDIDATE_WORK_ROOT}/development/manifests/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: DEV_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        f"{LOG_ROOT}/gate2/development/{{dataset}}.verify_h5ad.log",
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[python]} {input.script} --input {input.h5ad:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} --root . "
        "--candidate-manifest {input.candidate:q} --output {output:q} "
        "> {log:q} 2>&1"


rule build_development_pairs:
    input:
        h5ad=f"{CANDIDATE_WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
        manifest=f"{CANDIDATE_WORK_ROOT}/development/manifests/{{dataset}}.json",
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/build_pair_manifest.py",
    output:
        f"{CANDIDATE_WORK_ROOT}/development/pairs/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
        seed=DATASET_SPEC["seeds"]["calibration_profiles"],
    log:
        f"{LOG_ROOT}/gate2/development/{{dataset}}.pairs.log",
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[python]} {input.script} --h5ad {input.h5ad:q} "
        "--data-manifest {input.manifest:q} --dataset-id {params.dataset_id:q} "
        "--seed {params.seed} --root . --candidate-manifest {input.candidate:q} "
        "--output {output:q} > {log:q} 2>&1"


rule run_development_r_reference:
    input:
        h5ad=f"{CANDIDATE_WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
        pairs=f"{CANDIDATE_WORK_ROOT}/development/pairs/{{dataset}}.json",
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        script="validation/scripts/run_reference_comparator.R",
    output:
        f"{CANDIDATE_WORK_ROOT}/development/reference/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
    log:
        f"{LOG_ROOT}/gate2/development/{{dataset}}.r_reference.log",
    resources:
        mem_mb=64000,
        runtime=1440,
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --h5ad {input.h5ad:q} "
        "--pairs {input.pairs:q} --dataset-id {params.dataset_id:q} "
        "--reference-source {input.source:q} --seed 1 --output {output:q} "
        "> {log:q} 2>&1"


rule run_development_r_reference_variability:
    input:
        h5ad=lambda wildcards: h5ad_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        pairs=lambda wildcards: pair_manifest_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        script="validation/scripts/run_reference_comparator.R",
        contract=COMPARATOR_CONTRACT_PATH,
    output:
        f"{CANDIDATE_WORK_ROOT}/development/reference_variability/"
        "{dataset}.seed{seed}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
    log:
        f"{LOG_ROOT}/gate2/development/reference_variability/"
        "{dataset}.seed{seed}.log",
    resources:
        mem_mb=64000,
        runtime=1440,
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
        seed="|".join(str(value) for value in REFERENCE_SEEDS),
    shell:
        "{config[rscript]} --vanilla {input.script} --h5ad {input.h5ad:q} "
        "--pairs {input.pairs:q} --dataset-id {params.dataset_id:q} "
        "--reference-source {input.source:q} --seed {wildcards.seed} "
        "--output {output:q} > {log:q} 2>&1"


rule derive_comparator_fidelity_margins:
    input:
        candidate=CANDIDATE_MANIFEST,
        references=DEV_REFERENCE_VARIABILITY,
        contract=COMPARATOR_CONTRACT_PATH,
        script="validation/scripts/derive_comparator_margins.py",
    output:
        DEV_MARGIN_CONTRACT,
    params:
        runs=reference_variability_arguments(),
    log:
        f"{LOG_ROOT}/gate2/development/derive_comparator_margins.log",
    resources:
        mem_mb=4000,
        runtime=60,
    shell:
        "PYTHONPATH=src {config[python]} {input.script} --root . "
        "--contract {input.contract:q} {params.runs} "
        "--candidate-manifest {input.candidate:q} --output {output:q} "
        "> {log:q} 2>&1"


rule require_eligible_comparator_margins:
    input:
        margins=DEV_MARGIN_CONTRACT,
        script="validation/scripts/require_status.py",
    output:
        DEV_MARGIN_ELIGIBILITY,
    log:
        f"{LOG_ROOT}/gate2/development/require_eligible_margins.log",
    shell:
        "{config[python]} {input.script} --input {input.margins:q} "
        "--expected-status FROZEN --scope comparator_reference_variability "
        "--output {output:q} > {log:q} 2>&1"


rule run_development_gpu_profile_chunk:
    input:
        gate1=GATE1_GPU_EVIDENCE,
        candidate=CANDIDATE_MANIFEST,
        reservation=DEVELOPMENT_RESERVATION,
        resource_canary=WORST_SHARD_CANARY,
        margin_eligibility=DEV_MARGIN_ELIGIBILITY,
        h5ad=lambda wildcards: h5ad_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        manifest=lambda wildcards: data_manifest_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        pairs=lambda wildcards: pair_manifest_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        reference=lambda wildcards: reference_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        contract=COMPARATOR_CONTRACT_PATH,
        script="validation/scripts/run_gpu_comparisons.py",
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        summary=(
            f"{EVIDENCE_ROOT}/gate2/calibration/chunks/"
            "{profile}/{dataset}.repeat{repeat}.json"
        ),
        parquet=(
            f"{EVIDENCE_ROOT}/gate2/calibration/chunks/"
            "{profile}/{dataset}.repeat{repeat}.parquet"
        ),
    params:
        dataset=lambda wildcards: dataset_arguments(
            [DEV_BY_SLUG[wildcards.dataset]], "development", True
        ),
        n_bins=lambda wildcards: RF_BY_ID[wildcards.profile]["n_bins"],
        max_depth=lambda wildcards: RF_BY_ID[wildcards.profile]["max_depth"],
        n_streams=lambda wildcards: RF_BY_ID[wildcards.profile]["n_streams"],
    log:
        (
            f"{LOG_ROOT}/gate2/calibration/chunks/"
            "{profile}/{dataset}.repeat{repeat}.log"
        ),
    threads: config["gpu_chunk"]["cpus_per_task"]
    resources:
        mem_mb=config["gpu_chunk"]["mem_mb"],
        runtime=config["gpu_chunk"]["runtime_minutes"],
        slurm_partition=config["gpu_chunk"]["partition"],
        gres=config["gpu_chunk"]["gres"],
    wildcard_constraints:
        profile="|".join(RF_BY_ID),
        dataset="|".join(DEV_BY_SLUG),
        repeat="[1-5]",
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} {config[gpu_python]} {input.script} "
        "--root . {params.dataset} --n-bins {params.n_bins} "
        "--max-depth {params.max_depth} --n-streams {params.n_streams} "
        "--comparator-contract {input.contract:q} "
        "--candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--repeat-index {wildcards.repeat} --output-parquet {output.parquet:q} "
        "--output-summary {output.summary:q} > {log:q} 2>&1"


rule aggregate_development_gpu_profile:
    input:
        summaries=lambda wildcards: [
            calibration_chunk_path(
                wildcards.profile, dataset_slug(record["id"]), repeat, "json"
            )
            for record in DEVELOPMENT
            for repeat in GPU_REPEAT_INDICES
        ],
        parquets=lambda wildcards: [
            calibration_chunk_path(
                wildcards.profile, dataset_slug(record["id"]), repeat, "parquet"
            )
            for record in DEVELOPMENT
            for repeat in GPU_REPEAT_INDICES
        ],
        script="validation/scripts/aggregate_gpu_comparisons.py",
    output:
        summary=f"{EVIDENCE_ROOT}/gate2/calibration/{{profile}}.json",
        parquet=f"{EVIDENCE_ROOT}/gate2/calibration/{{profile}}.parquet",
    params:
        runs=lambda wildcards: chunk_run_arguments(
            DEVELOPMENT, "development", wildcards.profile
        ),
        datasets=expected_dataset_arguments(DEVELOPMENT),
    log:
        f"{LOG_ROOT}/gate2/calibration/{{profile}}.aggregate.log",
    resources:
        mem_mb=8000,
        runtime=60,
    wildcard_constraints:
        profile="|".join(RF_BY_ID),
    shell:
        "{config[python]} {input.script} --root . {params.runs} "
        "{params.datasets} --expected-repeats 5 "
        "--output-parquet {output.parquet:q} "
        "--output-summary {output.summary:q} > {log:q} 2>&1"


rule freeze_development_rf_profile:
    input:
        summaries=CALIBRATION_SUMMARIES,
        parquets=CALIBRATION_PARQUETS,
        margins=DEV_MARGIN_CONTRACT,
        contract=COMPARATOR_CONTRACT_PATH,
        script="validation/scripts/freeze_rf_profile.py",
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        profile=CALIBRATION_PROFILE,
        aggregate=f"{EVIDENCE_ROOT}/gate2/calibration_results.parquet",
    params:
        runs=calibration_run_arguments(),
    log:
        f"{LOG_ROOT}/gate2/freeze_rf_profile.log",
    shell:
        "PYTHONPATH=src {config[python]} {input.script} --root . {params.runs} "
        "--margin-contract {input.margins:q} --comparator-contract {input.contract:q} "
        "--conda-lock {input.conda_lock:q} --pip-lock {input.pip_lock:q} "
        "--output {output.profile:q} --aggregate-parquet {output.aggregate:q} "
        "> {log:q} 2>&1"


rule verify_frozen_rf_profile_reproduction:
    input:
        frozen=RF_PROFILE,
        reproduced=CALIBRATION_PROFILE,
        script="validation/scripts/verify_frozen_rf_profile.py",
    output:
        RF_PROFILE_REPRODUCTION,
    log:
        f"{LOG_ROOT}/gate2/verify_rf_profile_reproduction.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} "
        "--frozen {input.frozen:q} --reproduced {input.reproduced:q} "
        "--output {output:q} > {log:q} 2>&1"


rule seal_committed_rf_profile:
    input:
        profile=RF_PROFILE,
        candidate=CANDIDATE_MANIFEST,
        script="validation/scripts/check_profile_seal.py",
    output:
        PROFILE_SEAL,
    log:
        f"{LOG_ROOT}/gate2/seal_rf_profile.log",
    shell:
        "{config[python]} {input.script} --root . --profile {input.profile:q} "
        "--candidate-manifest {input.candidate:q} "
        "--output {output:q} > {log:q} 2>&1"


rule authorize_comparator_confirm:
    input:
        candidate=CANDIDATE_MANIFEST,
        parity_framework=PARITY_FRAMEWORK_SEAL,
        profile=RF_PROFILE,
        profile_reproduction=RF_PROFILE_REPRODUCTION,
        manifest=COMPARATOR_AUTHORIZATION["manifest"],
        signature=COMPARATOR_AUTHORIZATION["signature"],
        allowed_signers=AUTHORIZATION_CONFIG["allowed_signers"],
        script="validation/scripts/check_authorization.py",
        runtime=RUNTIME_CONTROL_FILES,
    output:
        COMPARATOR_AUTHORIZATION_SEAL,
    log:
        f"{LOG_ROOT}/authorization/comparator_confirm.log",
    shell:
        "{config[python]} {input.script} --root . "
        "--manifest {input.manifest:q} --signature {input.signature:q} "
        "--allowed-signers {input.allowed_signers:q} "
        "--expected-target comparator_confirm "
        "--expected-operator {AUTHORIZATION_CONFIG[operator]:q} "
        "--expected-attempt {CANDIDATE_ATTEMPT} "
        "--expected-profile {input.profile:q} "
        "--expected-profile-reproduction {input.profile_reproduction:q} "
        "--expected-role comparator_holdout --output {output:q} "
        "> {log:q} 2>&1"


rule extract_holdout_datasets:
    input:
        archive=ARCHIVE,
        verification=ARCHIVE_VERIFY,
        inventory=ARCHIVE_INVENTORY,
        candidate=CANDIDATE_MANIFEST,
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        parity_framework=PARITY_FRAMEWORK_SEAL,
        script="validation/scripts/extract_datasets.py",
    output:
        manifest=HOLD_EXTRACT_MANIFEST,
        counts=HOLD_RDS,
        metadata=HOLD_METADATA,
    params:
        datasets=extraction_arguments(HOLDOUT),
        output_root=HOLD_RAW_ROOT,
    log:
        f"{LOG_ROOT}/gate2/extract_holdout.log",
    resources:
        mem_mb=4000,
        runtime=480,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--verification {input.verification:q} --inventory {input.inventory:q} "
        "--mode holdout {params.datasets} --output-root {params.output_root:q} "
        "--manifest {output.manifest:q} --profile-seal {input.seal:q} "
        "--authorization-seal {input.authorization:q} "
        "--candidate-manifest {input.candidate:q} --root . > {log:q} 2>&1"


rule preprocess_holdout_dataset:
    input:
        counts=lambda wildcards: raw_path(
            "holdout", HOLD_BY_SLUG[wildcards.dataset]
        ),
        metadata=lambda wildcards: metadata_path(
            "holdout", HOLD_BY_SLUG[wildcards.dataset]
        ),
        truth=TRUTH_CONTRACT,
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        script="validation/scripts/preprocess_seurat.R",
    output:
        f"{CANDIDATE_WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        f"{LOG_ROOT}/gate2/holdout/{{dataset}}.preprocess.log",
    resources:
        mem_mb=64000,
        runtime=240,
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --counts {input.counts:q} "
        "--metadata {input.metadata:q} "
        "--output {output:q} --truth-contract {input.truth:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} > {log:q} 2>&1"


rule verify_holdout_h5ad:
    input:
        h5ad=f"{CANDIDATE_WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
        candidate=CANDIDATE_MANIFEST,
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        script="validation/scripts/verify_h5ad_contract.py",
    output:
        f"{CANDIDATE_WORK_ROOT}/holdout/manifests/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        f"{LOG_ROOT}/gate2/holdout/{{dataset}}.verify_h5ad.log",
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[python]} {input.script} --input {input.h5ad:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} --root . "
        "--candidate-manifest {input.candidate:q} --output {output:q} "
        "> {log:q} 2>&1"


rule build_holdout_pairs:
    input:
        h5ad=f"{CANDIDATE_WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
        manifest=f"{CANDIDATE_WORK_ROOT}/holdout/manifests/{{dataset}}.json",
        candidate=CANDIDATE_MANIFEST,
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        script="validation/scripts/build_pair_manifest.py",
    output:
        f"{CANDIDATE_WORK_ROOT}/holdout/pairs/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
        seed=DATASET_SPEC["seeds"]["calibration_profiles"],
    log:
        f"{LOG_ROOT}/gate2/holdout/{{dataset}}.pairs.log",
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[python]} {input.script} --h5ad {input.h5ad:q} "
        "--data-manifest {input.manifest:q} --dataset-id {params.dataset_id:q} "
        "--seed {params.seed} --root . --candidate-manifest {input.candidate:q} "
        "--output {output:q} > {log:q} 2>&1"


rule run_holdout_r_reference:
    input:
        h5ad=f"{CANDIDATE_WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
        pairs=f"{CANDIDATE_WORK_ROOT}/holdout/pairs/{{dataset}}.json",
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        script="validation/scripts/run_reference_comparator.R",
    output:
        f"{CANDIDATE_WORK_ROOT}/holdout/reference/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
    log:
        f"{LOG_ROOT}/gate2/holdout/{{dataset}}.r_reference.log",
    resources:
        mem_mb=64000,
        runtime=1440,
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --h5ad {input.h5ad:q} "
        "--pairs {input.pairs:q} --dataset-id {params.dataset_id:q} "
        "--reference-source {input.source:q} --seed 1 --output {output:q} "
        "> {log:q} 2>&1"


rule run_holdout_gpu_profile_chunk:
    input:
        profile=RF_PROFILE,
        candidate=CANDIDATE_MANIFEST,
        reservation=CONFIRM_RESERVATION,
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        parity_framework=PARITY_FRAMEWORK_SEAL,
        h5ad=lambda wildcards: h5ad_path(
            "holdout", HOLD_BY_SLUG[wildcards.dataset]
        ),
        manifest=lambda wildcards: data_manifest_path(
            "holdout", HOLD_BY_SLUG[wildcards.dataset]
        ),
        pairs=lambda wildcards: pair_manifest_path(
            "holdout", HOLD_BY_SLUG[wildcards.dataset]
        ),
        reference=lambda wildcards: reference_path(
            "holdout", HOLD_BY_SLUG[wildcards.dataset]
        ),
        contract=COMPARATOR_CONTRACT_PATH,
        script="validation/scripts/run_gpu_comparisons.py",
    output:
        summary=(
            f"{EVIDENCE_ROOT}/gate2/holdout/chunks/"
            "{dataset}.repeat{repeat}.json"
        ),
        parquet=(
            f"{EVIDENCE_ROOT}/gate2/holdout/chunks/"
            "{dataset}.repeat{repeat}.parquet"
        ),
    params:
        dataset=lambda wildcards: dataset_arguments(
            [HOLD_BY_SLUG[wildcards.dataset]], "holdout", True
        ),
    log:
        f"{LOG_ROOT}/gate2/holdout/chunks/"
        "{dataset}.repeat{repeat}.log",
    threads: config["gpu_chunk"]["cpus_per_task"]
    resources:
        mem_mb=config["gpu_chunk"]["mem_mb"],
        runtime=config["gpu_chunk"]["runtime_minutes"],
        slurm_partition=config["gpu_chunk"]["partition"],
        gres=config["gpu_chunk"]["gres"],
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
        repeat="[1-5]",
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} {config[gpu_python]} {input.script} "
        "--root . {params.dataset} --profile-json {input.profile:q} "
        "--comparator-contract {input.contract:q} "
        "--authorization-seal {input.authorization:q} "
        "--candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--repeat-index {wildcards.repeat} "
        "--output-parquet {output.parquet:q} --output-summary {output.summary:q} "
        "> {log:q} 2>&1"


rule aggregate_holdout_gpu_profile:
    input:
        authorization=current_comparator_authorization,
        summaries=[
            holdout_chunk_path(dataset_slug(record["id"]), repeat, "json")
            for record in HOLDOUT
            for repeat in GPU_REPEAT_INDICES
        ],
        parquets=[
            holdout_chunk_path(dataset_slug(record["id"]), repeat, "parquet")
            for record in HOLDOUT
            for repeat in GPU_REPEAT_INDICES
        ],
        script="validation/scripts/aggregate_gpu_comparisons.py",
    output:
        summary=f"{EVIDENCE_ROOT}/gate2/holdout_repeats.json",
        parquet=f"{EVIDENCE_ROOT}/gate2/holdout_repeats.parquet",
    params:
        runs=chunk_run_arguments(HOLDOUT, "holdout"),
        datasets=expected_dataset_arguments(HOLDOUT),
    log:
        f"{LOG_ROOT}/gate2/holdout_gpu.aggregate.log",
    resources:
        mem_mb=8000,
        runtime=60,
    shell:
        "{config[python]} {input.script} --root . {params.runs} "
        "{params.datasets} --expected-repeats 5 "
        "--output-parquet {output.parquet:q} "
        "--output-summary {output.summary:q} > {log:q} 2>&1"


rule assemble_gate2_acceptance:
    input:
        profile=RF_PROFILE,
        profile_reproduction=RF_PROFILE_REPRODUCTION,
        seal=PROFILE_SEAL,
        authorization=current_comparator_authorization,
        summary=f"{EVIDENCE_ROOT}/gate2/holdout_repeats.json",
        parquet=f"{EVIDENCE_ROOT}/gate2/holdout_repeats.parquet",
        spec=DATASET_SPEC_PATH,
        contract=COMPARATOR_CONTRACT_PATH,
        script="validation/scripts/assemble_gate2.py",
    output:
        evidence=GATE2_EVIDENCE,
        aggregate=f"{EVIDENCE_ROOT}/gate2/holdout_results.parquet",
    log:
        f"{LOG_ROOT}/gate2/assemble_gate2.log",
    shell:
        "PYTHONPATH=src {config[python]} {input.script} --root . "
        "--dataset-spec {input.spec:q} --profile {input.profile:q} "
        "--profile-reproduction {input.profile_reproduction:q} "
        "--comparator-contract {input.contract:q} "
        "--authorization-seal {input.authorization:q} "
        "--profile-seal {input.seal:q} --run-summary {input.summary:q} "
        "--run-parquet {input.parquet:q} --aggregate-parquet {output.aggregate:q} "
        "--output {output.evidence:q} > {log:q} 2>&1"


rule require_gate2_pass:
    input:
        evidence=GATE2_EVIDENCE,
        script="validation/scripts/require_status.py",
    output:
        GATE2_PASS_SEAL,
    log:
        f"{LOG_ROOT}/gate2/require_pass.log",
    shell:
        "{config[python]} {input.script} --input {input.evidence:q} "
        "--expected-status PASS --scope comparator_confirm "
        "--output {output:q} > {log:q} 2>&1"


rule reserve_hierarchy_development_resources:
    input:
        candidate=CANDIDATE_MANIFEST,
        gate1=GATE1_GPU_EVIDENCE,
        profile=CALIBRATION_PROFILE,
        budget=RESOURCE_BUDGET_PATH,
        script="validation/scripts/resource_ledger.py",
    output:
        HIERARCHY_DEVELOPMENT_RESERVATION,
    params:
        hours=RESOURCE_BUDGET["frozen_reservations"][
            "hierarchy_development"
        ]["l40s_hours"],
        scratch=RESOURCE_BUDGET["frozen_reservations"][
            "hierarchy_development"
        ]["scratch_bytes"],
    log:
        f"{LOG_ROOT}/reservations/hierarchy_development.log",
    shell:
        "{config[python]} {input.script} reserve --ledger {RESOURCE_LEDGER:q} "
        "--budget {input.budget:q} --phase hierarchy_development "
        "--candidate-commit {CANDIDATE_COMMIT:q} "
        "--candidate-attempt {CANDIDATE_ATTEMPT} "
        "--l40s-hours {params.hours} --scratch-bytes {params.scratch} "
        "--output {output:q} > {log:q} 2>&1"


rule generate_hierarchy_development_object:
    input:
        contract=HIERARCHY_DEVELOPMENT_CONTRACT_PATH,
        script="validation/scripts/generate_hierarchy_development_object.R",
        wrapper="validation/scripts/run_preserving_command.py",
        lock="renv.lock",
    output:
        recipe=hierarchy_development_path("{dataset}", "recipe.rds"),
        h5ad=hierarchy_development_path("{dataset}", "h5ad"),
        manifest=hierarchy_development_path("{dataset}", "manifest.json"),
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.generate.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
    resources:
        mem_mb=16000,
        runtime=120,
    shell:
        "PYTHONPATH=. {config[python]} {input.wrapper} --log {log:q} "
        "--failure-anchor {output.manifest:q} "
        "--scope hierarchy_development_object --stage generation -- "
        "{config[rscript]} --vanilla {input.script} "
        "--contract {input.contract:q} --dataset-id {wildcards.dataset:q} "
        "--output-recipe {output.recipe:q} --output-h5ad {output.h5ad:q} "
        "--output-manifest {output.manifest:q}"


rule write_hierarchy_development_config:
    input:
        contract=HIERARCHY_DEVELOPMENT_CONTRACT_PATH,
        manifest=hierarchy_development_path("{dataset}", "manifest.json"),
        profile=CALIBRATION_PROFILE,
        script="validation/scripts/write_hierarchy_config.py",
    output:
        hierarchy_development_path("{dataset}", "config.json"),
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.config.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--object-contract {input.contract:q} --dataset-id {wildcards.dataset:q} "
        "--rf-profile {input.profile:q} --object-manifest {input.manifest:q} "
        "--output {output:q} > {log:q} 2>&1"


rule run_hierarchy_development_python:
    input:
        candidate=CANDIDATE_MANIFEST,
        reservation=HIERARCHY_DEVELOPMENT_RESERVATION,
        h5ad=hierarchy_development_path("{dataset}", "h5ad"),
        manifest=hierarchy_development_path("{dataset}", "manifest.json"),
        config=hierarchy_development_path("{dataset}", "config.json"),
        script="validation/scripts/run_python_semantic_trace.py",
    output:
        trace=hierarchy_python_trace_path("{dataset}"),
        result=hierarchy_python_result_path("{dataset}"),
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.python.log",
    threads: 4
    resources:
        mem_mb=32000,
        runtime=120,
        slurm_partition="gpu-short",
        gres="gpu:L40S:1",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} "
        "PYTHONPATH=src:validation/scripts:. {config[gpu_python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--resource-phase hierarchy_development --h5ad {input.h5ad:q} "
        "--config {input.config:q} --dataset-id {wildcards.dataset:q} "
        "--output-jsonl {output.trace:q} --output-result {output.result:q} "
        "> {log:q} 2>&1"


rule run_hierarchy_development_r_reference:
    input:
        recipe=hierarchy_development_path("{dataset}", "recipe.rds"),
        python_result=hierarchy_python_result_path("{dataset}"),
        reference=REFERENCE_FILES,
        script="validation/scripts/run_reference_hierarchy.R",
        wrapper="validation/scripts/run_preserving_command.py",
        lock="renv.lock",
    output:
        bundle=hierarchy_r_path("{dataset}", "{seed}", "bundle.rds"),
        equivalence=hierarchy_r_path(
            "{dataset}", "{seed}", "instrumentation.json"
        ),
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.r.seed{{seed}}.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
        seed="104729|104759",
    resources:
        mem_mb=32000,
        runtime=1440,
    shell:
        "PYTHONPATH=. {config[python]} {input.wrapper} --log {log:q} "
        "--failure-anchor {output.equivalence:q} "
        "--scope hierarchy_reference --stage instrumented_execution -- "
        "{config[rscript]} --vanilla {input.script} "
        "--reference-root {REFERENCE_SOURCE_ROOT:q} --recipe {input.recipe:q} "
        "--python-result {input.python_result:q} --seed {wildcards.seed} "
        "--bundle {output.bundle:q} --equivalence {output.equivalence:q}"


rule canonicalize_hierarchy_development_r_reference:
    input:
        candidate=CANDIDATE_MANIFEST,
        bundle=hierarchy_r_path("{dataset}", "{seed}", "bundle.rds"),
        equivalence=hierarchy_r_path(
            "{dataset}", "{seed}", "instrumentation.json"
        ),
        script="validation/scripts/run_reference_semantic_trace.R",
        wrapper="validation/scripts/run_preserving_command.py",
    output:
        trace=hierarchy_r_path("{dataset}", "{seed}", "trace.jsonl"),
        result=hierarchy_r_path("{dataset}", "{seed}", "result.json"),
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.r.seed{{seed}}.canonical.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
        seed="104729|104759",
    resources:
        mem_mb=8000,
        runtime=60,
    shell:
        "PYTHONPATH=. {config[python]} {input.wrapper} --log {log:q} "
        "--failure-anchor {output.result:q} "
        "--scope hierarchy_reference --stage canonicalization -- "
        "{config[rscript]} --vanilla {input.script} --bundle {input.bundle:q} "
        "--candidate-manifest {input.candidate:q} --output {output.trace:q} "
        "--result {output.result:q}"


rule calibrate_hierarchy_development_pair:
    input:
        first=hierarchy_r_path("{dataset}", 104729, "result.json"),
        second=hierarchy_r_path("{dataset}", 104759, "result.json"),
        script="validation/scripts/calibrate_hierarchy_pair.py",
    output:
        f"{EVIDENCE_ROOT}/hierarchy/calibration/{{dataset}}.json",
    params:
        role=lambda wildcards: HIERARCHY_ROLE_BY_ID[wildcards.dataset],
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.calibration.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--first {input.first:q} --second {input.second:q} "
        "--role {params.role:q} --repeat-pair seed104729-vs-seed104759 "
        "--output {output:q} > {log:q} 2>&1"


rule derive_hierarchy_fidelity_margins:
    input:
        candidate=CANDIDATE_MANIFEST,
        design=HIERARCHY_DESIGN_PATH,
        calibration=HIERARCHY_CALIBRATION_REPORTS,
        script="validation/scripts/derive_hierarchy_margins.py",
    output:
        HIERARCHY_MARGIN_EVIDENCE,
    params:
        calibration=lambda _wildcards, input: repeated_path_arguments(
            "--calibration", input.calibration
        ),
    log:
        f"{LOG_ROOT}/hierarchy/derive_margins.log",
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--contract {input.design:q} {params.calibration} "
        "--output {output:q} > {log:q} 2>&1"


rule verify_hierarchy_margin_reproduction:
    input:
        frozen=HIERARCHY_FROZEN_MARGIN_CONTRACT,
        reproduced=HIERARCHY_MARGIN_EVIDENCE,
        script="validation/scripts/verify_frozen_hierarchy_margins.py",
    output:
        HIERARCHY_MARGIN_REPRODUCTION,
    log:
        f"{LOG_ROOT}/hierarchy/verify_margin_reproduction.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} "
        "--frozen {input.frozen:q} --reproduced {input.reproduced:q} "
        "--output {output:q} > {log:q} 2>&1"


rule evaluate_hierarchy_development:
    input:
        reference=hierarchy_r_path("{dataset}", 104729, "result.json"),
        candidate=hierarchy_python_result_path("{dataset}"),
        margins=HIERARCHY_MARGIN_EVIDENCE,
        reproduction=HIERARCHY_MARGIN_REPRODUCTION,
        role_contract=HIERARCHY_DEVELOPMENT_CONTRACT_PATH,
        script="validation/scripts/evaluate_hierarchy_fidelity.py",
    output:
        f"{EVIDENCE_ROOT}/hierarchy/development/{{dataset}}.json",
    params:
        role=lambda wildcards: HIERARCHY_ROLE_BY_ID[wildcards.dataset],
    log:
        f"{LOG_ROOT}/hierarchy/development/{{dataset}}.evaluate.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_DEVELOPMENT_IDS),
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--reference {input.reference:q} --candidate {input.candidate:q} "
        "--margins {input.margins:q} --role-contract {input.role_contract:q} "
        "--expected-role {params.role:q} --output {output:q} "
        "> {log:q} 2>&1"


rule authorize_hierarchy_confirmation:
    input:
        candidate=CANDIDATE_MANIFEST,
        parity_framework=PARITY_FRAMEWORK_SEAL,
        comparator=GATE2_PASS_SEAL,
        profile_seal=PROFILE_SEAL,
        margins=HIERARCHY_MARGIN_EVIDENCE,
        margin_reproduction=HIERARCHY_MARGIN_REPRODUCTION,
        data_roles=PUBLICATION_DATA_ROLES_PATH,
        profile=RF_PROFILE,
        manifest=lambda wildcards: hierarchy_authorization_config(
            wildcards.dataset, "manifest"
        ),
        signature=lambda wildcards: hierarchy_authorization_config(
            wildcards.dataset, "signature"
        ),
        allowed_signers=AUTHORIZATION_CONFIG["allowed_signers"],
        script="validation/scripts/check_authorization.py",
        runtime=RUNTIME_CONTROL_FILES,
    output:
        hierarchy_confirmation_work_path(
            "{dataset}", "authorization.json"
        ),
    params:
        target=lambda wildcards: f"hierarchy_confirm_{wildcards.dataset}",
        role=lambda wildcards: (
            f"hierarchy_parity_confirmation:{wildcards.dataset}"
        ),
        bindings=lambda _wildcards, input: " ".join(
            (
                "--expected-bound-artifact "
                + shlex.quote(f"hierarchy_margins={input.margins}"),
                "--expected-bound-artifact "
                + shlex.quote(f"data_roles={input.data_roles}"),
                "--expected-bound-artifact "
                + shlex.quote(f"rf_profile={input.profile}"),
                "--required-bound-role h5ad",
                "--required-bound-role recipe",
                "--required-bound-role python_config",
                "--required-bound-role custody_record",
            )
        ),
    log:
        f"{LOG_ROOT}/authorization/hierarchy_{{dataset}}.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    shell:
        "PYTHONPATH=validation/scripts {config[python]} {input.script} --root . "
        "--manifest {input.manifest:q} --signature {input.signature:q} "
        "--allowed-signers {input.allowed_signers:q} "
        "--expected-target {params.target:q} "
        "--expected-operator {AUTHORIZATION_CONFIG[operator]:q} "
        "--expected-attempt {CANDIDATE_ATTEMPT} "
        "--expected-role {params.role:q} --expected-profile {input.profile:q} "
        "{params.bindings} "
        "--output {output:q} > {log:q} 2>&1"


rule seal_hierarchy_confirmation_dataset:
    input:
        candidate=CANDIDATE_MANIFEST,
        authorization=current_hierarchy_authorization,
        margins=HIERARCHY_MARGIN_EVIDENCE,
        data_roles=PUBLICATION_DATA_ROLES_PATH,
        profile=RF_PROFILE,
        source_h5ad=lambda wildcards: hierarchy_confirmation_config(
            wildcards.dataset, "h5ad"
        ),
        source_recipe=lambda wildcards: hierarchy_confirmation_config(
            wildcards.dataset, "recipe"
        ),
        source_config=lambda wildcards: hierarchy_confirmation_config(
            wildcards.dataset, "python_config"
        ),
        source_custody=lambda wildcards: hierarchy_confirmation_config(
            wildcards.dataset, "custody_record"
        ),
        script="validation/scripts/seal_hierarchy_confirmation.py",
    output:
        h5ad=hierarchy_confirmation_work_path("{dataset}", "sealed.h5ad"),
        recipe=hierarchy_confirmation_work_path(
            "{dataset}", "sealed.recipe.rds"
        ),
        config=hierarchy_confirmation_work_path(
            "{dataset}", "sealed.config.json"
        ),
        custody=hierarchy_confirmation_work_path(
            "{dataset}", "sealed.custody.json"
        ),
        manifest=hierarchy_confirmation_work_path(
            "{dataset}", "opening.json"
        ),
    log:
        f"{LOG_ROOT}/hierarchy/confirmation/{{dataset}}.seal.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    shell:
        "PYTHONPATH=. {config[python]} {input.script} --root . "
        "--candidate-manifest {input.candidate:q} "
        "--authorization {input.authorization:q} --margins {input.margins:q} "
        "--data-roles {input.data_roles:q} --rf-profile {input.profile:q} "
        "--dataset-id {wildcards.dataset:q} "
        "--source-h5ad {input.source_h5ad:q} "
        "--source-recipe {input.source_recipe:q} "
        "--source-python-config {input.source_config:q} "
        "--source-custody-record {input.source_custody:q} "
        "--sealed-h5ad {output.h5ad:q} --sealed-recipe {output.recipe:q} "
        "--sealed-python-config {output.config:q} "
        "--sealed-custody-record {output.custody:q} "
        "--output {output.manifest:q} > {log:q} 2>&1"


rule reserve_hierarchy_confirmation_resources:
    input:
        candidate=CANDIDATE_MANIFEST,
        authorization=current_hierarchy_authorization,
        opening=hierarchy_confirmation_work_path("{dataset}", "opening.json"),
        budget=RESOURCE_BUDGET_PATH,
        script="validation/scripts/resource_ledger.py",
    output:
        hierarchy_confirmation_reservation_path("{dataset}"),
    params:
        phase=lambda wildcards: f"hierarchy_confirm_{wildcards.dataset}",
        hours=(
            RESOURCE_BUDGET["frozen_reservations"]["hierarchy_confirm"]
            ["l40s_hours"] / len(HIERARCHY_CONFIRMATION_IDS)
        ),
        scratch=int(
            RESOURCE_BUDGET["frozen_reservations"]["hierarchy_confirm"]
            ["scratch_bytes"] / len(HIERARCHY_CONFIRMATION_IDS)
        ),
    log:
        f"{LOG_ROOT}/reservations/hierarchy_confirm_{{dataset}}.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    shell:
        "{config[python]} {input.script} reserve --ledger {RESOURCE_LEDGER:q} "
        "--budget {input.budget:q} --phase {params.phase:q} "
        "--candidate-commit {CANDIDATE_COMMIT:q} "
        "--candidate-attempt {CANDIDATE_ATTEMPT} "
        "--l40s-hours {params.hours} --scratch-bytes {params.scratch} "
        "--output {output:q} > {log:q} 2>&1"


rule run_hierarchy_confirmation_python:
    input:
        candidate=CANDIDATE_MANIFEST,
        authorization=current_hierarchy_authorization,
        opening=hierarchy_confirmation_work_path("{dataset}", "opening.json"),
        reservation=hierarchy_confirmation_reservation_path("{dataset}"),
        h5ad=hierarchy_confirmation_work_path("{dataset}", "sealed.h5ad"),
        config=hierarchy_confirmation_work_path(
            "{dataset}", "sealed.config.json"
        ),
        script="validation/scripts/run_python_semantic_trace.py",
    output:
        trace=hierarchy_confirmation_python_path("{dataset}", "trace.jsonl"),
        result=hierarchy_confirmation_python_path("{dataset}", "result.json"),
    params:
        phase=lambda wildcards: f"hierarchy_confirm_{wildcards.dataset}",
    log:
        f"{LOG_ROOT}/hierarchy/confirmation/{{dataset}}.python.log",
    threads: 4
    resources:
        mem_mb=32000,
        runtime=240,
        slurm_partition="gpu-short",
        gres="gpu:L40S:1",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} "
        "PYTHONPATH=src:validation/scripts:. {config[gpu_python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--resource-reservation {input.reservation:q} "
        "--resource-ledger {RESOURCE_LEDGER:q} "
        "--resource-phase {params.phase:q} --h5ad {input.h5ad:q} "
        "--config {input.config:q} --dataset-id {wildcards.dataset:q} "
        "--output-jsonl {output.trace:q} --output-result {output.result:q} "
        "> {log:q} 2>&1"


rule run_hierarchy_confirmation_r_reference:
    input:
        authorization=current_hierarchy_authorization,
        opening=hierarchy_confirmation_work_path("{dataset}", "opening.json"),
        recipe=hierarchy_confirmation_work_path(
            "{dataset}", "sealed.recipe.rds"
        ),
        python_result=hierarchy_confirmation_python_path(
            "{dataset}", "result.json"
        ),
        reference=REFERENCE_FILES,
        script="validation/scripts/run_reference_hierarchy.R",
        wrapper="validation/scripts/run_preserving_command.py",
        lock="renv.lock",
    output:
        bundle=hierarchy_r_path(
            "{dataset}", 104729, "bundle.rds", phase="confirmation"
        ),
        equivalence=hierarchy_r_path(
            "{dataset}", 104729, "instrumentation.json", phase="confirmation"
        ),
    log:
        f"{LOG_ROOT}/hierarchy/confirmation/{{dataset}}.r.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    resources:
        mem_mb=64000,
        runtime=2880,
    shell:
        "PYTHONPATH=. {config[python]} {input.wrapper} --log {log:q} "
        "--failure-anchor {output.equivalence:q} "
        "--scope hierarchy_confirmation --stage instrumented_execution -- "
        "{config[rscript]} --vanilla {input.script} "
        "--reference-root {REFERENCE_SOURCE_ROOT:q} --recipe {input.recipe:q} "
        "--python-result {input.python_result:q} --seed 104729 "
        "--bundle {output.bundle:q} --equivalence {output.equivalence:q}"


rule canonicalize_hierarchy_confirmation_r_reference:
    input:
        candidate=CANDIDATE_MANIFEST,
        authorization=current_hierarchy_authorization,
        bundle=hierarchy_r_path(
            "{dataset}", 104729, "bundle.rds", phase="confirmation"
        ),
        equivalence=hierarchy_r_path(
            "{dataset}", 104729, "instrumentation.json", phase="confirmation"
        ),
        script="validation/scripts/run_reference_semantic_trace.R",
        wrapper="validation/scripts/run_preserving_command.py",
    output:
        trace=hierarchy_r_path(
            "{dataset}", 104729, "trace.jsonl", phase="confirmation"
        ),
        result=hierarchy_r_path(
            "{dataset}", 104729, "result.json", phase="confirmation"
        ),
    log:
        f"{LOG_ROOT}/hierarchy/confirmation/{{dataset}}.r.canonical.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    resources:
        mem_mb=8000,
        runtime=60,
    shell:
        "PYTHONPATH=. {config[python]} {input.wrapper} --log {log:q} "
        "--failure-anchor {output.result:q} "
        "--scope hierarchy_confirmation --stage canonicalization -- "
        "{config[rscript]} --vanilla {input.script} --bundle {input.bundle:q} "
        "--candidate-manifest {input.candidate:q} --output {output.trace:q} "
        "--result {output.result:q}"


rule evaluate_hierarchy_confirmation:
    input:
        authorization=current_hierarchy_authorization,
        opening=hierarchy_confirmation_work_path("{dataset}", "opening.json"),
        reference=hierarchy_r_path(
            "{dataset}", 104729, "result.json", phase="confirmation"
        ),
        candidate=hierarchy_confirmation_python_path(
            "{dataset}", "result.json"
        ),
        margins=HIERARCHY_MARGIN_EVIDENCE,
        reproduction=HIERARCHY_MARGIN_REPRODUCTION,
        role_contract=PUBLICATION_DATA_ROLES_PATH,
        script="validation/scripts/evaluate_hierarchy_fidelity.py",
    output:
        f"{EVIDENCE_ROOT}/hierarchy/confirmation/{{dataset}}.json",
    log:
        f"{LOG_ROOT}/hierarchy/confirmation/{{dataset}}.evaluate.log",
    wildcard_constraints:
        dataset="|".join(HIERARCHY_CONFIRMATION_IDS),
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--reference {input.reference:q} --candidate {input.candidate:q} "
        "--margins {input.margins:q} --role-contract {input.role_contract:q} "
        "--expected-role hierarchy_parity_confirmation "
        "--opening {input.opening:q} --output {output:q} "
        "> {log:q} 2>&1"


rule aggregate_hierarchy_fidelity:
    input:
        candidate=CANDIDATE_MANIFEST,
        design=HIERARCHY_DESIGN_PATH,
        margins=HIERARCHY_MARGIN_EVIDENCE,
        margin_reproduction=HIERARCHY_MARGIN_REPRODUCTION,
        development_role_contract=HIERARCHY_DEVELOPMENT_CONTRACT_PATH,
        confirmation_role_contract=PUBLICATION_DATA_ROLES_PATH,
        development=HIERARCHY_DEVELOPMENT_REPORTS,
        confirmation=HIERARCHY_CONFIRMATION_REPORTS,
        script="validation/scripts/aggregate_hierarchy_fidelity.py",
    output:
        HIERARCHY_AGGREGATE_EVIDENCE,
    params:
        development=lambda _wildcards, input: repeated_path_arguments(
            "--development", input.development
        ),
        confirmation=lambda _wildcards, input: repeated_path_arguments(
            "--confirmation", input.confirmation
        ),
    log:
        f"{LOG_ROOT}/hierarchy/aggregate_fidelity.log",
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--design {input.design:q} --margins {input.margins:q} "
        "--development-role-contract {input.development_role_contract:q} "
        "--confirmation-role-contract {input.confirmation_role_contract:q} "
        "{params.development} {params.confirmation} "
        "--output {output:q} > {log:q} 2>&1"


rule assess_parity_promotion:
    input:
        candidate=CANDIDATE_MANIFEST,
        gate0=GATE0_EVIDENCE,
        gate1_static=GATE1_STATIC_EVIDENCE,
        gate1_gpu=GATE1_GPU_EVIDENCE,
        framework=PARITY_FRAMEWORK_SEAL,
        comparator=GATE2_EVIDENCE,
        comparator_seal=GATE2_PASS_SEAL,
        hierarchy=HIERARCHY_AGGREGATE_EVIDENCE,
        hierarchy_margins=HIERARCHY_MARGIN_EVIDENCE,
        hierarchy_margin_reproduction=HIERARCHY_MARGIN_REPRODUCTION,
        rf_profile_reproduction=RF_PROFILE_REPRODUCTION,
        ledger="validation/contracts/parity_ledger_v1.json",
        script="validation/scripts/assess_parity_promotion.py",
    output:
        PARITY_PROMOTION_EVIDENCE,
    log:
        f"{LOG_ROOT}/parity/assess_promotion.log",
    shell:
        "PYTHONPATH=src:. {config[python]} {input.script} "
        "--root . --candidate-manifest {input.candidate:q} "
        "--gate0 {input.gate0:q} --gate1-static {input.gate1_static:q} "
        "--gate1-gpu {input.gate1_gpu:q} --framework {input.framework:q} "
        "--comparator {input.comparator:q} "
        "--comparator-seal {input.comparator_seal:q} "
        "--hierarchy {input.hierarchy:q} "
        "--hierarchy-margins {input.hierarchy_margins:q} "
        "--hierarchy-margin-reproduction "
        "{input.hierarchy_margin_reproduction:q} "
        "--rf-profile-reproduction {input.rf_profile_reproduction:q} "
        "--ledger {input.ledger:q} --output {output:q} > {log:q} 2>&1"


# Guarded publication rules are intentionally absent from rule all. Their
# ungenerated per-analysis inputs keep them inert until explicitly requested.
rule aggregate_secondary_evidence:
    input:
        contract=SECONDARY_EVIDENCE_CONTRACT,
        reports=SECONDARY_REPORTS,
        script="validation/scripts/aggregate_secondary_evidence.py",
    output:
        SECONDARY_AGGREGATE,
    params:
        reports=lambda _wildcards, input: repeated_path_arguments(
            "--report", input.reports
        ),
    log:
        f"{LOG_ROOT}/publication/secondary.aggregate.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} "
        "--contract {input.contract:q} {params.reports} "
        "--output {output:q} > {log:q} 2>&1"


rule aggregate_sant_reproduction:
    input:
        contract=SANT_REPRODUCTION_CONTRACT,
        records=SANT_RECORDS,
        script="validation/scripts/aggregate_sant_reproduction.py",
    output:
        SANT_AGGREGATE,
    params:
        records=lambda _wildcards, input: repeated_path_arguments(
            "--record", input.records
        ),
    log:
        f"{LOG_ROOT}/publication/sant.aggregate.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} "
        "--contract {input.contract:q} {params.records} "
        "--output {output:q} > {log:q} 2>&1"


rule evaluate_l40s_performance:
    input:
        contract=PERFORMANCE_CONTRACT,
        workloads=PERFORMANCE_RECORDS,
        script="validation/scripts/evaluate_l40s_performance.py",
    output:
        PERFORMANCE_AGGREGATE,
    params:
        workloads=lambda _wildcards, input: repeated_path_arguments(
            "--workload", input.workloads
        ),
    log:
        f"{LOG_ROOT}/publication/performance.evaluate.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} "
        "--contract {input.contract:q} {params.workloads} "
        "--output {output:q} > {log:q} 2>&1"


rule publication_evidence:
    input:
        candidate=CANDIDATE_MANIFEST,
        claims="validation/contracts/publication_claims_v2.json",
        data_roles=PUBLICATION_DATA_ROLES_PATH,
        parity=PARITY_PROMOTION_EVIDENCE,
        secondary=SECONDARY_AGGREGATE,
        sant=SANT_AGGREGATE,
        performance=PERFORMANCE_AGGREGATE,
        release=RELEASE_CANDIDATE_EVIDENCE,
        claim_to_evidence=lambda _wildcards: configured_artifact(
            "publication_artifacts", "claim_to_evidence_manifest"
        ),
        manuscript=lambda _wildcards: configured_artifact(
            "publication_artifacts", "manuscript"
        ),
        figures=lambda _wildcards: configured_artifact(
            "publication_artifacts", "figures_manifest"
        ),
        tables=lambda _wildcards: configured_artifact(
            "publication_artifacts", "tables_manifest"
        ),
        supplement=lambda _wildcards: configured_artifact(
            "publication_artifacts", "supplement"
        ),
        negative=lambda _wildcards: configured_artifact(
            "publication_artifacts", "negative_results_appendix"
        ),
        bmc=lambda _wildcards: configured_artifact(
            "publication_artifacts", "bmc_package"
        ),
        venue=lambda _wildcards: configured_artifact(
            "publication_artifacts", "venue_requirements_audit"
        ),
        script="validation/scripts/assemble_publication_evidence.py",
    output:
        PUBLICATION_EVIDENCE,
    params:
        artifacts=lambda _wildcards: configured_artifact_arguments(
            "publication_artifacts",
            (
                "claim_to_evidence_manifest",
                "manuscript",
                "figures_manifest",
                "tables_manifest",
                "supplement",
                "negative_results_appendix",
                "bmc_package",
                "venue_requirements_audit",
            ),
        ),
    log:
        f"{LOG_ROOT}/publication/assemble.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} --root . "
        "--candidate-manifest {input.candidate:q} --claims {input.claims:q} "
        "--data-roles {input.data_roles:q} --parity {input.parity:q} "
        "--secondary {input.secondary:q} --sant {input.sant:q} "
        "--performance {input.performance:q} "
        "--release-candidate {input.release:q} {params.artifacts} "
        "--output {output:q} > {log:q} 2>&1"


rule release_candidate:
    input:
        candidate=CANDIDATE_MANIFEST,
        parity=PARITY_PROMOTION_EVIDENCE,
        contract=RELEASE_CANDIDATE_CONTRACT,
        stable_profile=RF_PROFILE,
        installed_profile="src/rapids_choir/_data/rf_profile.json",
        wheel=lambda _wildcards: configured_artifact(
            "release_artifacts", "wheel"
        ),
        sdist=lambda _wildcards: configured_artifact(
            "release_artifacts", "sdist"
        ),
        oci=lambda _wildcards: configured_artifact(
            "release_artifacts", "oci_image_manifest"
        ),
        apptainer=lambda _wildcards: configured_artifact(
            "release_artifacts", "apptainer_definition"
        ),
        documentation=lambda _wildcards: configured_artifact(
            "release_artifacts", "documentation"
        ),
        smoke_data=lambda _wildcards: configured_artifact(
            "release_artifacts", "smoke_data"
        ),
        sbom=lambda _wildcards: configured_artifact(
            "release_artifacts", "sbom"
        ),
        release_manifest=lambda _wildcards: configured_artifact(
            "release_artifacts", "release_manifest"
        ),
        checksums=lambda _wildcards: configured_artifact(
            "release_artifacts", "checksums"
        ),
        rights=lambda _wildcards: configured_artifact(
            "release_artifacts", "rights_audit"
        ),
        credentials=lambda _wildcards: configured_artifact(
            "release_artifacts", "credential_audit"
        ),
        restricted=lambda _wildcards: configured_artifact(
            "release_artifacts", "restricted_data_audit"
        ),
        stale=lambda _wildcards: configured_artifact(
            "release_artifacts", "stale_evidence_audit"
        ),
        script="validation/scripts/assemble_release_candidate.py",
    output:
        RELEASE_CANDIDATE_EVIDENCE,
    params:
        artifacts=lambda _wildcards: configured_artifact_arguments(
            "release_artifacts",
            (
                "wheel",
                "sdist",
                "oci_image_manifest",
                "apptainer_definition",
                "documentation",
                "smoke_data",
                "sbom",
                "release_manifest",
                "checksums",
                "rights_audit",
                "credential_audit",
                "restricted_data_audit",
                "stale_evidence_audit",
            ),
        ),
    log:
        f"{LOG_ROOT}/release/assemble.log",
    shell:
        "PYTHONPATH=. {config[python]} {input.script} --root . "
        "--candidate-manifest {input.candidate:q} "
        "--contract {input.contract:q} "
        "--parity-promotion {input.parity:q} "
        "--stable-profile {input.stable_profile:q} "
        "--installed-profile {input.installed_profile:q} {params.artifacts} "
        "--output {output:q} > {log:q} 2>&1"
