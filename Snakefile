from pathlib import Path
from itertools import product
import shlex
import yaml

configfile: "workflow/config.yaml"

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
    and path.suffix in {".py", ".R"}
    and "__pycache__" not in path.parts
)

DATASET_SPEC_PATH = "validation/fixtures/specs/datasets.yaml"
with open(DATASET_SPEC_PATH, encoding="utf-8") as handle:
    DATASET_SPEC = yaml.safe_load(handle)


def dataset_slug(dataset_id):
    return f"simulated_dataset_{int(dataset_id.rsplit(' ', 1)[1]):03d}"


DEVELOPMENT = DATASET_SPEC["development"]["datasets"]
HOLDOUT = DATASET_SPEC["holdout"]["datasets"]
DEV_BY_SLUG = {dataset_slug(record["id"]): record for record in DEVELOPMENT}
HOLD_BY_SLUG = {dataset_slug(record["id"]): record for record in HOLDOUT}
RAW_ROOT = config["data"]["raw_root"]
WORK_ROOT = config["data"]["work_root"]
ARCHIVE = f"{RAW_ROOT}/{DATASET_SPEC['dataset_source']['archive']['name']}"
ARCHIVE_VERIFY = f"{RAW_ROOT}/Simulated_Datasets.md5.json"
ARCHIVE_INVENTORY = f"{WORK_ROOT}/inventory/Simulated_Datasets.members.json"
DEV_RAW_ROOT = f"{WORK_ROOT}/development/raw"
HOLD_RAW_ROOT = f"{WORK_ROOT}/holdout/raw"
DEV_EXTRACT_MANIFEST = f"{WORK_ROOT}/development/extraction.json"
HOLD_EXTRACT_MANIFEST = f"{WORK_ROOT}/holdout/extraction.json"
TRUTH_CONTRACT = f"{WORK_ROOT}/development/truth_contract.json"
PROFILE_SEAL = f"{WORK_ROOT}/holdout/rf_profile.sealed.json"
RF_PROFILE = "validation/evidence/rf_profile.json"


def raw_path(mode, record):
    root = DEV_RAW_ROOT if mode == "development" else HOLD_RAW_ROOT
    return f"{root}/{dataset_slug(record['id'])}.rds"


def h5ad_path(mode, record):
    return f"{WORK_ROOT}/{mode}/h5ad/{dataset_slug(record['id'])}.h5ad"


def data_manifest_path(mode, record):
    return f"{WORK_ROOT}/{mode}/manifests/{dataset_slug(record['id'])}.json"


def pair_manifest_path(mode, record):
    return f"{WORK_ROOT}/{mode}/pairs/{dataset_slug(record['id'])}.json"


def reference_path(mode, record):
    return f"{WORK_ROOT}/{mode}/reference/{dataset_slug(record['id'])}.json"


DEV_RDS = [raw_path("development", record) for record in DEVELOPMENT]
HOLD_RDS = [raw_path("holdout", record) for record in HOLDOUT]
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

RF_GRID = [
    {
        "id": f"bins{n_bins}_depth{'none' if depth is None else depth}_streams{streams}",
        "n_bins": n_bins,
        "max_depth": "none" if depth is None else str(depth),
        "n_streams": streams,
    }
    for n_bins, depth, streams in product((128, 256, 512), (16, None), (1, 4))
]
RF_BY_ID = {record["id"]: record for record in RF_GRID}
CALIBRATION_SUMMARIES = [
    f"validation/evidence/gate2/calibration/{record['id']}.json"
    for record in RF_GRID
]
CALIBRATION_PARQUETS = [
    f"validation/evidence/gate2/calibration/{record['id']}.parquet"
    for record in RF_GRID
]


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
rule all:
    input:
        "validation/evidence/gate0.json",
        "validation/evidence/gate1_static.json",


rule gate2_development:
    input:
        DEV_PAIR_MANIFESTS,
        DEV_REFERENCES,


rule gate2_calibration:
    input:
        RF_PROFILE,


rule gate2_holdout:
    input:
        "validation/evidence/gate2.json",


rule r_logic_fixtures:
    input:
        script="validation/scripts/generate_logic_fixtures.R",
        lock="renv.lock",
        reference=REFERENCE_FILES,
    output:
        FIXTURE_FILES,
    log:
        "validation/logs/r_logic_fixtures.log",
    shell:
        "{config[rscript]} --vanilla {input.script} validation/fixtures/generated > {log} 2>&1"


rule fixture_manifest:
    input:
        script="validation/scripts/write_manifest.py",
        fixtures=FIXTURE_FILES,
    output:
        "validation/fixtures/generated/MANIFEST.json",
    log:
        "validation/logs/fixture_manifest.log",
    shell:
        "{config[python]} {input.script} --root . --output {output} {input.fixtures} > {log} 2>&1"


rule gate0:
    input:
        script="validation/scripts/check_gate0.py",
        manifest="validation/fixtures/generated/MANIFEST.json",
        fixtures=FIXTURE_FILES,
        reference=REFERENCE_FILES,
        lock="renv.lock",
        datasets="validation/fixtures/specs/datasets.yaml",
        audit="docs/validation/source-audit.md",
        architecture="docs/architecture/architecture.md",
        decisions="docs/decisions/README.md",
        license="LICENSE",
        notice="NOTICE",
        citation="CITATION.cff",
    output:
        "validation/evidence/gate0.json",
    log:
        "validation/logs/gate0.log",
    shell:
        "{config[python]} {input.script} --root . --fixtures validation/fixtures/generated --output {output} > {log} 2>&1"


rule gate1_static:
    input:
        gate0="validation/evidence/gate0.json",
        script="validation/scripts/run_gate1_static.py",
        package=STATIC_PACKAGE_FILES,
        project="pyproject.toml",
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        "validation/evidence/gate1_static.json",
    log:
        "validation/logs/gate1_static.log",
    shell:
        "{config[gpu_python]} {input.script} --root . --output {output} --log {log}"


rule synthetic_comparator_fixture:
    input:
        script="validation/scripts/generate_comparator_fixture.R",
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        lock="renv.lock",
    output:
        "validation/fixtures/comparator_synthetic.json",
    log:
        "validation/logs/comparator_synthetic.log",
    shell:
        "{config[rscript]} --vanilla {input.script} {input.source:q} {output:q} 1 "
        "> {log:q} 2>&1"


rule gate1_gpu:
    input:
        static="validation/evidence/gate1_static.json",
        synthetic="validation/fixtures/comparator_synthetic.json",
        script="validation/scripts/run_gate1_gpu.py",
        package=STATIC_PACKAGE_FILES,
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        "validation/evidence/gate1_gpu.json",
    log:
        "validation/logs/gate1_gpu.log",
    threads: 4
    resources:
        mem_mb=32000,
        runtime=30,
        slurm_partition="gpu-short",
        gres="gpu:L40S:1",
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} "
        "{config[gpu_python]} {input.script} --root . --output {output} --log {log}"


rule download_dataset_archive:
    input:
        script="validation/scripts/download_zenodo.py",
    output:
        ARCHIVE,
    params:
        url=config["data"]["zenodo_archive_url"],
        expected_bytes=DATASET_SPEC["dataset_source"]["archive"]["bytes"],
    log:
        "validation/logs/gate2/download_archive.log",
    resources:
        mem_mb=2000,
        runtime=1440,
    shell:
        "{config[python]} {input.script} --url {params.url:q} --output {output:q} "
        "--expected-bytes {params.expected_bytes} > {log:q} 2>&1"


rule verify_dataset_archive:
    input:
        archive=ARCHIVE,
        script="validation/scripts/verify_dataset_archive.py",
    output:
        ARCHIVE_VERIFY,
    params:
        expected_bytes=DATASET_SPEC["dataset_source"]["archive"]["bytes"],
        expected_md5=DATASET_SPEC["dataset_source"]["archive"]["md5"],
    log:
        "validation/logs/gate2/verify_archive.log",
    resources:
        mem_mb=2000,
        runtime=240,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--expected-bytes {params.expected_bytes} --expected-md5 {params.expected_md5} "
        "--output {output:q} > {log:q} 2>&1"


rule inventory_dataset_archive:
    input:
        archive=ARCHIVE,
        verification=ARCHIVE_VERIFY,
        script="validation/scripts/inventory_dataset_archive.py",
    output:
        ARCHIVE_INVENTORY,
    log:
        "validation/logs/gate2/inventory_archive.log",
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
        script="validation/scripts/extract_datasets.py",
    output:
        manifest=DEV_EXTRACT_MANIFEST,
        datasets=DEV_RDS,
    params:
        datasets=extraction_arguments(DEVELOPMENT),
        output_root=DEV_RAW_ROOT,
    log:
        "validation/logs/gate2/extract_development.log",
    resources:
        mem_mb=4000,
        runtime=480,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--verification {input.verification:q} --inventory {input.inventory:q} "
        "--mode development {params.datasets} --output-root {params.output_root:q} "
        "--manifest {output.manifest:q} --root . > {log:q} 2>&1"


rule discover_development_truth_label:
    input:
        datasets=DEV_RDS,
        extraction=DEV_EXTRACT_MANIFEST,
        script="validation/scripts/discover_truth_label.R",
    output:
        TRUTH_CONTRACT,
    params:
        datasets=truth_arguments(DEVELOPMENT),
    log:
        "validation/logs/gate2/discover_truth_label.log",
    resources:
        mem_mb=64000,
        runtime=240,
    shell:
        "{config[rscript]} --vanilla {input.script} {params.datasets} "
        "--output {output:q} > {log:q} 2>&1"


rule preprocess_development_dataset:
    input:
        raw=lambda wildcards: raw_path(
            "development", DEV_BY_SLUG[wildcards.dataset]
        ),
        truth=TRUTH_CONTRACT,
        script="validation/scripts/preprocess_seurat.R",
    output:
        f"{WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: DEV_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        "validation/logs/gate2/development/{dataset}.preprocess.log",
    resources:
        mem_mb=64000,
        runtime=240,
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --input {input.raw:q} "
        "--output {output:q} --truth-contract {input.truth:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} > {log:q} 2>&1"


rule verify_development_h5ad:
    input:
        h5ad=f"{WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
        script="validation/scripts/verify_h5ad_contract.py",
    output:
        f"{WORK_ROOT}/development/manifests/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: DEV_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        "validation/logs/gate2/development/{dataset}.verify_h5ad.log",
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[python]} {input.script} --input {input.h5ad:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} --output {output:q} "
        "> {log:q} 2>&1"


rule build_development_pairs:
    input:
        h5ad=f"{WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
        manifest=f"{WORK_ROOT}/development/manifests/{{dataset}}.json",
        script="validation/scripts/build_pair_manifest.py",
    output:
        f"{WORK_ROOT}/development/pairs/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
        seed=DATASET_SPEC["seeds"]["calibration_profiles"],
    log:
        "validation/logs/gate2/development/{dataset}.pairs.log",
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[python]} {input.script} --h5ad {input.h5ad:q} "
        "--data-manifest {input.manifest:q} --dataset-id {params.dataset_id:q} "
        "--seed {params.seed} --output {output:q} > {log:q} 2>&1"


rule run_development_r_reference:
    input:
        h5ad=f"{WORK_ROOT}/development/h5ad/{{dataset}}.h5ad",
        pairs=f"{WORK_ROOT}/development/pairs/{{dataset}}.json",
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        script="validation/scripts/run_reference_comparator.R",
    output:
        f"{WORK_ROOT}/development/reference/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: DEV_BY_SLUG[wildcards.dataset]["id"],
    log:
        "validation/logs/gate2/development/{dataset}.r_reference.log",
    resources:
        mem_mb=64000,
        runtime=1440,
    wildcard_constraints:
        dataset="|".join(DEV_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --h5ad {input.h5ad:q} "
        "--pairs {input.pairs:q} --dataset-id {params.dataset_id:q} "
        "--reference-source {input.source:q} --output {output:q} "
        "> {log:q} 2>&1"


rule run_development_gpu_profile:
    input:
        gate1="validation/evidence/gate1_gpu.json",
        h5ad=DEV_H5AD,
        manifests=DEV_DATA_MANIFESTS,
        pairs=DEV_PAIR_MANIFESTS,
        references=DEV_REFERENCES,
        script="validation/scripts/run_gpu_comparisons.py",
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        summary="validation/evidence/gate2/calibration/{profile}.json",
        parquet="validation/evidence/gate2/calibration/{profile}.parquet",
    params:
        datasets=dataset_arguments(DEVELOPMENT, "development", True),
        n_bins=lambda wildcards: RF_BY_ID[wildcards.profile]["n_bins"],
        max_depth=lambda wildcards: RF_BY_ID[wildcards.profile]["max_depth"],
        n_streams=lambda wildcards: RF_BY_ID[wildcards.profile]["n_streams"],
    log:
        "validation/logs/gate2/calibration/{profile}.log",
    threads: config["slurm"]["cpus_per_task"]
    resources:
        mem_mb=64000,
        runtime=config["slurm"]["runtime_minutes"],
        slurm_partition=config["slurm"]["partition"],
        gres=config["slurm"]["gres"],
    wildcard_constraints:
        profile="|".join(RF_BY_ID),
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} {config[gpu_python]} {input.script} "
        "--root . {params.datasets} --n-bins {params.n_bins} "
        "--max-depth {params.max_depth} --n-streams {params.n_streams} "
        "--repeats 5 --output-parquet {output.parquet:q} "
        "--output-summary {output.summary:q} > {log:q} 2>&1"


rule freeze_development_rf_profile:
    input:
        summaries=CALIBRATION_SUMMARIES,
        parquets=CALIBRATION_PARQUETS,
        script="validation/scripts/freeze_rf_profile.py",
        conda_lock="workflow/envs/rapids-cu128-linux-64.lock",
        pip_lock="workflow/envs/rapids-cu128-pip.lock",
    output:
        profile=RF_PROFILE,
        aggregate="validation/evidence/gate2/calibration_results.parquet",
    params:
        runs=calibration_run_arguments(),
    log:
        "validation/logs/gate2/freeze_rf_profile.log",
    shell:
        "PYTHONPATH=src {config[python]} {input.script} --root . {params.runs} "
        "--conda-lock {input.conda_lock:q} --pip-lock {input.pip_lock:q} "
        "--output {output.profile:q} --aggregate-parquet {output.aggregate:q} "
        "> {log:q} 2>&1"


rule seal_committed_rf_profile:
    input:
        profile=RF_PROFILE,
        script="validation/scripts/check_profile_seal.py",
    output:
        PROFILE_SEAL,
    log:
        "validation/logs/gate2/seal_rf_profile.log",
    shell:
        "{config[python]} {input.script} --root . --profile {input.profile:q} "
        "--output {output:q} > {log:q} 2>&1"


rule extract_holdout_datasets:
    input:
        archive=ARCHIVE,
        verification=ARCHIVE_VERIFY,
        inventory=ARCHIVE_INVENTORY,
        seal=PROFILE_SEAL,
        script="validation/scripts/extract_datasets.py",
    output:
        manifest=HOLD_EXTRACT_MANIFEST,
        datasets=HOLD_RDS,
    params:
        datasets=extraction_arguments(HOLDOUT),
        output_root=HOLD_RAW_ROOT,
    log:
        "validation/logs/gate2/extract_holdout.log",
    resources:
        mem_mb=4000,
        runtime=480,
    shell:
        "{config[python]} {input.script} --archive {input.archive:q} "
        "--verification {input.verification:q} --inventory {input.inventory:q} "
        "--mode holdout {params.datasets} --output-root {params.output_root:q} "
        "--manifest {output.manifest:q} --profile-seal {input.seal:q} "
        "--root . > {log:q} 2>&1"


rule preprocess_holdout_dataset:
    input:
        raw=lambda wildcards: raw_path("holdout", HOLD_BY_SLUG[wildcards.dataset]),
        truth=TRUTH_CONTRACT,
        seal=PROFILE_SEAL,
        script="validation/scripts/preprocess_seurat.R",
    output:
        f"{WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        "validation/logs/gate2/holdout/{dataset}.preprocess.log",
    resources:
        mem_mb=64000,
        runtime=240,
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --input {input.raw:q} "
        "--output {output:q} --truth-contract {input.truth:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} > {log:q} 2>&1"


rule verify_holdout_h5ad:
    input:
        h5ad=f"{WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
        seal=PROFILE_SEAL,
        script="validation/scripts/verify_h5ad_contract.py",
    output:
        f"{WORK_ROOT}/holdout/manifests/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
        cells=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["cells"],
        groups=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset][
            "ground_truth_groups"
        ],
    log:
        "validation/logs/gate2/holdout/{dataset}.verify_h5ad.log",
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[python]} {input.script} --input {input.h5ad:q} "
        "--dataset-id {params.dataset_id:q} --expected-cells {params.cells} "
        "--expected-groups {params.groups} --output {output:q} "
        "> {log:q} 2>&1"


rule build_holdout_pairs:
    input:
        h5ad=f"{WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
        manifest=f"{WORK_ROOT}/holdout/manifests/{{dataset}}.json",
        seal=PROFILE_SEAL,
        script="validation/scripts/build_pair_manifest.py",
    output:
        f"{WORK_ROOT}/holdout/pairs/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
        seed=DATASET_SPEC["seeds"]["calibration_profiles"],
    log:
        "validation/logs/gate2/holdout/{dataset}.pairs.log",
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[python]} {input.script} --h5ad {input.h5ad:q} "
        "--data-manifest {input.manifest:q} --dataset-id {params.dataset_id:q} "
        "--seed {params.seed} --output {output:q} > {log:q} 2>&1"


rule run_holdout_r_reference:
    input:
        h5ad=f"{WORK_ROOT}/holdout/h5ad/{{dataset}}.h5ad",
        pairs=f"{WORK_ROOT}/holdout/pairs/{{dataset}}.json",
        seal=PROFILE_SEAL,
        source="validation/reference/choir_v0_3_0/R/ComparisonUtils.R",
        script="validation/scripts/run_reference_comparator.R",
    output:
        f"{WORK_ROOT}/holdout/reference/{{dataset}}.json",
    params:
        dataset_id=lambda wildcards: HOLD_BY_SLUG[wildcards.dataset]["id"],
    log:
        "validation/logs/gate2/holdout/{dataset}.r_reference.log",
    resources:
        mem_mb=64000,
        runtime=1440,
    wildcard_constraints:
        dataset="|".join(HOLD_BY_SLUG),
    shell:
        "{config[rscript]} --vanilla {input.script} --h5ad {input.h5ad:q} "
        "--pairs {input.pairs:q} --dataset-id {params.dataset_id:q} "
        "--reference-source {input.source:q} --output {output:q} "
        "> {log:q} 2>&1"


rule run_holdout_gpu_profile:
    input:
        profile=RF_PROFILE,
        seal=PROFILE_SEAL,
        h5ad=HOLD_H5AD,
        manifests=HOLD_DATA_MANIFESTS,
        pairs=HOLD_PAIR_MANIFESTS,
        references=HOLD_REFERENCES,
        script="validation/scripts/run_gpu_comparisons.py",
    output:
        summary="validation/evidence/gate2/holdout_repeats.json",
        parquet="validation/evidence/gate2/holdout_repeats.parquet",
    params:
        datasets=dataset_arguments(HOLDOUT, "holdout", True),
    log:
        "validation/logs/gate2/holdout_gpu.log",
    threads: config["slurm"]["cpus_per_task"]
    resources:
        mem_mb=64000,
        runtime=config["slurm"]["runtime_minutes"],
        slurm_partition=config["slurm"]["partition"],
        gres=config["slurm"]["gres"],
    shell:
        "CONDA_PREFIX={config[gpu_prefix]} {config[gpu_python]} {input.script} "
        "--root . {params.datasets} --profile-json {input.profile:q} --repeats 5 "
        "--output-parquet {output.parquet:q} --output-summary {output.summary:q} "
        "> {log:q} 2>&1"


rule assemble_gate2_acceptance:
    input:
        profile=RF_PROFILE,
        seal=PROFILE_SEAL,
        summary="validation/evidence/gate2/holdout_repeats.json",
        parquet="validation/evidence/gate2/holdout_repeats.parquet",
        spec=DATASET_SPEC_PATH,
        script="validation/scripts/assemble_gate2.py",
    output:
        evidence="validation/evidence/gate2.json",
        aggregate="validation/evidence/gate2/holdout_results.parquet",
    log:
        "validation/logs/gate2/assemble_gate2.log",
    shell:
        "PYTHONPATH=src {config[python]} {input.script} --root . "
        "--dataset-spec {input.spec:q} --profile {input.profile:q} "
        "--profile-seal {input.seal:q} --run-summary {input.summary:q} "
        "--run-parquet {input.parquet:q} --aggregate-parquet {output.aggregate:q} "
        "--output {output.evidence:q} > {log:q} 2>&1"
