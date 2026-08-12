from pathlib import Path

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
)
rule all:
    input:
        "validation/evidence/gate0.json",
        "validation/evidence/gate1_static.json",


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


rule gate1_gpu:
    input:
        static="validation/evidence/gate1_static.json",
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
        "{config[gpu_python]} {input.script} --root . --output {output} --log {log}"
