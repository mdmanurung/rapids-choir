# Environment locks

`rapids-cu128.yml` is the human-readable stable baseline. The release and GPU
validation environment is created only from the fully solved explicit
`rapids-cu128-linux-64.lock`, followed by the hash-checked prebuilt package in
`rapids-cu128-pip.lock`; no existing workspace RAPIDS environment is a
substitute.

RAPIDS' supported flexible channel priority is used with only `rapidsai`,
`conda-forge`, and `nvidia`. The explicit lock is regenerated from the YAML,
then installed at a new prefix. The exact `rapids-singlecell-cu12==0.16.1`
wheel is installed with `--no-deps --require-hashes` so pip cannot replace the
Conda-solved RAPIDS stack. Both lock checksums are recorded in GPU evidence.
CUDA 13 requires a separate YAML, explicit lock, and complete evidence gates;
it is not part of the stable baseline.

Snakemake rules call these governed absolute runtimes rather than asking
Snakemake to solve them again. Consequently, `snakemake --lint` reports its
generic conda/container recommendation for those rules. That warning is an
intentional exception: the workflow inputs include both locks, and every
evidence record stores their checksums and the resolved Python executable.
