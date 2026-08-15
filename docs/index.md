# Installation

`rapids-choir` requires Python 3.12, CUDA 12.8, RAPIDS 26.06, and one NVIDIA
L40S. It has no CPU analysis backend.

Create the environment from the explicit lock. Install the hash-checked
`rapids-singlecell` wheel, the source checkout, and then check the allocated
GPU:

```bash
conda create --prefix .venv/rapids-choir \
  --file workflow/envs/rapids-cu128-linux-64.lock
conda run --prefix .venv/rapids-choir \
  python -m pip install --no-deps --require-hashes \
  -r workflow/envs/rapids-cu128-pip.lock
conda run --prefix .venv/rapids-choir python -m pip install -e .
conda run --prefix .venv/rapids-choir \
  rapids-choir doctor --device cuda:0 --strict
```

Do not start an analysis if this command fails.

:::{warning}
The package is pre-alpha software under gated scientific validation. It has no
authorized parity, performance, or release claim.
:::

```{toctree}
:maxdepth: 2
:hidden:

quick-start
detailed-step-by-step
api
changelog
```
