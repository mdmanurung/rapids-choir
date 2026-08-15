# Command-line interface

The command-line interface checks the runtime. It does not run an analysis.

## Show the version

```bash
rapids-choir --version
```

## Check the GPU runtime

```bash
rapids-choir doctor --device cuda:0 --strict
```

The doctor checks the CUDA toolkit location, device, and RAPIDS Memory Manager
(RMM) allocation. It also checks RAPIDS versions, a small cuML forest, and a
small cuGraph Louvain run. With `--strict`, any failed check returns status 1.

Use JSON output for an evidence or scheduler wrapper:

```bash
rapids-choir doctor --device cuda:0 --strict --json
```
