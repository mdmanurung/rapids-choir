# Documentation development

Build the site in an environment that can import the package. The import does
not initialize CUDA.

## Install documentation tools

```bash
python -m pip install -e ".[docs]"
```

## Build HTML

```bash
make -C docs html
```

Sphinx writes the site to `docs/_build/html` and treats warnings as errors.

## Check links

```bash
make -C docs linkcheck
```

The link check needs network access for external targets. Fix all internal
link failures before you publish the site.

## Writing rules

Keep procedures short and use one term for each package concept. Link to the
contract or API definition. Do not copy it. Verify every symbol, path, flag,
and number against the source before you change a page.
