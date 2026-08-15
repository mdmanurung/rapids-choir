SPHINXBUILD ?= sphinx-build
SOURCEDIR = .
BUILDDIR = _build

.PHONY: help html linkcheck clean

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)"

html:
	@$(SPHINXBUILD) -W --keep-going -M html "$(SOURCEDIR)" "$(BUILDDIR)"

linkcheck:
	@$(SPHINXBUILD) -W --keep-going -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)"

clean:
	@$(SPHINXBUILD) -M clean "$(SOURCEDIR)" "$(BUILDDIR)"
