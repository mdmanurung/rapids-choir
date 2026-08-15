"""Sphinx configuration for the rapids-choir documentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rapids_choir import __version__  # noqa: E402

project = "rapids-choir"
author = "M. D. Manurung"
copyright = "2026, M. D. Manurung"
version = __version__
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "architecture/**",
    "decisions/**",
    "validation/**",
    "wayfinder/**",
    "cli.md",
    "contributing.md",
    "getting-started.md",
    "overview.md",
    "user-guide.md",
]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
myst_heading_anchors = 3

html_theme = "furo"
html_title = f"rapids-choir {version}"
html_theme_options = {
    "navigation_with_keys": True,
    "top_of_page_buttons": ["view", "edit"],
    "source_repository": "https://github.com/mdmanurung/rapids-choir/",
    "source_branch": "main",
    "source_directory": "docs/",
    "light_css_variables": {
        "color-brand-primary": "#126E82",
        "color-brand-content": "#126E82",
    },
    "dark_css_variables": {
        "color-brand-primary": "#58C5D8",
        "color-brand-content": "#58C5D8",
    },
}
