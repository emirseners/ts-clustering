import os
import sys
from datetime import date


# Add project root to sys.path so Sphinx can import the package without install
CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)


# Project information
project = "ts_clustering"
author = "Your Name"
current_year = str(date.today().year)
copyright = f"{current_year}, {author}"

try:
    import ts_clustering as _pkg

    release = getattr(_pkg, "__version__", "0.1.0")
except Exception:
    release = "0.1.0"


# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

typehints_defaults = "comma"
autodoc_typehints = "description"
autodoc_typehints_format = "short"


# Options for HTML output
html_theme = "alabaster"
try:
    import sphinx_rtd_theme  # noqa: F401

    html_theme = "sphinx_rtd_theme"
except Exception:
    # fall back to default theme if RTD theme is not installed
    pass

html_static_path = ["_static"]


