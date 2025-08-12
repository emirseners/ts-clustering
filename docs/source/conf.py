import os
import sys
from datetime import date

CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

project = "Time Series Clustering"
author = "Gonenc"
current_year = str(date.today().year)
copyright = f"{current_year}, {author}"

release = "0.1.0"

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

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]


autodoc_mock_imports = [
    "numpy",
    "scipy", 
    "scipy.spatial",
    "scipy.spatial.distance",
    "scipy.cluster",
    "scipy.cluster.hierarchy", 
    "pandas",
    "matplotlib",
    "matplotlib.pyplot",
    "xlsxwriter",
    "numba",
    "pysd",
]