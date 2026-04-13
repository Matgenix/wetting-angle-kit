# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# Add the project root to the Python path so autodoc can find the package
import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "wetting_angle_kit"  # Capitalized for proper naming conventions
copyright = "2025, Gabriel"
author = "Gabriel"
release = "0.1.2"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinxemoji.sphinxemoji",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "sphinx_code_tabs",
    "sphinx_issues",
    "sphinx.ext.mathjax",  # Kept from original
    "sphinx.ext.coverage",  # Kept from original
    "sphinx.ext.autosummary",  # Kept from original
    "sphinxcontrib.mermaid",
    "myst_parser",
]

# Autosummary settings
autosummary_generate = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Exclude input prompts from copybutton
copybutton_exclude = ".linenos, .gp, .go"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Path to GitHub repo {group}/{project}
issues_github_path = "Matgenix/wetting_angle_kit"

# which is the equivalent to:
issues_uri = "https://github.com/{group}/{project}/issues/{issue}"
issues_prefix = "#"
issues_pr_uri = "https://github.com/{group}/{project}/pull/{pr}"
issues_pr_prefix = "#"
issues_commit_uri = "https://github.com/{group}/{project}/commit/{commit}"
issues_commit_prefix = "@"
issues_user_uri = "https://github.com/{user}"
issues_user_prefix = "@"

# -- Napoleon options (for better docstring parsing) -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_attr_annotations = True
