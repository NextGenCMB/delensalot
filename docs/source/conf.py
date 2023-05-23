# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Path setup --------------------------------------------------------------
import os, sys

autodoc_mock_imports = ['plancklens', ', 'MSC', 'bicubic', 'mpi4py', 'attr', 'attrs', 'lensitbiases']
# sys.path.insert(0, os.path.abspath("./../"))

sys.path.insert(0, "./../")
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'delensalot'
copyright = '2023, S. Belkner, J. Carron, L. Legrand'
author = 'S. Belkner, J. Carron, L. Legrand'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    'sphinx_rtd_theme',
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    'sphinx.ext.mathjax',
]

html_theme_path = ['/home/belkner/anaconda3/envs/delensalot/lib/python3.11/site-packages/sphinx_rtd_theme',]

html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    # Toc options
    # 'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    # 'includehidden': True,
    'titles_only': True
}


# mathjax settings
mathjax_options ={
    'extensions': ['tex2jax.js'],
    'tex2jax': {
        'inlineMath': [["$", "$"], ["\\(", "\\)"]],
        'displayMath': [['$$', '$$'], ["\\[", "\\]"]],
        'processEscapes': True
    },
    'HTML-CSS': {'fonts': ['TeX']},
    "mathjax_path": "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS-MML_HTMLorMML" }

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
# html_theme_options = {
#     "rightsidebar": "true",
#     "relbarbgcolor": "black"
# }
html_logo = "_static/dlensalot.PNG"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']