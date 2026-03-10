# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RegMMD'
copyright = '2026, Hidde Fokkema'
author = 'Hidde Fokkema'
release = '0.1.0'

extlinks = {
    'blob': ('https://github.com/HiddeFok/reg-mmd-scikit/reg-mmd-scikit/blob/{}/main', '%s'),
    'tree': ('https://github.com/HiddeFok/reg-mmd-scikit/reg-mmd-scikit/tree/{}/main', '%s')
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.extlinks',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax'
]

templates_path = ['_templates']
exclude_patterns = []

# conf.py
suppress_warnings = ['ref.term', 'ref.ref']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'

html_static_path = ['_static']
html_css_files = ['custom.css']

