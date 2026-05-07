# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = "RegMMD"
copyright = "2026, Hidde Fokkema"
author = "Hidde Fokkema"
release = "0.1.0"

extlinks = {
    "blob": (
        "https://github.com/HiddeFok/reg-mmd-scikit/reg-mmd-scikit/blob/{}/main",
        "%s",
    ),
    "tree": (
        "https://github.com/HiddeFok/reg-mmd-scikit/reg-mmd-scikit/tree/{}/main",
        "%s",
    ),
}

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.doctest",
    "sphinx.ext.linkcode",
]

templates_path = ["_templates"]
exclude_patterns = []

# conf.py
suppress_warnings = ["ref.term", "ref.ref"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"

html_static_path = ["_static"]
html_css_files = ["custom.css"]

doctest_global_setup = """
import numpy as np
rng = np.random.default_rng(0)
"""

# GH Pages
html_baseurl = "https://hiddefok.github.io/reg-mmd-scikit/"


def linkcode_resolve(domain, info):
    if domain != "py" or not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    return f"https://github.com/hiddefok/reg-mmd-scikit/blob/main/{filename}.py"


autodoc_type_aliases = {
    "NDArray": "NDArray",
    "ndarray[tuple[Any, ...], dtype[_ScalarT]]": "NDArray",
}
autodoc_typehints = "description"

# Auto-generate stub .rst pages for every entry in :toctree: autosummary blocks.
autosummary_generate = True


def _write_model_index(app):
    """Regenerate the estimation/regression model API pages from ``__all__``.

    Keeps the autosummary lists on those pages in lockstep with the package
    so adding a new model class to ``regmmd.models`` automatically surfaces it
    in the docs after a rebuild.
    """
    from pathlib import Path
    from regmmd.models.estimation import __all__ as estimation_all
    from regmmd.models.regression import __all__ as regression_all

    api_dir = Path(app.srcdir) / "api"

    def _render(title, intro, names):
        lines = [
            title,
            "-" * len(title),
            "",
            intro,
            "",
            ".. autosummary::",
            "   :toctree: generated/",
            "   :nosignatures:",
            "",
        ]
        lines += [f"   regmmd.models.{name}" for name in names]
        lines.append("")
        return "\n".join(lines)

    estimation_intro = (
        "The package ships with parametric estimation models covering the "
        "most common univariate distributions. Each model can be selected by "
        "string name in :class:`~regmmd.MMDEstimator` (e.g. "
        "``model=\"gaussian-loc\"``) or by passing a class instance directly. "
        "``par_v`` denotes the variable parameter(s) that are optimised; "
        "``par_c`` denotes constant parameter(s) that are held fixed. Click "
        "a model name for its full reference page."
    )
    regression_intro = (
        "The regression models follow the same conventions as the estimation "
        "models (see :doc:`estimation_models`). Select a model by string name "
        "in :class:`~regmmd.MMDRegressor` (e.g. ``model=\"linear-gaussian-loc\"``) "
        "or pass a class instance directly. Click a model name for its full "
        "reference page."
    )

    (api_dir / "estimation_models.rst").write_text(
        _render("Estimation Models", estimation_intro, estimation_all)
    )
    (api_dir / "regression_models.rst").write_text(
        _render("Regression Models", regression_intro, regression_all)
    )


def setup(app):
    app.connect("builder-inited", _write_model_index)
