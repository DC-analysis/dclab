dclab documentation
===================
To install the requirements for building the documentation, run

    pip install -r requirements.txt

To compile the documentation, run

    sphinx-build -n -W --keep-going . _build

The `-n` (nitpicky) and `-W` (warnings are errors) flags are also used in
the "Docs" GitHub Actions workflow, so broken cross-references and
docstring markup errors fail CI. Informal type descriptions in docstrings
(e.g. "1d ndarray of length N") are excluded via `nitpick_ignore_regex`
in `conf.py`.

When adding new public functions or classes, add them to
`sec_code_reference.rst` (modules that re-export members from submodules
need explicit `autoclass`/`autofunction` entries, because `automodule`
skips imported members).

Notes
=====
To view the sphinx inventory of dclab, run

   python -m sphinx.ext.intersphinx 'https://dclab.readthedocs.io/en/latest/objects.inv'
