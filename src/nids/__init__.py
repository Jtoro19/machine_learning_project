"""Shared preprocessing and analysis package for the UNSW-NB15 network intrusion project.

This module intentionally stays free of re-exports and heavy imports. Anything that
imports pandas, scikit-learn, or matplotlib lives in a submodule (for example
``nids.data``), so ``import nids`` alone never triggers those imports and stays cheap
enough to run in every notebook cell and test module.
"""

__all__: list[str] = []
