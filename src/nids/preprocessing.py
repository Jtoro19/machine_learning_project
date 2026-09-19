"""Factory for the project's one fitted preprocessing pipeline.

No function in this module accepts a DataFrame or any data parameter at all
(design Decision 5) -- the entire public surface is one `bool` and one
no-argument function. A caller who wants to fit must write
`preprocessor.fit(cleaned.train[feature_columns()])` itself, naming the frame
at the call site, which is precisely the line the AGENTS.md review checklist
looks for.

    pipe = build_preprocessor()
    pipe.fit(cleaned.train[feature_columns()])
    X_test = pipe.transform(cleaned.test[feature_columns()])
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from nids.columns import (
    RARE_GROUPED_COLUMNS,
    feature_columns,
    numeric_feature_columns,
    skewed_feature_columns,
)
from nids.transformers import RareCategoryGrouper


def build_preprocessor(include_ttl_features: bool = True) -> Pipeline:
    """Return an UNFITTED preprocessing pipeline.

    Assembles a `ColumnTransformer` with three branches: `categorical`
    (rare-category grouping then one-hot encoding, over `RARE_GROUPED_COLUMNS`),
    `skewed` (`log1p` then `StandardScaler`, in that order -- reversing it
    produces NaN, see design Decision 6), and `numeric` (`StandardScaler`
    only). `remainder="drop"` so `attack_cat`/`label` are dropped even if the
    caller passes the full cleaned frame instead of `feature_columns()`.

    Args:
        include_ttl_features: When `False`, `sttl` and `ct_state_ttl` are
            excluded from the `numeric` branch's column selection. This is a
            structural toggle over which columns enter the transformer; it
            never changes an already-fitted value.

    Returns:
        An unfitted `Pipeline` wrapping the `ColumnTransformer` as its single
        `"features"` step, so a notebook can append a model step and the
        returned object always answers `.fit`, `.transform`, and
        `.get_feature_names_out` uniformly.
    """
    selected = feature_columns(include_ttl_features)
    selected_set = set(selected)
    categorical = [column for column in RARE_GROUPED_COLUMNS if column in selected_set]
    skewed = skewed_feature_columns(include_ttl_features)
    numeric = numeric_feature_columns(include_ttl_features)

    # Runtime invariant: catches a bad edit to columns.py at construction time
    # rather than silently producing a narrower design matrix.
    groups = (set(categorical), set(skewed), set(numeric))
    pairwise_disjoint = all(
        left.isdisjoint(right) for i, left in enumerate(groups) for right in groups[i + 1 :]
    )
    if not pairwise_disjoint:
        raise ValueError("categorical/skewed/numeric routing groups overlap.")
    if (groups[0] | groups[1] | groups[2]) != selected_set:
        raise ValueError(
            "categorical/skewed/numeric routing groups do not exactly cover the "
            "selected feature set."
        )

    column_transformer = ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("group_rare", RareCategoryGrouper()),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                dtype=np.float64,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
            (
                "skewed",
                Pipeline(
                    [
                        (
                            "log1p",
                            FunctionTransformer(
                                np.log1p,
                                inverse_func=np.expm1,
                                feature_names_out="one-to-one",
                                validate=False,
                            ),
                        ),
                        ("scale", StandardScaler()),
                    ]
                ),
                skewed,
            ),
            ("numeric", StandardScaler(), numeric),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline([("features", column_transformer)])


def build_preprocessor_pair() -> dict[str, Pipeline]:
    """Return `{"with_ttl": ..., "without_ttl": ...}` for AGENTS.md's paired reporting.

    AGENTS.md requires supervised results to be reported both with and
    without `sttl`/`ct_state_ttl`. This function lets a notebook iterate a
    dict instead of copy-pasting the with/without cell.

    Returns:
        Two independently constructed, unfitted pipelines differing only in
        whether `sttl` and `ct_state_ttl` enter the `numeric` branch; every
        other fitted parameter is computed independently and correctly for
        each once the caller fits both.
    """
    return {
        "with_ttl": build_preprocessor(include_ttl_features=True),
        "without_ttl": build_preprocessor(include_ttl_features=False),
    }
