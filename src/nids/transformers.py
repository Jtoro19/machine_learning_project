"""The project's first fitted component: rare-category grouping.

`RareCategoryGrouper` learns, at `fit()` time, which categories of a set of
columns are "frequent" in the training partition it is fitted on, and folds
every other value into a single `"other"` label at `transform()` time. It
MUST be fitted on the CLEANED training partition -- the output of
`nids.cleaning.clean_partitions()` -- never on the raw partition, because
deduplication reshapes the categorical distributions the threshold is
computed from (see design Decision 3).
"""

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from nids.columns import OTHER_CATEGORY, RARE_CATEGORY_THRESHOLD
from nids.loading import STRING_DTYPE

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fold categories below a train-frequency threshold into a single label.

    MUST be fitted on the CLEANED training partition. Fitting on the raw
    partition learns a different category set (see design Decision 3):
    deduplication changes each column's row-count denominator and therefore
    which category values cross the threshold.

    Fitted attributes (set by `fit`, trailing underscore per scikit-learn
    convention):
        frequent_categories_: `dict[str, frozenset[str]]` mapping each input
            column to the set of category values kept as themselves.
        category_counts_: `dict[str, dict[str, int]]` mapping each input
            column to the full observed value-count mapping, including rare
            categories, so a report can print them alongside the frequent
            ones.
        n_rows_fit_: The number of rows the transformer was fitted on.
        feature_names_in_: The fitted column names, in fit order.
        n_features_in_: `len(feature_names_in_)`.
    """

    def __init__(
        self,
        threshold: float = RARE_CATEGORY_THRESHOLD,
        other_label: str = OTHER_CATEGORY,
    ) -> None:
        """Configure the grouper.

        Args:
            threshold: Minimum share of fitted rows a category must reach to be
                kept as itself. A category whose share is strictly below this
                value is replaced by `other_label`. Defaults to
                `RARE_CATEGORY_THRESHOLD` (1 percent).
            other_label: Replacement label for rare-in-fit and unseen-in-fit
                categories. Defaults to `OTHER_CATEGORY` ("other").
        """
        self.threshold = threshold
        self.other_label = other_label

    def fit(self, X: pd.DataFrame, y: object = None) -> "RareCategoryGrouper":
        """Learn the frequent-category set for every column of `X`.

        Args:
            X: The cleaned training partition's categorical columns.
            y: Ignored. Present for scikit-learn `Pipeline` compatibility.

        Returns:
            `self`, fitted.

        Raises:
            TypeError: `X` is not a `pandas.DataFrame`.
            ValueError: `threshold` is outside `[0.0, 1.0)`, `X` is empty, or
                `other_label` already appears among the fitted frequent
                categories of any column.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"RareCategoryGrouper.fit expects a pandas DataFrame, got {type(X).__name__}."
            )
        if not (0.0 <= self.threshold < 1.0):
            raise ValueError(f"threshold must be in [0.0, 1.0), got {self.threshold!r}.")
        n_rows = len(X)
        if n_rows == 0:
            raise ValueError("RareCategoryGrouper cannot be fitted on an empty DataFrame.")

        frequent_categories: dict[str, frozenset[str]] = {}
        category_counts: dict[str, dict[str, int]] = {}
        for column in X.columns:
            counts = X[column].value_counts()
            category_counts[column] = {str(key): int(value) for key, value in counts.items()}
            shares = counts / n_rows
            # Boundary rule: `>=` keeps -- a category at exactly `threshold` is
            # kept as itself; strictly below is grouped (see design's
            # "Controlling the 1% threshold precisely" fixture table).
            frequent = frozenset(str(value) for value in counts.index[shares >= self.threshold])
            if self.other_label in frequent:
                raise ValueError(
                    f"other_label {self.other_label!r} already appears as a fitted "
                    f"frequent category in column {column!r}; choose a different "
                    "other_label."
                )
            frequent_categories[column] = frequent

        self.n_rows_fit_ = n_rows
        self.frequent_categories_ = frequent_categories
        self.category_counts_ = category_counts
        self.feature_names_in_ = np.asarray(list(X.columns), dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Map every rare-in-train or unseen-in-train value to `other_label`.

        Rare-in-train and unseen-in-train values travel the identical code
        path: a value absent from `frequent_categories_[column]` is absent
        for the same reason whether it was rare in training or never seen at
        all, so there is no `if unseen` branch that could diverge from the
        rare-value path under a later edit.

        Args:
            X: A frame carrying exactly `feature_names_in_`, in that order.

        Returns:
            A new `DataFrame`, same index and column order as `X`, with every
            out-of-set value replaced by `other_label` and cast to
            `STRING_DTYPE`. Never mutates `X`.

        Raises:
            ValueError: `X`'s columns do not match `feature_names_in_`.
        """
        check_is_fitted(self)
        if tuple(X.columns) != tuple(self.feature_names_in_):
            raise ValueError(
                "RareCategoryGrouper.transform received columns "
                f"{tuple(X.columns)!r}, expected {tuple(self.feature_names_in_)!r}."
            )
        data = {}
        for column in X.columns:
            frequent = self.frequent_categories_[column]
            data[column] = (
                X[column].where(X[column].isin(frequent), self.other_label).astype(STRING_DTYPE)
            )
        return pd.DataFrame(data, index=X.index, columns=list(X.columns))

    def get_feature_names_out(self, input_features: "ArrayLike | None" = None) -> np.ndarray:
        """Return the output feature names -- one-to-one with the input.

        Implemented without the private
        `sklearn.utils.validation._check_feature_names_in` helper, so a minor
        scikit-learn version bump that removes or renames it cannot silently
        break the build.

        Args:
            input_features: When given, MUST equal `feature_names_in_` and is
                echoed back unchanged. When omitted, `feature_names_in_` is
                returned.

        Returns:
            The output column names as an object-dtype numpy array.

        Raises:
            ValueError: `input_features` is given and does not match
                `feature_names_in_`.
        """
        check_is_fitted(self)
        if input_features is None:
            return np.asarray(self.feature_names_in_, dtype=object)
        candidate = np.asarray(list(input_features), dtype=object)
        if tuple(candidate) != tuple(self.feature_names_in_):
            raise ValueError(
                f"input_features {tuple(candidate)!r} does not match the fitted "
                f"feature_names_in_ {tuple(self.feature_names_in_)!r}."
            )
        return candidate
