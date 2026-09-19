"""Dataset-agnostic tabular preprocessing helpers shared by the three
`skills/` scripts (`skills/eda-reduction-clustering`, `skills/clustering-reduction`,
`skills/classification`).

Moved here from `skills/_shared/common.py` (round-3 review, standards finding 3):
AGENTS.md states "Shared preprocessing lives in `src/`." `is_unsw_raw_schema`,
`clean_dataset_if_unsw`, `build_preprocessor_for`, `numeric_and_categorical_columns`,
and `compute_features` are dataset column/dtype-shaping logic, not CLI/argparse/IO
plumbing, so they live here; `skills/_shared/common.py` keeps only argument parsing,
file loading, and `nids.results.ResultsWriter` wiring, importing this module for
everything else. `split_and_dedupe`/`dedupe_train_split` are new in this round (round-3
blocking finding 2 -- see their docstrings).

The only UNSW-NB15-specific literal in this module is the set comparison in
`is_unsw_raw_schema` against the frozen `nids.columns.RAW_COLUMNS` (design Decision 2 --
"detect, never assume"); every other function here is fully dataset-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from sklearn.model_selection import train_test_split

from nids.columns import DROPPED_COLUMNS, TARGET_COLUMNS, TTL_SHORTCUT_COLUMNS, feature_columns
from nids.preprocessing import build_generic_preprocessor, build_preprocessor

if TYPE_CHECKING:
    from sklearn.pipeline import Pipeline


def assert_no_leakage(
    columns: "list[str] | pd.Index",
    target: str,
    exclude: list[str],
    df: "pd.DataFrame | None" = None,
    is_unsw_schema: bool | None = None,
) -> None:
    """Raise if `target` or any `exclude` entry is present in `columns`.

    Called immediately before `pipeline.fit`/`pipeline.transform` as a second,
    defensive check beyond `compute_features`'s.

    Round-4 review standards finding 3: moved here from `skills/_shared/common.py`
    (AGENTS.md: "Shared preprocessing lives in `src/`") -- this is the single
    function enforcing the project's hardest data rule (no target/leak-prone column
    ever reaches a fit/transform call), the same category of UNSW-specific,
    column-shaping logic every other function in this module already covers.
    `skills/_shared/common.py` re-exports it under the same name for its call sites.

    When the UNSW-NB15 raw schema applies, this also asserts that no member of
    `nids.columns.TARGET_COLUMNS` (`attack_cat`, `label`) and no member of
    `nids.columns.DROPPED_COLUMNS` (`id`, `stcpb`, `dtcpb`, `is_ftp_login`) is
    present in `columns` -- regardless of whether that column happens to be
    `target` or a member of `exclude`. This closes the leak where, for
    example, `--target attack_cat --exclude id` let `label` (a perfect
    `attack_cat != "Normal"` predictor) reach the feature matrix because
    `label` was neither `target` nor an excluded column.

    Args:
        columns: The columns about to be fit/transformed.
        target: The target column name.
        exclude: The excluded column names.
        df: The dataset `columns` was derived from, used only to detect the
            UNSW-NB15 schema when `is_unsw_schema` is not supplied directly.
            When both are omitted, only the basic `target`/`exclude` checks
            run.
        is_unsw_schema: The UNSW-schema routing decision, when the caller
            already knows it (for example because it computed it on the raw
            frame before cleaning, and cleaning removes `DROPPED_COLUMNS`,
            which would make a fresh `is_unsw_raw_schema(df)` call on the
            already-cleaned `df` always return False). When omitted and `df`
            is given, this is detected via `is_unsw_raw_schema(df)`.

    Raises:
        ValueError: `target` or any `exclude` entry is present in `columns`,
            or (when the UNSW schema applies) any UNSW target or dropped
            column is present in `columns`.
    """
    column_set = set(columns)
    if target in column_set:
        raise ValueError(f"target column {target!r} present at fit/transform time.")
    leaked = column_set & set(exclude)
    if leaked:
        raise ValueError(f"excluded column(s) present at fit/transform time: {leaked!r}.")

    unsw = is_unsw_schema if is_unsw_schema is not None else (
        df is not None and is_unsw_raw_schema(df)
    )
    if unsw:
        forbidden = set(TARGET_COLUMNS) | set(DROPPED_COLUMNS)
        leaked_unsw = column_set & forbidden
        if leaked_unsw:
            raise ValueError(
                f"UNSW-NB15 target/dropped column(s) present at fit/transform time: "
                f"{leaked_unsw!r}."
            )


def is_unsw_raw_schema(df: pd.DataFrame) -> bool:
    """Detect whether `df`'s columns match the UNSW-NB15 raw schema.

    The only UNSW-specific reference in this module: a set-subset comparison against
    the frozen `nids.columns.RAW_COLUMNS`, never a hardcoded re-typed list (design
    Decision 2 -- "detect, never assume").

    Args:
        df: The loaded dataset.

    Returns:
        True if every column in `nids.columns.RAW_COLUMNS` is present in `df`.
    """
    from nids.columns import RAW_COLUMNS

    return set(RAW_COLUMNS) <= set(df.columns)


def resolve_stratify(y: pd.Series) -> pd.Series | None:
    """Return `y` when safe to stratify a `train_test_split` on, else `None`.

    `train_test_split(..., stratify=y)` raises `ValueError` when any class in
    `y` has fewer than 2 members (there is no row left for one side of the
    split). Rather than let a real dataset's naturally rare class crash the
    run, callers use this helper to fall back to an unstratified split for
    that case, and a stratified one otherwise.

    Args:
        y: The target series about to be split.

    Returns:
        `y` unchanged if every class has at least 2 rows, else `None`.
    """
    counts = y.value_counts()
    if len(counts) == 0 or counts.min() < 2:
        return None
    return y


def clean_dataset_if_unsw(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Route `df` through the project's own COLUMN-level cleaning steps when it
    matches the UNSW-NB15 raw schema, before any preprocessing pipeline is fit on it.

    `nids.transformers.RareCategoryGrouper` MUST be fitted on the CLEANED
    training partition, never the raw one (see its module docstring):
    reuses the project's own `nids.cleaning` step functions and `nids.loading`'s
    dtype declarations directly -- never re-implements them -- applying the
    two ordered COLUMN-shaping steps that apply to a single already-loaded
    frame regardless of any later split: drop the fixed unused columns
    (`nids.cleaning.drop_unused_columns`) and normalize `service` `"-"` ->
    `"none"` (`nids.cleaning.normalize_service`).

    Round-3 blocking finding 2: this function no longer runs
    `nids.cleaning.drop_train_duplicates` -- that step's own docstring
    restricts it to "the training partition only", so it MUST run after a
    `train_test_split`, on the resulting train split alone, never on the
    full frame before splitting (deduplicating first would also remove
    duplicate rows that happened to land in the held-out split, silently
    reshaping it). Callers use `split_and_dedupe` (below) for the correct
    split-then-dedupe ordering.

    Args:
        df: The dataset as returned by `skills._shared.common.load_dataset`.

    Returns:
        `(cleaned_or_original_df, is_unsw_schema)`. `is_unsw_schema` reflects
        detection on `df` BEFORE cleaning -- cleaning removes `DROPPED_COLUMNS`,
        so a caller that re-ran `is_unsw_raw_schema` on the returned frame
        would always get `False`. Every caller that needs the UNSW routing
        decision downstream (`build_preprocessor_for`, `split_and_dedupe`,
        `skills._shared.common.assert_no_leakage`) MUST be given this returned
        flag explicitly rather than re-detecting on the returned frame.
        Non-UNSW datasets are returned unchanged, with `is_unsw_schema=False`.
    """
    is_unsw = is_unsw_raw_schema(df)
    if not is_unsw:
        return df, False

    from nids.cleaning import drop_unused_columns, normalize_service
    from nids.loading import STRING_COLUMNS, STRING_DTYPE

    typed = df.astype(
        {column: STRING_DTYPE for column in STRING_COLUMNS if column in df.columns}
    )
    dropped, _ = drop_unused_columns(typed)
    normalized, _ = normalize_service(dropped, partition="skill_dataset")
    return normalized, True


def dedupe_train_split(train_df: pd.DataFrame, *, is_unsw_schema: bool) -> pd.DataFrame:
    """Drop exact-duplicate rows from an ALREADY-SPLIT training partition only.

    MUST be called after `train_test_split`, on the resulting train split alone --
    never before the split, and never on the held-out/test split.
    `nids.cleaning.drop_train_duplicates`'s own docstring restricts it to "the
    training partition only" (round-3 blocking finding 2): deduplicating before
    the split would also remove duplicate rows that happened to land in the
    held-out split, silently reshaping it and breaking the "held-out data is
    never touched" guarantee.

    Args:
        train_df: The TRAIN partition returned by `train_test_split`, already
            column-cleaned via `clean_dataset_if_unsw` (`drop_unused_columns` +
            `normalize_service` already applied) but not yet deduplicated.
        is_unsw_schema: Whether the UNSW-NB15 raw schema was detected (see
            `clean_dataset_if_unsw`). Deduplication is UNSW-schema-specific --
            `nids.cleaning.drop_train_duplicates` keys on the full
            `nids.columns.KEPT_COLUMNS` set (including `state`), which a
            generic, non-UNSW dataset does not necessarily carry -- so a
            non-UNSW `train_df` is returned unchanged.

    Returns:
        `train_df` with exact-duplicate rows removed when `is_unsw_schema` is
        True, else `train_df` unchanged.
    """
    if not is_unsw_schema:
        return train_df

    from nids.cleaning import drop_train_duplicates

    deduped, _ = drop_train_duplicates(train_df)
    return deduped


def split_and_dedupe(
    df: pd.DataFrame,
    target: str,
    *,
    test_size: float,
    seed: int,
    is_unsw_schema: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the FULL frame first, then deduplicate the TRAIN split only.

    The single call site every `skills/` script MUST use instead of calling
    `train_test_split` and `dedupe_train_split` separately, so the ordering
    (split, THEN dedupe train only -- round-3 blocking finding 2) cannot drift
    apart at a call site again.

    Args:
        df: The column-cleaned (via `clean_dataset_if_unsw`), NOT-YET-DEDUPLICATED
            full dataset (or subsample, for `clustering-reduction`), still carrying
            every column (including `target` and anything later excluded), so the
            UNSW dedup key (`nids.columns.KEPT_COLUMNS`, which includes `state`) is
            available.
        target: The target column name, used only to resolve stratification.
        test_size: Forwarded to `train_test_split` (0.3 in every current skill).
        seed: Forwarded to `train_test_split` as `random_state`.
        is_unsw_schema: Forwarded to `dedupe_train_split`.

    Returns:
        `(train_df, test_df)`, both with a fresh `RangeIndex`. `train_df` has
        exact-duplicate rows removed when `is_unsw_schema` is True; `test_df` is
        returned exactly as `train_test_split` produced it -- never deduplicated,
        never otherwise touched.
    """
    stratify = resolve_stratify(df[target])
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=stratify
    )
    train_df = dedupe_train_split(train_df, is_unsw_schema=is_unsw_schema)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def compute_features(
    df: pd.DataFrame,
    target: str,
    exclude: list[str],
    original_columns: "list[str] | pd.Index | None" = None,
) -> list[str]:
    """Return every column of `df` except `target` and `exclude`, raising on any leak.

    Args:
        df: The full loaded dataset (the CLEANED frame, when the caller already
            ran it through `clean_dataset_if_unsw`).
        target: The target column name.
        exclude: Additional columns to exclude from the feature set.
        original_columns: The dataset's column names BEFORE any cleaning step
            removed some of them (for example `clean_dataset_if_unsw` dropping
            `nids.columns.DROPPED_COLUMNS`). When given, `--exclude` names are
            validated against this PRE-CLEANING set instead of `df.columns`
            (round-3 blocking finding 1): this tolerates an `--exclude` naming
            a column the project's own cleaning already removed (for example
            `--exclude id`, since `id` is a member of `DROPPED_COLUMNS`), while
            a genuine typo (a name absent from the pre-cleaning set too) still
            raises. When omitted, `df.columns` is used for this validation --
            the previous behavior, for callers that pass an already-loaded,
            never-cleaned frame directly.

    Returns:
        The feature column names, in `df`'s original column order.

    Raises:
        ValueError: `target` is not a column of `df`, an `--exclude` entry is
            not a column of `original_columns` (or `df.columns` when
            `original_columns` is omitted) -- a guard against a silently-ignored
            typo, since a non-existent exclude name would otherwise never
            remove anything -- or the computed feature set still contains
            `target`.
    """
    if target not in df.columns:
        raise ValueError(f"target column {target!r} not found in dataset columns.")
    known_columns = set(original_columns) if original_columns is not None else set(df.columns)
    missing_excluded = [column for column in exclude if column not in known_columns]
    if missing_excluded:
        raise ValueError(
            f"--exclude named column(s) not present in dataset columns: {missing_excluded!r}."
        )
    excluded_set = set(exclude) | {target}
    features = [column for column in df.columns if column not in excluded_set]
    if target in features:
        raise ValueError(f"target column {target!r} leaked into features.")
    return features


def numeric_and_categorical_columns(
    df: pd.DataFrame, features: list[str]
) -> tuple[list[str], list[str]]:
    """Split `features` into numeric and categorical column names by detected dtype.

    Args:
        df: The dataset `features` was derived from.
        features: The feature column names to split.

    Returns:
        `(numeric_columns, categorical_columns)`, each in `features` order.
    """
    numeric_df = df[features].select_dtypes(include="number")
    numeric = [column for column in features if column in numeric_df.columns]
    categorical = [column for column in features if column not in numeric_df.columns]
    return numeric, categorical


def build_preprocessor_for(
    df: pd.DataFrame,
    features: list[str],
    exclude: list[str],
    is_unsw_schema: bool | None = None,
) -> tuple["Pipeline", list[str]]:
    """Route to the frozen UNSW pipeline or the generic fallback (design Decision 2).

    Routing is keyed on schema detection alone, never on the shape of `exclude`.
    Whenever the UNSW schema applies, `nids.columns.DROPPED_COLUMNS` and
    `nids.columns.TARGET_COLUMNS` are ALWAYS subtracted from the feature set,
    regardless of what `--exclude` was passed -- `exclude` can only ever remove
    MORE columns than that, never fewer.

    Args:
        df: The loaded dataset. When the caller already cleaned it via
            `clean_dataset_if_unsw`, this is the CLEANED frame (schema
            detection on it alone would return False, since cleaning removes
            `DROPPED_COLUMNS` -- see `is_unsw_schema`).
        features: The computed feature column names (already excludes `target` and
            `exclude`, per `compute_features`, but not necessarily `DROPPED_COLUMNS`/
            `TARGET_COLUMNS` when the caller's `exclude` didn't happen to name them).
        exclude: The resolved `--exclude` list.
        is_unsw_schema: The UNSW-schema routing decision, when the caller
            already knows it (typically the second element `clean_dataset_if_unsw`
            returned, computed on the RAW frame before cleaning). When omitted,
            this is detected via `is_unsw_raw_schema(df)` directly, matching the
            previous behavior for callers that pass an uncleaned frame.

    Returns:
        `(pipeline, feature_columns_to_use)`: an unfitted `Pipeline`, and the exact
        feature column list the caller must select from `df` before calling
        `pipeline.fit`/`.transform`. For the UNSW path this is either
        `nids.columns.feature_columns(include_ttl_features=...)` (when it exactly
        matches the schema-safe feature set) or that same safe feature set passed
        through the generic pipeline; for the non-UNSW path it is `features`
        unchanged.
    """
    unsw = is_unsw_schema if is_unsw_schema is not None else is_unsw_raw_schema(df)
    if unsw:
        always_drop = set(DROPPED_COLUMNS) | set(TARGET_COLUMNS)
        safe_features = [column for column in features if column not in always_drop]

        # Round-5 review finding 1: re-typed the TTL shortcut pair literal here instead
        # of importing `nids.columns.TTL_SHORTCUT_COLUMNS` -- the single place
        # `columns.py`'s own module docstring says a raw/derived column name is
        # written in this package.
        ttl_pair = set(TTL_SHORTCUT_COLUMNS)
        exclude_set = set(exclude)
        include_ttl = not ttl_pair.issubset(exclude_set)
        # A partial TTL exclusion (exactly one of the pair) is not supported by the
        # frozen pipeline's boolean toggle; fall back to the generic path for that
        # case rather than silently including/excluding the wrong column.
        partial_ttl_exclusion = len(exclude_set & ttl_pair) == 1

        candidate = feature_columns(include_ttl)
        if not partial_ttl_exclusion and set(safe_features) == set(candidate):
            return build_preprocessor(include_ttl_features=include_ttl), candidate

        numeric_columns, categorical_columns = numeric_and_categorical_columns(
            df, safe_features
        )
        return (
            build_generic_preprocessor(numeric_columns, categorical_columns),
            safe_features,
        )

    numeric_columns, categorical_columns = numeric_and_categorical_columns(df, features)
    return build_generic_preprocessor(numeric_columns, categorical_columns), features
