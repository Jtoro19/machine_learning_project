#!/usr/bin/env python
"""Supervised classification report: RandomForest, macro F1 + balanced accuracy, with an
automatic with/without-TTL-shortcut variant when the TTL columns are present in the
feature set (AGENTS.md: "Report supervised results with and without sttl and
ct_state_ttl").

Parametrized, dataset-agnostic CLI. See `skills/classification/SKILL.md` for usage and
`openspec/changes/skills/design.md` Decision 5 for the exact algorithm.

Ordering (round-3 review blocking finding 2): a recognised UNSW-NB15 raw input is first
COLUMN-cleaned (`skills/_shared/common.clean_dataset_if_unsw`: drop the fixed unused
columns, normalize `service`), THEN `train_test_split` runs, THEN exact-duplicate rows
are dropped from the resulting TRAIN split only (`common.split_and_dedupe`) -- never
before the split, and never from the held-out split, per `nids.cleaning.
drop_train_duplicates`'s own "training partition only" contract.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import common  # noqa: E402

# Round-5 review finding 1: this module previously re-typed the TTL shortcut pair
# as a local module-level literal, shadowing `nids.columns.TTL_SHORTCUT_COLUMNS` --
# the single place `columns.py`'s own module
# docstring says a raw/derived column name is written in this package. Imported
# directly instead; `common.py`'s own module-scope `ensure_repo_root_on_syspath()`
# call (round-5 finding 2) already put `src/` on `sys.path` by the time the
# `from _shared import common` import above finished, so `nids` is importable here
# regardless of whether the package is separately installed.
from nids.columns import TTL_SHORTCUT_COLUMNS  # noqa: E402

if TYPE_CHECKING:
    import pandas as pd

    from nids.results import ResultsWriter


def _ttl_shortcut_state(features: list[str]) -> str:
    """Classify how much of `TTL_SHORTCUT_COLUMNS` is present in `features`.

    Round-5 review finding 4: when the TTL pair was absent from `features`, only
    the `"full"` variant ran and nothing in `manifest.json` recorded whether the
    testbed shortcut was in play at all -- a reader could not tell "shortcut
    included" apart from "shortcut already excluded upstream", and a partial
    exclusion (one of the pair present, one absent) was equally silent.

    Args:
        features: The resolved feature column names for this run (before any
            TTL-specific exclusion).

    Returns:
        `"both"` when both `sttl` and `ct_state_ttl` are present (the
        `without_ttl` variant runs), `"none"` when neither is present, or
        `"partial"` when exactly one of the pair is present.
    """
    present = set(TTL_SHORTCUT_COLUMNS) & set(features)
    if len(present) == len(TTL_SHORTCUT_COLUMNS):
        return "both"
    if not present:
        return "none"
    return "partial"


def _run_variant(
    variant_name: str,
    df: "pd.DataFrame",
    features: list[str],
    exclude: list[str],
    target: str,
    seed: int,
    writer: "ResultsWriter",
    is_unsw_schema: bool = False,
) -> tuple[float, float]:
    """Fit, evaluate, and report one classification variant.

    Args:
        variant_name: `"full"` or `"without_ttl"` — used to suffix table rows,
            figure names, and metric names for the second variant.
        df: The full loaded dataset (already cleaned via
            `common.clean_dataset_if_unsw` when it matched the UNSW schema).
        features: The feature column names for this variant.
        exclude: The resolved `--exclude` list (already reflects this variant).
        target: The target column name.
        seed: Random seed for the split and the classifier.
        writer: The open `ResultsWriter`.
        is_unsw_schema: The UNSW-schema routing decision computed once in
            `main` before cleaning (see `common.clean_dataset_if_unsw`).

    Returns:
        `(macro_f1, balanced_accuracy)` for this variant.
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        ConfusionMatrixDisplay,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
    )

    pipeline, feature_columns_to_use = common.build_preprocessor_for(
        df, features, exclude, is_unsw_schema=is_unsw_schema
    )
    common.assert_no_leakage(
        feature_columns_to_use, target, exclude, df=df, is_unsw_schema=is_unsw_schema
    )

    train_df, test_df = common.split_and_dedupe(
        df, target, test_size=0.3, seed=seed, is_unsw_schema=is_unsw_schema
    )
    y_train = train_df[target]
    X_train = train_df[feature_columns_to_use]
    y_test = test_df[target]
    X_test = test_df[feature_columns_to_use]
    common.assert_no_leakage(X_train.columns, target, exclude, df=df, is_unsw_schema=is_unsw_schema)

    X_train_t = pipeline.fit_transform(X_train)
    X_test_t = pipeline.transform(X_test)

    clf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1)
    clf.fit(X_train_t, y_train)
    predictions = clf.predict(X_test_t)

    macro_f1 = float(f1_score(y_test, predictions, average="macro"))
    balanced_accuracy = float(balanced_accuracy_score(y_test, predictions))

    report_dict = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index()
    report_df = report_df.rename(columns={"index": "class"})
    report_df.insert(0, "variant", variant_name)
    writer.add_table(
        report_df,
        name=f"classification_report_{variant_name}",
        title=f"Classification report ({variant_name})",
        description="Per-class precision/recall/f1-score and support, plus macro/weighted averages.",
    )

    labels_sorted = sorted(y_test.astype(str).unique())
    cm = confusion_matrix(y_test.astype(str), pd.Series(predictions).astype(str), labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted).plot(
        ax=ax, xticks_rotation="vertical", colorbar=False
    )
    ax.set_title(f"Confusion matrix ({variant_name})")
    writer.add_figure(
        fig,
        name=f"confusion_matrix_{variant_name}",
        title=f"Confusion matrix ({variant_name})",
        description="Confusion matrix (raw counts, not row-normalized) on the held-out test split.",
    )
    plt.close(fig)

    return macro_f1, balanced_accuracy


def main(argv: list[str] | None = None) -> None:
    """Run the classification report end-to-end.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]` when None (argparse's own default).
    """
    common.ensure_repo_root_on_syspath()
    import matplotlib

    matplotlib.use("Agg")

    parser = common.build_arg_parser(
        "Supervised classification report: RandomForest, macro F1 + balanced accuracy."
    )
    args = parser.parse_args(argv)

    exclude = common.resolve_exclude(args.exclude)
    raw_df = common.load_dataset(args.dataset)
    original_columns = list(raw_df.columns)
    df, is_unsw_schema = common.clean_dataset_if_unsw(raw_df)
    features = common.compute_features(
        df, args.target, exclude, original_columns=original_columns
    )

    with common.resolve_results_writer(args.output) as writer:
        macro_f1, balanced_accuracy = _run_variant(
            "full", df, features, exclude, args.target, args.seed, writer, is_unsw_schema
        )
        writer.add_metric("macro_f1", macro_f1, "Macro-averaged F1 score on the held-out test split.")
        writer.add_metric(
            "balanced_accuracy", balanced_accuracy, "Balanced accuracy on the held-out test split."
        )

        ttl_state = _ttl_shortcut_state(features)
        writer.add_metric(
            "ttl_shortcut_columns_present",
            ttl_state,
            "Whether the sttl/ct_state_ttl testbed-shortcut pair (AGENTS.md: 'Report "
            "supervised results with and without sttl and ct_state_ttl') is present in "
            "this run's feature set: 'both' (the without_ttl variant below also ran), "
            "'none' (already excluded upstream, e.g. via --exclude, so there is nothing "
            "to compare), or 'partial' (only one of the pair is present).",
        )
        if ttl_state == "both":
            writer.add_note(
                "ttl_shortcut_comparison",
                "Both sttl and ct_state_ttl are present in the feature set, so this run "
                "reports two variants: 'full' (with the TTL shortcut pair) and "
                "'without_ttl' (with sttl/ct_state_ttl excluded), per AGENTS.md's "
                "paired-reporting rule for the known testbed shortcut.",
            )
            without_ttl_exclude = sorted(set(exclude) | set(TTL_SHORTCUT_COLUMNS))
            without_ttl_features = [c for c in features if c not in TTL_SHORTCUT_COLUMNS]
            macro_f1_wo, balanced_accuracy_wo = _run_variant(
                "without_ttl",
                df,
                without_ttl_features,
                without_ttl_exclude,
                args.target,
                args.seed,
                writer,
                is_unsw_schema,
            )
            writer.add_metric(
                "macro_f1_without_ttl",
                macro_f1_wo,
                "Macro-averaged F1 score with sttl/ct_state_ttl excluded from features.",
            )
            writer.add_metric(
                "balanced_accuracy_without_ttl",
                balanced_accuracy_wo,
                "Balanced accuracy with sttl/ct_state_ttl excluded from features.",
            )
        else:
            reason = (
                "neither sttl nor ct_state_ttl is present in the feature set (already "
                "excluded upstream, e.g. via --exclude)"
                if ttl_state == "none"
                else "only one of sttl/ct_state_ttl is present in the feature set (a "
                "partial exclusion)"
            )
            writer.add_note(
                "ttl_shortcut_comparison",
                f"Only the 'full' variant ran this run: {reason}, so no "
                "with/without-TTL-shortcut comparison was possible. AGENTS.md's "
                "paired-reporting rule for the known testbed shortcut applies only "
                "when both sttl and ct_state_ttl are present.",
            )


if __name__ == "__main__":
    main()
