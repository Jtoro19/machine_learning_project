#!/usr/bin/env python
"""EDA + dimensionality reduction (PCA) + clustering (KMeans) report.

Parametrized, dataset-agnostic CLI. See `skills/eda-reduction-clustering/SKILL.md` for
usage and `openspec/changes/skills/design.md` Decision 5 for the exact algorithm.

Ordering (round-3 review blocking finding 2): a recognised UNSW-NB15 raw input is first
COLUMN-cleaned (`skills/_shared/common.clean_dataset_if_unsw`: drop the fixed unused
columns, normalize `service`), THEN `train_test_split` runs, THEN exact-duplicate rows
are dropped from the resulting TRAIN split only (`common.split_and_dedupe`) -- never
before the split, and never from the held-out split, per `nids.cleaning.
drop_train_duplicates`'s own "training partition only" contract. The preprocessing
pipeline is fit on that train split only, in line with AGENTS.md's split-first rule, even
though this skill fits no supervised model: the rule governs the OPERATION (fitting an
imputer/encoder/scaler), not the model that follows it. PCA and KMeans are then fit on
the transformed train split.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import common  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Run the EDA/reduction/clustering report end-to-end.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]` when None (argparse's own default).
    """
    common.ensure_repo_root_on_syspath()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    parser = common.build_arg_parser(
        "EDA, PCA dimensionality reduction, and KMeans clustering report."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Fixed KMeans cluster count. When omitted, chosen by an unsupervised "
            "silhouette sweep over k=2..10 (never derived from the target)."
        ),
    )
    args = parser.parse_args(argv)

    exclude = common.resolve_exclude(args.exclude)
    raw_df = common.load_dataset(args.dataset)
    original_columns = list(raw_df.columns)
    df, is_unsw_schema = common.clean_dataset_if_unsw(raw_df)
    features = common.compute_features(
        df, args.target, exclude, original_columns=original_columns
    )

    pipeline, feature_columns_to_use = common.build_preprocessor_for(
        df, features, exclude, is_unsw_schema=is_unsw_schema
    )
    common.assert_no_leakage(
        feature_columns_to_use, args.target, exclude, df=df, is_unsw_schema=is_unsw_schema
    )

    y = df[args.target]

    train_df, _test_df = common.split_and_dedupe(
        df, args.target, test_size=0.3, seed=args.seed, is_unsw_schema=is_unsw_schema
    )
    y_train = train_df[args.target]
    X_train = train_df[feature_columns_to_use]
    common.assert_no_leakage(
        X_train.columns, args.target, exclude, df=df, is_unsw_schema=is_unsw_schema
    )

    pipeline.fit(X_train)
    X_t = pipeline.transform(X_train)

    n_components = min(2, X_t.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    X_pca = pca.fit_transform(X_t)

    if args.n_clusters is not None:
        n_clusters, n_clusters_method = args.n_clusters, "user_specified"
    else:
        n_clusters, n_clusters_method = common.choose_n_clusters(X_pca, args.seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    clusters = kmeans.fit_predict(X_pca)

    with common.resolve_results_writer(args.output) as writer:
        # Descriptive-only table over the full input (not a fit operation, so it is not
        # subject to the split-first rule): every feature column actually used by the
        # pipeline (`feature_columns_to_use`, never the pre-UNSW-routing `features`,
        # which can still carry columns the UNSW routing drops, e.g. `id`).
        summary = df[feature_columns_to_use].describe(include="all").transpose().reset_index()
        summary = summary.rename(columns={"index": "column"})
        writer.add_table(
            summary,
            name="summary_statistics",
            title="Feature summary statistics",
            description="describe() over every feature column, before preprocessing.",
        )

        target_dist = (
            y.value_counts()
            .rename_axis(args.target)
            .reset_index(name="count")
            .sort_values(args.target, kind="stable")
        )
        writer.add_table(
            target_dist,
            name="target_distribution",
            title="Target class distribution",
            description=f"Row count per value of the target column ({args.target}).",
        )

        explained = pd.DataFrame(
            {
                "component": [f"PC{i + 1}" for i in range(n_components)],
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )
        writer.add_table(
            explained,
            name="pca_explained_variance",
            title="PCA explained variance ratio",
            description="Explained variance ratio per principal component.",
        )

        crosstab = pd.crosstab(
            pd.Series(clusters, name="cluster"),
            y_train.reset_index(drop=True).rename(args.target),
        ).reset_index()
        writer.add_table(
            crosstab,
            name="cluster_target_crosstab",
            title="Cluster vs target crosstab",
            description="Row count per (KMeans cluster, target value) pair.",
        )

        if n_components >= 2:
            fig, ax = plt.subplots(figsize=(6, 5))
            y_str = y_train.astype(str).reset_index(drop=True)
            for label in sorted(y_str.unique()):
                mask = y_str.to_numpy() == label
                ax.scatter(X_pca[mask, 0], X_pca[mask, 1], s=8, label=str(label), alpha=0.7)
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title(f"PCA projection colored by {args.target}")
            ax.legend(fontsize="x-small", markerscale=2)
            writer.add_figure(
                fig,
                name="pca_scatter_by_target",
                title="PCA scatter colored by target",
                description="First two principal components, points colored by target value.",
            )
            plt.close(fig)

            fig2, ax2 = plt.subplots(figsize=(6, 5))
            scatter = ax2.scatter(
                X_pca[:, 0], X_pca[:, 1], c=clusters, s=8, cmap="tab10", alpha=0.7
            )
            ax2.set_xlabel("PC1")
            ax2.set_ylabel("PC2")
            ax2.set_title("PCA projection colored by KMeans cluster")
            fig2.colorbar(scatter, ax=ax2, label="cluster")
            writer.add_figure(
                fig2,
                name="pca_scatter_by_cluster",
                title="PCA scatter colored by cluster",
                description="First two principal components, points colored by KMeans cluster id.",
            )
            plt.close(fig2)

        writer.add_metric(
            "pca_explained_variance_ratio_sum",
            float(sum(pca.explained_variance_ratio_)),
            "Sum of explained variance ratio across the retained principal components.",
        )
        writer.add_metric(
            "n_clusters", int(n_clusters), "Number of KMeans clusters used."
        )
        writer.add_metric(
            "n_clusters_selection_method",
            n_clusters_method,
            "How n_clusters was chosen: 'user_specified' (--n-clusters), "
            "'silhouette_sweep' (unsupervised sweep over k=2..10), or "
            "'fixed_minimum' (dataset too small/degenerate for the sweep).",
        )


if __name__ == "__main__":
    main()
