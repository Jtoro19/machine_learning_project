#!/usr/bin/env python
"""Stratified-subsample clustering comparison: KMeans vs Agglomerative on a PCA(2D)
embedding, scored with silhouette and adjusted Rand index against the target.

Parametrized, dataset-agnostic CLI. See `skills/clustering-reduction/SKILL.md` for usage
and `openspec/changes/skills/design.md` Decision 5 for the exact algorithm.

Ordering (round-3 review blocking finding 2): a recognised UNSW-NB15 raw input is first
COLUMN-cleaned (`skills/_shared/common.clean_dataset_if_unsw`: drop the fixed unused
columns, normalize `service`), THEN stratified-subsampled, THEN `train_test_split` runs
on the subsample, THEN exact-duplicate rows are dropped from the resulting TRAIN split
only (`common.split_and_dedupe`) -- never before the split, and never from the held-out
split, per `nids.cleaning.drop_train_duplicates`'s own "training partition only"
contract. The preprocessing pipeline is fit on that train split only, in line with
AGENTS.md's split-first rule, even though this skill fits no supervised model: the rule
governs the OPERATION (fitting an imputer/encoder/scaler), not the model that follows it.
PCA and both clustering algorithms are then fit on the transformed train split.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import common  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    """Run the clustering-reduction comparison end-to-end.

    Args:
        argv: Command-line arguments, excluding the program name. Defaults to
            `sys.argv[1:]` when None (argparse's own default).
    """
    common.ensure_repo_root_on_syspath()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from sklearn.preprocessing import LabelEncoder

    from nids.sampling import stratified_subsample

    parser = common.build_arg_parser(
        "Stratified-subsample clustering comparison (KMeans vs Agglomerative) on a PCA "
        "embedding."
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help=(
            "Fixed cluster count for both algorithms. When omitted, chosen by an "
            "unsupervised silhouette sweep over k=2..10 (never derived from the target)."
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

    subsample_result = stratified_subsample(
        df, stratify_by=args.target, max_rows=10_000, seed=args.seed
    )
    subsample = subsample_result.data

    pipeline, feature_columns_to_use = common.build_preprocessor_for(
        subsample, features, exclude, is_unsw_schema=is_unsw_schema
    )
    common.assert_no_leakage(
        feature_columns_to_use, args.target, exclude, df=subsample, is_unsw_schema=is_unsw_schema
    )

    train_df, _test_df = common.split_and_dedupe(
        subsample, args.target, test_size=0.3, seed=args.seed, is_unsw_schema=is_unsw_schema
    )
    y_train = train_df[args.target]
    X_train = train_df[feature_columns_to_use]
    common.assert_no_leakage(
        X_train.columns, args.target, exclude, df=subsample, is_unsw_schema=is_unsw_schema
    )

    pipeline.fit(X_train)
    X_t = pipeline.transform(X_train)
    n_components = min(2, X_t.shape[1])
    pca = PCA(n_components=n_components, random_state=args.seed)
    X_pca = pca.fit_transform(X_t)

    y_encoded = LabelEncoder().fit_transform(y_train.astype(str))

    if args.n_clusters is not None:
        n_clusters, n_clusters_method = args.n_clusters, "user_specified"
    else:
        n_clusters, n_clusters_method = common.choose_n_clusters(X_pca, args.seed)
    kmeans = KMeans(n_clusters=n_clusters, random_state=args.seed, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_pca)

    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative_labels = agglomerative.fit_predict(X_pca)

    results = []
    for algo_name, labels in (("kmeans", kmeans_labels), ("agglomerative", agglomerative_labels)):
        sil = float(silhouette_score(X_pca, labels)) if len(set(labels)) > 1 else float("nan")
        ari = float(adjusted_rand_score(y_encoded, labels))
        results.append({"algorithm": algo_name, "silhouette_score": sil, "adjusted_rand_index": ari})

    comparison = pd.DataFrame(results)
    best_row = comparison.sort_values("silhouette_score", ascending=False, kind="stable").iloc[0]

    with common.resolve_results_writer(args.output) as writer:
        writer.add_table(
            subsample_result.to_frame(),
            name="subsample_allocation",
            title="Stratified subsample class allocation",
            description="Per-class allocation realized by nids.sampling.stratified_subsample.",
        )
        writer.add_table(
            comparison,
            name="clustering_comparison",
            title="Clustering algorithm comparison",
            description="Silhouette score and adjusted Rand index per clustering algorithm.",
        )

        if n_components >= 2:
            for algo_name, labels, fig_name in (
                ("KMeans", kmeans_labels, "clusters_kmeans"),
                ("Agglomerative", agglomerative_labels, "clusters_agglomerative"),
            ):
                fig, ax = plt.subplots(figsize=(6, 5))
                scatter = ax.scatter(
                    X_pca[:, 0], X_pca[:, 1], c=labels, s=8, cmap="tab10", alpha=0.7
                )
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")
                ax.set_title(f"{algo_name} clusters on PCA(2D) embedding")
                fig.colorbar(scatter, ax=ax, label="cluster")
                writer.add_figure(
                    fig,
                    name=fig_name,
                    title=f"{algo_name} clusters",
                    description=f"PCA(2D) embedding of the stratified subsample, colored by {algo_name} cluster id.",
                )
                plt.close(fig)

        writer.add_metric(
            "best_silhouette_score",
            float(best_row["silhouette_score"]),
            "Highest silhouette score among compared clustering algorithms.",
        )
        writer.add_metric(
            "best_algorithm",
            str(best_row["algorithm"]),
            "Algorithm achieving the highest silhouette score.",
        )
        writer.add_metric(
            "n_clusters", int(n_clusters), "Cluster count used by both compared algorithms."
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
