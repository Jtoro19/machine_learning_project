"""Tests for the three `skills/` notebook scripts.

Each script's `main(argv)` is called in-process against a small synthetic dataset
(never `data/raw/`) so tests run fast and give real tracebacks on failure. A
`@pytest.mark.slow` variant per script additionally exercises the real UNSW-NB15 raw
data, skipped when `data/raw/` is absent.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

from nids.paths import raw_data_dir, repo_root

REPO_ROOT = repo_root()
SKILLS_DIR = REPO_ROOT / "skills"

if str(SKILLS_DIR) not in sys.path:
    sys.path.insert(0, str(SKILLS_DIR))


def _load_script(module_name: str, relative_path: str) -> ModuleType:
    """Load a `skills/<dir>/<script>.py` module directly from its file path.

    The skill directories use hyphenated names (`eda-reduction-clustering`), which are
    not valid Python package identifiers, so each script is loaded by explicit file
    path via `importlib.util` rather than a normal `import` statement.

    Args:
        module_name: A unique name to register the loaded module under in
            `sys.modules`.
        relative_path: Path to the script, relative to the repository root.

    Returns:
        The executed module, exposing `main(argv)`.
    """
    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


eda_reduction_clustering = _load_script(
    "skill_eda_reduction_clustering",
    "skills/eda-reduction-clustering/eda_reduction_clustering.py",
)
clustering_reduction = _load_script(
    "skill_clustering_reduction",
    "skills/clustering-reduction/clustering_reduction.py",
)
classification = _load_script(
    "skill_classification",
    "skills/classification/classification.py",
)


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """A small synthetic CSV: numeric + categorical features, a multiclass target with
    a rare class, and an `id` column meant to be excluded.

    Returns:
        Path to the written `dataset.csv`.
    """
    rng = np.random.default_rng(42)
    n_rows = 300

    target = np.array(["A"] * 150 + ["B"] * 100 + ["C"] * 44 + ["D"] * 6)
    rng.shuffle(target)

    cat1_values = rng.choice(
        ["red", "blue", "green"], size=n_rows - 2, p=[0.5, 0.35, 0.15]
    )
    cat1 = np.concatenate([cat1_values, ["rare_color", "rare_color"]])
    rng.shuffle(cat1)

    frame = pd.DataFrame(
        {
            "id": np.arange(1, n_rows + 1),
            "feat_num1": rng.normal(loc=10.0, scale=3.0, size=n_rows),
            "feat_num2": rng.exponential(scale=2.0, size=n_rows),
            "feat_num3": rng.uniform(0.0, 100.0, size=n_rows),
            "feat_cat1": cat1,
            "feat_cat2": rng.choice(["x", "y"], size=n_rows),
            "target": target,
        }
    )
    path = tmp_path / "dataset.csv"
    frame.to_csv(path, index=False)
    return path


def _assert_valid_manifest(output_dir: Path) -> dict:
    """Assert `output_dir/manifest.json` exists, parses, and every artifact exists.

    Returns:
        The parsed manifest payload.
    """
    manifest_path = output_dir / "manifest.json"
    assert manifest_path.is_file(), f"manifest.json missing under {output_dir}"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("schema_version", "notebook_id", "generated_at", "tables", "figures", "metrics"):
        assert key in payload, f"manifest missing required key {key!r}"
    for entry in (*payload["tables"], *payload["figures"]):
        artifact_path = output_dir / entry["path"]
        assert artifact_path.is_file(), f"declared artifact missing: {artifact_path}"
    return payload


def _require_training_csv() -> Path:
    """Return the real UNSW-NB15 training partition path, skipping if absent.

    Deliberately requires `nids.loading.RAW_TRAIN_FILENAME` by exact name rather
    than falling back to "any CSV under `data/raw/`" — that fallback could silently
    select `UNSW_NB15_testing-set.csv`, which is exactly the unsplit-partition bug
    these `@pytest.mark.slow` tests exist to catch (review finding 1)."""
    from nids.loading import RAW_TRAIN_FILENAME

    path = raw_data_dir() / RAW_TRAIN_FILENAME
    if not path.is_file():
        pytest.skip(f"{RAW_TRAIN_FILENAME} not found under data/raw/")
    return path


def _assert_unsw_run_excludes_dropped_columns(
    dataset_path: Path, *, target: str, exclude: list[str]
) -> None:
    """Prove `id` (and every other `DROPPED_COLUMNS`/`TARGET_COLUMNS` member) is
    absent from the fitted feature names for a real-UNSW-NB15 run, using the exact
    same `load_dataset` -> `clean_dataset_if_unsw` -> `compute_features` ->
    `build_preprocessor_for` call sequence the skill scripts make internally (review
    finding 2: this is the root-cause proof, not a call rewritten to dodge the bug).

    `original_columns` is captured BEFORE cleaning and threaded into
    `compute_features`, exactly as each skill script's `main()` does (round-3
    blocking finding 1): this lets `exclude` name a column cleaning already
    dropped (for example `--exclude id`) without `compute_features` raising."""
    from _shared import common
    from nids.columns import DROPPED_COLUMNS, TARGET_COLUMNS

    raw_df = common.load_dataset(str(dataset_path))
    assert common.is_unsw_raw_schema(raw_df)
    original_columns = list(raw_df.columns)

    df, is_unsw_schema = common.clean_dataset_if_unsw(raw_df)
    assert is_unsw_schema

    features = common.compute_features(
        df, target=target, exclude=exclude, original_columns=original_columns
    )
    pipeline, feature_columns_to_use = common.build_preprocessor_for(
        df, features, exclude, is_unsw_schema=is_unsw_schema
    )
    common.assert_no_leakage(
        feature_columns_to_use, target, exclude, df=df, is_unsw_schema=is_unsw_schema
    )

    assert "id" not in feature_columns_to_use
    for column in DROPPED_COLUMNS:
        assert column not in feature_columns_to_use
    for column in TARGET_COLUMNS:
        assert column not in feature_columns_to_use

    # Round-5 review finding 6: this fit the whole cleaned UNSW frame, unsplit --
    # fit on the TRAIN split only, per AGENTS.md's "Every imputer, encoder and
    # scaler is fitted inside a scikit-learn Pipeline on training data only."
    train_df, _test_df = common.split_and_dedupe(
        df, target, test_size=0.3, seed=42, is_unsw_schema=is_unsw_schema
    )
    pipeline.fit(train_df[feature_columns_to_use])
    output_names = list(pipeline.named_steps["features"].get_feature_names_out())
    assert not any(name == "id" or name.startswith("id_") for name in output_names)


class TestCommonHelpers:
    """Direct tests of the shared leakage-prevention helpers."""

    def test_resolve_exclude_repeated_flag_form(self) -> None:
        """Round-5 review finding 7: `resolve_exclude` had no direct test, despite
        implementing the comma form the TTL rule depends on (`--exclude
        sttl,ct_state_ttl`)."""
        from _shared import common

        assert common.resolve_exclude(["id", "sttl"]) == ["id", "sttl"]

    def test_resolve_exclude_comma_separated_form(self) -> None:
        from _shared import common

        assert common.resolve_exclude(["sttl,ct_state_ttl"]) == ["sttl", "ct_state_ttl"]

    def test_resolve_exclude_mixed_repeated_and_comma_form(self) -> None:
        from _shared import common

        assert common.resolve_exclude(["id", "sttl,ct_state_ttl"]) == [
            "id",
            "sttl",
            "ct_state_ttl",
        ]

    def test_resolve_exclude_strips_whitespace_and_drops_empty_pieces(self) -> None:
        from _shared import common

        assert common.resolve_exclude([" sttl , ct_state_ttl ", " ", ""]) == [
            "sttl",
            "ct_state_ttl",
        ]

    def test_resolve_exclude_de_duplicates_preserving_first_occurrence_order(self) -> None:
        from _shared import common

        assert common.resolve_exclude(["id", "sttl", "id,ct_state_ttl"]) == [
            "id",
            "sttl",
            "ct_state_ttl",
        ]

    def test_resolve_exclude_empty_input_returns_empty_list(self) -> None:
        from _shared import common

        assert common.resolve_exclude([]) == []

    def test_compute_features_excludes_target_and_excluded_columns(self) -> None:
        from _shared import common

        df = pd.DataFrame({"id": [1, 2], "a": [1.0, 2.0], "target": ["x", "y"]})
        features = common.compute_features(df, target="target", exclude=["id"])
        assert "target" not in features
        assert "id" not in features
        assert features == ["a"]

    def test_assert_no_leakage_raises_on_target(self) -> None:
        from _shared import common

        with pytest.raises(ValueError, match="target"):
            common.assert_no_leakage(["a", "target"], target="target", exclude=[])

    def test_assert_no_leakage_raises_on_excluded_column(self) -> None:
        from _shared import common

        with pytest.raises(ValueError, match="excluded"):
            common.assert_no_leakage(["a", "id"], target="target", exclude=["id"])

    def test_compute_features_raises_valueerror_on_missing_exclude_column(self) -> None:
        """Review finding 5: the dead `leaked_excluded` check (provably unreachable
        by construction) was replaced with a real guard — an `--exclude` name that
        does not exist in the dataset is now a hard error instead of a silent
        no-op."""
        from _shared import common

        df = pd.DataFrame({"a": [1.0, 2.0], "target": ["x", "y"]})
        with pytest.raises(ValueError, match="not present in dataset columns"):
            common.compute_features(df, target="target", exclude=["typo_column"])

    def test_compute_features_raises_valueerror_on_missing_target(self) -> None:
        from _shared import common

        df = pd.DataFrame({"a": [1.0, 2.0]})
        with pytest.raises(ValueError, match="not found in dataset columns"):
            common.compute_features(df, target="missing_target", exclude=[])

    def test_is_unsw_raw_schema_false_for_synthetic_columns(self) -> None:
        from _shared import common

        df = pd.DataFrame({"a": [1], "b": [2]})
        assert common.is_unsw_raw_schema(df) is False

    def test_build_preprocessor_for_generic_never_leaks_target_or_excluded_names(
        self, synthetic_dataset: Path
    ) -> None:
        from _shared import common

        df = pd.read_csv(synthetic_dataset)
        features = common.compute_features(df, target="target", exclude=["id"])
        pipeline, feature_columns_to_use = common.build_preprocessor_for(df, features, ["id"])
        common.assert_no_leakage(feature_columns_to_use, "target", ["id"])

        # Round-5 review finding 6: fit on the TRAIN split only, matching
        # AGENTS.md's "split first, fit on training data only" rule -- this test
        # previously fit on the whole, unsplit `df`.
        train_df, _test_df = common.split_and_dedupe(
            df, "target", test_size=0.3, seed=42, is_unsw_schema=False
        )
        X = train_df[feature_columns_to_use]
        pipeline.fit(X)
        output_names = list(pipeline.named_steps["features"].get_feature_names_out())
        assert not any(name == "target" or name.startswith("target_") for name in output_names)
        assert not any(name == "id" or name.startswith("id_") for name in output_names)

    @staticmethod
    def _make_unsw_shaped_frame(n_rows: int = 60) -> pd.DataFrame:
        """A synthetic frame carrying every `RAW_COLUMNS` name (so
        `is_unsw_raw_schema` is true), with `label` derived from `attack_cat` exactly
        as in the real dataset (`label = attack_cat != "Normal"`)."""
        from nids.columns import RAW_COLUMNS

        rng = np.random.default_rng(42)
        data: dict[str, object] = {}
        for column in RAW_COLUMNS:
            if column in ("proto", "service", "state"):
                data[column] = rng.choice(["tcp", "udp", "-"], size=n_rows)
            elif column == "attack_cat":
                data[column] = rng.choice(["Normal", "Exploits", "DoS"], size=n_rows)
            elif column == "label":
                continue  # filled below, derived from attack_cat
            else:
                # Non-negative: several RAW_COLUMNS entries route through log1p in
                # the frozen `nids.preprocessing.build_preprocessor` pipeline, which
                # emits NaN for inputs below -1.
                data[column] = rng.uniform(0.0, 100.0, size=n_rows)
        data["label"] = (np.asarray(data["attack_cat"]) != "Normal").astype(int)
        return pd.DataFrame(data)

    def test_build_preprocessor_for_unsw_schema_drops_target_and_dropped_columns_regardless_of_exclude(
        self,
    ) -> None:
        """Regression test for the schema-routing leak (review findings 1 and 2):
        previously `build_preprocessor_for` only subtracted `DROPPED_COLUMNS`/
        `TARGET_COLUMNS` from the feature set when `--exclude` was empty or
        TTL-shaped, so `--target attack_cat --exclude id` let `label` (a perfect
        `attack_cat != "Normal"` predictor) and `id` reach the feature matrix.
        Routing is now keyed on schema detection alone, so neither ever leaks in,
        regardless of what `--exclude` names."""
        from _shared import common
        from nids.columns import DROPPED_COLUMNS, TARGET_COLUMNS

        df = self._make_unsw_shaped_frame()
        assert common.is_unsw_raw_schema(df)

        features = common.compute_features(df, target="attack_cat", exclude=["id"])
        assert "label" in features  # compute_features alone doesn't know the schema

        pipeline, feature_columns_to_use = common.build_preprocessor_for(df, features, ["id"])
        assert "label" not in feature_columns_to_use
        assert "id" not in feature_columns_to_use
        for column in DROPPED_COLUMNS:
            assert column not in feature_columns_to_use
        for column in TARGET_COLUMNS:
            assert column not in feature_columns_to_use

        common.assert_no_leakage(feature_columns_to_use, "attack_cat", ["id"], df=df)

        # Round-5 review finding 6: fit on the TRAIN split only, matching
        # AGENTS.md's "split first, fit on training data only" rule -- this test
        # previously fit on the whole, unsplit synthetic UNSW-shaped `df`.
        train_df, _test_df = common.split_and_dedupe(
            df, "attack_cat", test_size=0.3, seed=42, is_unsw_schema=True
        )
        X = train_df[feature_columns_to_use]
        pipeline.fit(X)
        output_names = list(pipeline.named_steps["features"].get_feature_names_out())
        assert not any(name == "label" or name.startswith("label_") for name in output_names)
        assert not any(name == "id" or name.startswith("id_") for name in output_names)

    def test_assert_no_leakage_catches_unsw_target_sibling_when_schema_detected(
        self,
    ) -> None:
        """Before this fix, `assert_no_leakage` only checked `target`/`exclude`
        membership, so it silently accepted `label` leaking in alongside a
        `target="attack_cat"` call. It now also checks `TARGET_COLUMNS`/
        `DROPPED_COLUMNS` whenever the UNSW schema is detected via the `df` kwarg."""
        from _shared import common

        df = self._make_unsw_shaped_frame()
        assert common.is_unsw_raw_schema(df)

        with pytest.raises(ValueError, match="UNSW-NB15"):
            common.assert_no_leakage(["dur", "label"], target="attack_cat", exclude=["id"], df=df)

        with pytest.raises(ValueError, match="UNSW-NB15"):
            common.assert_no_leakage(["dur", "id"], target="attack_cat", exclude=[], df=df)

    def test_resolve_results_writer_rejects_invalid_output_name(self, tmp_path: Path) -> None:
        """Round-3 review worth-fixing finding 8: `--output`'s final path component
        must match `nids.paths.NOTEBOOK_ID_PATTERN` (lowercase snake_case), raising a
        clear, pattern-naming `ValueError` -- not silently accepting an invalid name
        and either crashing deeper inside `ResultsWriter` or writing somewhere odd."""
        from _shared import common

        with pytest.raises(ValueError, match=r"must match.*\^\[a-z\]\[a-z0-9_\]"):
            common.resolve_results_writer(str(tmp_path / "My-Invalid-Name"))

    def test_resolve_results_writer_rejects_out_of_tree_output_by_default(
        self, tmp_path: Path
    ) -> None:
        """Round-4 review blocking finding 1: `resolve_results_writer` validated
        `notebook_id` against `NOTEBOOK_ID_PATTERN` but never validated its LOCATION,
        so `--output /tmp/anything` (or any path outside `results/`) silently wrote a
        full manifest outside the project results tree via `ResultsWriter`'s `root=`
        escape hatch -- exactly the containment `nids.paths.results_dir()` exists to
        enforce. `tmp_path` is guaranteed to be outside the real repository's
        `results/` tree, so calling without `allow_external=True` must now raise
        `ValueError` naming the violation, with no filesystem side effect."""
        from _shared import common

        target = tmp_path / "valid_run"
        with pytest.raises(ValueError, match="resolves outside the project results tree"):
            common.resolve_results_writer(str(target))
        assert not target.exists()

    def test_resolve_results_writer_accepts_valid_output_name_with_allow_external(
        self, tmp_path: Path
    ) -> None:
        """The explicit, documented `allow_external=True` escape hatch tests use to
        target `tmp_path` -- the production CLI path never passes it, so containment
        stays enforced for every real `--output` invocation."""
        from _shared import common

        writer = common.resolve_results_writer(
            str(tmp_path / "valid_run"), allow_external=True
        )
        assert writer.directory == tmp_path / "valid_run"

    def test_resolve_results_writer_accepts_output_inside_results_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Proves containment is ENFORCED by default, not merely permissive: an
        `--output` whose parent resolves to (or under) `results_root()` is accepted
        with no `allow_external` flag needed. `nids.paths.results_root` is
        monkeypatched to `tmp_path` so this never touches the real repository's
        `results/` directory."""
        from _shared import common

        monkeypatch.setattr("nids.paths.results_root", lambda: tmp_path)
        writer = common.resolve_results_writer(str(tmp_path / "valid_run"))
        assert writer.directory == tmp_path / "valid_run"

    def test_choose_n_clusters_never_looks_at_the_target(self) -> None:
        """Round-3 review worth-fixing finding 7: `choose_n_clusters` takes only the
        (already preprocessed/PCA-transformed) feature matrix `X` and a seed -- no
        target/label argument exists for it to derive `k` from."""
        from _shared import common

        rng = np.random.default_rng(42)
        # Three well-separated 2D blobs -> the silhouette sweep should favor k=3.
        blob0 = rng.normal(loc=(0, 0), scale=0.3, size=(40, 2))
        blob1 = rng.normal(loc=(10, 0), scale=0.3, size=(40, 2))
        blob2 = rng.normal(loc=(5, 10), scale=0.3, size=(40, 2))
        X = np.vstack([blob0, blob1, blob2])

        n_clusters, method = common.choose_n_clusters(X, seed=42)
        assert method == "silhouette_sweep"
        assert n_clusters == 3

    def test_choose_n_clusters_falls_back_to_fixed_minimum_for_a_tiny_dataset(self) -> None:
        from _shared import common

        X = np.array([[0.0, 0.0]])  # a single row: no valid k in [2, 10] fits
        n_clusters, method = common.choose_n_clusters(X, seed=42)
        assert method == "fixed_minimum"
        assert n_clusters == 2


class TestEdaReductionClustering:
    @pytest.fixture(autouse=True)
    def _stub_results_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """This class's tests pass a `tmp_path`-based `--output` to `main()`, standing
        in for the real `results/` tree. `resolve_results_writer`'s containment check
        (round-4 review blocking finding 1) is scoped to `nids.paths.results_root()`,
        so this class-wide fixture points it at `tmp_path` for the duration of each
        test -- keeping every run's `--output` contained relative to ITS test's own
        result root, without ever touching the real repository's `results/`
        directory."""
        monkeypatch.setattr("nids.paths.results_root", lambda: tmp_path)

    def test_help_exits_zero_without_a_dataset(self, capsys: pytest.CaptureFixture) -> None:
        with pytest.raises(SystemExit) as exc_info:
            eda_reduction_clustering.main(["--help"])
        assert exc_info.value.code == 0

    def test_runs_end_to_end_on_synthetic_dataset(
        self, synthetic_dataset: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "eda_run"
        eda_reduction_clustering.main(
            [
                "--dataset",
                str(synthetic_dataset),
                "--target",
                "target",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        assert payload["notebook_id"] == "eda_run"

        summary_table = next(t for t in payload["tables"] if t["path"] == "tables/summary_statistics.csv")
        summary_df = pd.read_csv(output_dir / summary_table["path"])
        assert "target" not in set(summary_df["column"])
        assert "id" not in set(summary_df["column"])

        metrics_by_name = {m["name"]: m["value"] for m in payload["metrics"]}
        assert "pca_explained_variance_ratio_sum" in metrics_by_name
        assert "n_clusters" in metrics_by_name
        # Worth-fixing finding 7: n_clusters must never be derived from the target;
        # the default run records HOW it was chosen (an unsupervised sweep).
        assert metrics_by_name["n_clusters_selection_method"] in (
            "silhouette_sweep",
            "fixed_minimum",
        )

    def test_n_clusters_flag_overrides_the_silhouette_sweep(
        self, synthetic_dataset: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "eda_run_fixed_k"
        eda_reduction_clustering.main(
            [
                "--dataset",
                str(synthetic_dataset),
                "--target",
                "target",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
                "--n-clusters",
                "3",
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        metrics_by_name = {m["name"]: m["value"] for m in payload["metrics"]}
        assert metrics_by_name["n_clusters"] == 3
        assert metrics_by_name["n_clusters_selection_method"] == "user_specified"

    @pytest.mark.slow
    @pytest.mark.skipif(not raw_data_dir().is_dir(), reason="data/raw/ is absent")
    def test_runs_end_to_end_on_unsw_nb15(self, tmp_path: Path) -> None:
        dataset_path = _require_training_csv()
        output_dir = tmp_path / "eda_unsw_run"
        eda_reduction_clustering.main(
            [
                "--dataset",
                str(dataset_path),
                "--target",
                "attack_cat",
                "--exclude",
                "label",
                "--output",
                str(output_dir),
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        _assert_unsw_run_excludes_dropped_columns(dataset_path, target="attack_cat", exclude=["label"])

        # Review finding 3: `summary_statistics.csv` must be built from
        # `feature_columns_to_use`, never the schema-blind `features`, or a real UNSW
        # run exports `id`/`stcpb`/`dtcpb`/`is_ftp_login` in this table.
        from nids.columns import DROPPED_COLUMNS

        summary_table = next(
            t for t in payload["tables"] if t["path"] == "tables/summary_statistics.csv"
        )
        summary_df = pd.read_csv(output_dir / summary_table["path"])
        summary_columns = set(summary_df["column"])
        for column in DROPPED_COLUMNS:
            assert column not in summary_columns, (
                f"summary_statistics.csv leaked DROPPED_COLUMNS member {column!r}"
            )

    @pytest.mark.slow
    @pytest.mark.skipif(not raw_data_dir().is_dir(), reason="data/raw/ is absent")
    def test_runs_end_to_end_on_unsw_nb15_with_exclude_id(self, tmp_path: Path) -> None:
        """Round-3 review blocking finding 1 regression: every prior slow UNSW test
        used `--exclude label`, never `--exclude id` -- so the tests walked around the
        bug instead of through it. `clean_dataset_if_unsw` drops `id` (a
        `DROPPED_COLUMNS` member) before `compute_features` validates `--exclude`
        names, so `--exclude id` used to raise `ValueError` on every real UNSW run,
        despite being the exact example documented in `skills/classification/
        SKILL.md`'s usage block. Proves it now exits 0 and still excludes both `id`
        and `label` from the feature matrix (label via the always-on UNSW routing
        drop, not via this --exclude)."""
        dataset_path = _require_training_csv()
        output_dir = tmp_path / "eda_unsw_exclude_id_run"
        eda_reduction_clustering.main(
            [
                "--dataset",
                str(dataset_path),
                "--target",
                "attack_cat",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
            ]
        )
        _assert_valid_manifest(output_dir)
        _assert_unsw_run_excludes_dropped_columns(dataset_path, target="attack_cat", exclude=["id"])


class TestClusteringReduction:
    @pytest.fixture(autouse=True)
    def _stub_results_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """See `TestEdaReductionClustering._stub_results_root`: this class's tests
        pass a `tmp_path`-based `--output` to `main()`, standing in for the real
        `results/` tree, so `nids.paths.results_root()` is pointed at `tmp_path` for
        the duration of each test (round-4 review blocking finding 1)."""
        monkeypatch.setattr("nids.paths.results_root", lambda: tmp_path)

    def test_help_exits_zero_without_a_dataset(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            clustering_reduction.main(["--help"])
        assert exc_info.value.code == 0

    def test_runs_end_to_end_on_synthetic_dataset(
        self, synthetic_dataset: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "clustering_run"
        clustering_reduction.main(
            [
                "--dataset",
                str(synthetic_dataset),
                "--target",
                "target",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        assert payload["notebook_id"] == "clustering_run"

        allocation_table = next(
            t for t in payload["tables"] if t["path"] == "tables/subsample_allocation.csv"
        )
        allocation_df = pd.read_csv(output_dir / allocation_table["path"])
        assert set(allocation_df["label"]) == {"A", "B", "C", "D"}

        comparison_table = next(
            t for t in payload["tables"] if t["path"] == "tables/clustering_comparison.csv"
        )
        comparison_df = pd.read_csv(output_dir / comparison_table["path"])
        assert set(comparison_df["algorithm"]) == {"kmeans", "agglomerative"}

        metrics_by_name = {m["name"]: m["value"] for m in payload["metrics"]}
        assert "best_silhouette_score" in metrics_by_name
        assert "best_algorithm" in metrics_by_name
        assert "n_clusters" in metrics_by_name
        assert metrics_by_name["n_clusters_selection_method"] in (
            "silhouette_sweep",
            "fixed_minimum",
        )

    def test_n_clusters_flag_overrides_the_silhouette_sweep(
        self, synthetic_dataset: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "clustering_run_fixed_k"
        clustering_reduction.main(
            [
                "--dataset",
                str(synthetic_dataset),
                "--target",
                "target",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
                "--n-clusters",
                "3",
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        metrics_by_name = {m["name"]: m["value"] for m in payload["metrics"]}
        assert metrics_by_name["n_clusters"] == 3
        assert metrics_by_name["n_clusters_selection_method"] == "user_specified"

    @pytest.mark.slow
    @pytest.mark.skipif(not raw_data_dir().is_dir(), reason="data/raw/ is absent")
    def test_runs_end_to_end_on_unsw_nb15(self, tmp_path: Path) -> None:
        dataset_path = _require_training_csv()
        output_dir = tmp_path / "clustering_unsw_run"
        clustering_reduction.main(
            [
                "--dataset",
                str(dataset_path),
                "--target",
                "attack_cat",
                "--exclude",
                "label",
                "--output",
                str(output_dir),
            ]
        )
        _assert_valid_manifest(output_dir)
        _assert_unsw_run_excludes_dropped_columns(dataset_path, target="attack_cat", exclude=["label"])


class TestClassification:
    @pytest.fixture(autouse=True)
    def _stub_results_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """See `TestEdaReductionClustering._stub_results_root`: this class's tests
        pass a `tmp_path`-based `--output` to `main()`, standing in for the real
        `results/` tree, so `nids.paths.results_root()` is pointed at `tmp_path` for
        the duration of each test (round-4 review blocking finding 1)."""
        monkeypatch.setattr("nids.paths.results_root", lambda: tmp_path)

    def test_help_exits_zero_without_a_dataset(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            classification.main(["--help"])
        assert exc_info.value.code == 0

    def test_runs_end_to_end_on_synthetic_dataset(
        self, synthetic_dataset: Path, tmp_path: Path
    ) -> None:
        output_dir = tmp_path / "classification_run"
        classification.main(
            [
                "--dataset",
                str(synthetic_dataset),
                "--target",
                "target",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        assert payload["notebook_id"] == "classification_run"

        report_table = next(
            t for t in payload["tables"] if t["path"] == "tables/classification_report_full.csv"
        )
        report_df = pd.read_csv(output_dir / report_table["path"])
        assert "id" not in set(report_df["class"])

        metric_names = {m["name"] for m in payload["metrics"]}
        assert "macro_f1" in metric_names
        assert "balanced_accuracy" in metric_names
        # The synthetic dataset has no sttl/ct_state_ttl columns, so no TTL variant.
        assert "macro_f1_without_ttl" not in metric_names

    def test_ttl_variant_runs_when_ttl_columns_present(self, tmp_path: Path) -> None:
        rng = np.random.default_rng(42)
        n_rows = 200
        target = np.array(["Normal"] * 140 + ["Attack"] * 60)
        rng.shuffle(target)
        frame = pd.DataFrame(
            {
                "id": np.arange(n_rows),
                "sttl": rng.integers(0, 255, size=n_rows),
                "ct_state_ttl": rng.integers(0, 10, size=n_rows),
                "feat_num1": rng.normal(size=n_rows),
                "target": target,
            }
        )
        dataset_path = tmp_path / "ttl_dataset.csv"
        frame.to_csv(dataset_path, index=False)

        output_dir = tmp_path / "classification_ttl_run"
        classification.main(
            [
                "--dataset",
                str(dataset_path),
                "--target",
                "target",
                "--exclude",
                "id",
                "--output",
                str(output_dir),
            ]
        )
        payload = _assert_valid_manifest(output_dir)
        metric_names = {m["name"] for m in payload["metrics"]}
        assert "macro_f1_without_ttl" in metric_names
        assert "balanced_accuracy_without_ttl" in metric_names
        table_paths = {t["path"] for t in payload["tables"]}
        assert "tables/classification_report_without_ttl.csv" in table_paths

    @pytest.mark.slow
    @pytest.mark.skipif(not raw_data_dir().is_dir(), reason="data/raw/ is absent")
    def test_runs_end_to_end_on_unsw_nb15(self, tmp_path: Path) -> None:
        dataset_path = _require_training_csv()
        output_dir = tmp_path / "classification_unsw_run"
        classification.main(
            [
                "--dataset",
                str(dataset_path),
                "--target",
                "attack_cat",
                "--exclude",
                "label",
                "--output",
                str(output_dir),
            ]
        )
        _assert_valid_manifest(output_dir)
        _assert_unsw_run_excludes_dropped_columns(dataset_path, target="attack_cat", exclude=["label"])
