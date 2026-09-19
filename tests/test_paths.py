"""Tests for repository-root resolution and derived filesystem locations."""

import inspect
from pathlib import Path

import pytest

from nids import paths


def _make_fake_repo(root: Path) -> None:
    """Create the two REPO_MARKERS under `root` so it resolves as a fake repository."""
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    (root / "src" / "nids").mkdir(parents=True)


class TestFindRepoRoot:
    def test_succeeds_from_a_nested_temp_tree(self, tmp_path: Path) -> None:
        _make_fake_repo(tmp_path)
        nested_start = tmp_path / "notebooks" / "sub"
        nested_start.mkdir(parents=True)

        found = paths.find_repo_root(nested_start)

        assert found == tmp_path.resolve()

    def test_returns_none_outside_a_repository(self, tmp_path: Path) -> None:
        isolated = tmp_path / "no_markers_here"
        isolated.mkdir()

        assert paths.find_repo_root(isolated) is None


class TestRepoRoot:
    def test_raises_loudly_outside_a_repository(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(paths.REPO_ROOT_ENV_VAR, raising=False)
        fake_file = tmp_path / "fake_location" / "paths.py"
        monkeypatch.setattr(paths, "__file__", str(fake_file))
        monkeypatch.chdir(tmp_path)

        with pytest.raises(paths.RepoRootNotFoundError) as exc_info:
            paths.repo_root()

        message = str(exc_info.value)
        for marker in paths.REPO_MARKERS:
            assert repr(marker) in message
        assert str(fake_file.resolve()) in message
        assert str(tmp_path.resolve()) in message
        assert paths.REPO_ROOT_ENV_VAR in message

    def test_env_var_honored_when_it_carries_both_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_repo = tmp_path / "fake_repo"
        fake_repo.mkdir()
        _make_fake_repo(fake_repo)
        outside = tmp_path / "outside"
        outside.mkdir()

        monkeypatch.setattr(paths, "__file__", str(outside / "paths.py"))
        monkeypatch.chdir(outside)
        monkeypatch.setenv(paths.REPO_ROOT_ENV_VAR, str(fake_repo))

        assert paths.repo_root() == fake_repo.resolve()

    def test_env_var_ignored_when_it_lacks_both_markers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        invalid_target = tmp_path / "not_a_repo"
        invalid_target.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        monkeypatch.setattr(paths, "__file__", str(outside / "paths.py"))
        monkeypatch.chdir(outside)
        monkeypatch.setenv(paths.REPO_ROOT_ENV_VAR, str(invalid_target))

        with pytest.raises(paths.RepoRootNotFoundError):
            paths.repo_root()


class TestResultsDir:
    @pytest.fixture
    def fake_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> Path:
        _make_fake_repo(tmp_path)
        monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)
        return tmp_path

    @pytest.mark.parametrize("bad_id", ["../x", "/etc", "Bad-Id"])
    def test_rejects_invalid_notebook_ids(
        self, fake_repo: Path, bad_id: str
    ) -> None:
        with pytest.raises(ValueError):
            paths.results_dir(bad_id)

    def test_creates_tables_and_figures_on_success(self, fake_repo: Path) -> None:
        result = paths.results_dir("data_cleaning")

        assert result == fake_repo / "results" / "data_cleaning"
        assert (result / "tables").is_dir()
        assert (result / "figures").is_dir()

    def test_create_false_does_not_create_directories(
        self, fake_repo: Path
    ) -> None:
        result = paths.results_dir("data_cleaning", create=False)

        assert not result.exists()


class TestRawDataDir:
    def test_has_no_create_parameter(self) -> None:
        signature = inspect.signature(paths.raw_data_dir)
        assert "create" not in signature.parameters

    def test_never_creates_its_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_fake_repo(tmp_path)
        monkeypatch.setattr(paths, "repo_root", lambda: tmp_path)

        result = paths.raw_data_dir()

        assert result == tmp_path / "data" / "raw"
        assert not result.exists()
