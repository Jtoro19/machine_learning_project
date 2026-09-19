"""Shared pytest fixtures for the `nids` test suite.

`row_template`, `row_factory`, and `frame_factory` build synthetic raw-schema data
without touching `data/raw/`. `tmp_results_root` isolates filesystem writes under
`results/` to a temporary directory. `frozen_clock` supplies a fixed timestamp for
determinism tests.
"""

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nids import paths
from nids.columns import RAW_COLUMNS
from nids.loading import STRING_COLUMNS, STRING_DTYPE


@pytest.fixture(scope="session")
def row_template() -> dict[str, Any]:
    """One valid 45-column raw row with neutral, schema-conforming default values.

    Every key in `RAW_COLUMNS` is present. Callers that need a specific value
    should use `row_factory` rather than mutating this session-scoped dict.
    """
    return {
        "id": 1,
        "dur": 0.0,
        "proto": "tcp",
        "service": "-",
        "state": "FIN",
        "spkts": 1,
        "dpkts": 1,
        "sbytes": 100,
        "dbytes": 100,
        "rate": 1.0,
        "sttl": 254,
        "dttl": 252,
        "sload": 100.0,
        "dload": 100.0,
        "sloss": 0,
        "dloss": 0,
        "sinpkt": 0.5,
        "dinpkt": 0.5,
        "sjit": 0.0,
        "djit": 0.0,
        "swin": 255,
        "stcpb": 0,
        "dtcpb": 0,
        "dwin": 255,
        "tcprtt": 0.0,
        "synack": 0.0,
        "ackdat": 0.0,
        "smean": 100,
        "dmean": 100,
        "trans_depth": 0,
        "response_body_len": 0,
        "ct_srv_src": 1,
        "ct_state_ttl": 1,
        "ct_dst_ltm": 1,
        "ct_src_dport_ltm": 1,
        "ct_dst_sport_ltm": 1,
        "ct_dst_src_ltm": 1,
        "is_ftp_login": 0,
        "ct_ftp_cmd": 0,
        "ct_flw_http_mthd": 0,
        "ct_src_ltm": 1,
        "ct_srv_dst": 1,
        "is_sm_ips_ports": 0,
        "attack_cat": "Normal",
        "label": 0,
    }


@pytest.fixture
def row_factory(
    row_template: Mapping[str, Any],
) -> Callable[..., dict[str, Any]]:
    """Build a full 45-column raw row dict from `row_template`, plus overrides.

    Returns:
        A callable `factory(**overrides) -> dict[str, Any]` that returns a fresh
        copy of `row_template` with `overrides` applied. `row_template` itself is
        never mutated.
    """

    def factory(**overrides: Any) -> dict[str, Any]:
        row = dict(row_template)
        row.update(overrides)
        return row

    return factory


@pytest.fixture
def frame_factory() -> Callable[[list[Mapping[str, Any]]], pd.DataFrame]:
    """Build a raw-schema data frame from a list of row dicts.

    Returns:
        A callable `factory(rows) -> pd.DataFrame` that builds a `DataFrame` with
        columns in `RAW_COLUMNS` order, using the same dtype map as
        `loading._read_partition` (STRING_COLUMNS cast to STRING_DTYPE).
    """

    def factory(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
        frame = pd.DataFrame(list(rows), columns=list(RAW_COLUMNS))
        dtype_map = {column: STRING_DTYPE for column in STRING_COLUMNS}
        return frame.astype(dtype_map)

    return factory


@pytest.fixture
def tmp_results_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect `paths.results_root()` to an isolated temporary directory.

    Returns:
        The temporary `results/` root that `paths.results_root()` now resolves to.
    """
    root = tmp_path / "results"
    monkeypatch.setattr(paths, "results_root", lambda: root)
    return root


@pytest.fixture
def frozen_clock() -> datetime:
    """A fixed, timezone-aware timestamp for determinism tests.

    Returns:
        A constant `datetime` in UTC, suitable for a `generated_at` override.
    """
    return datetime(2024, 1, 1, tzinfo=UTC)
