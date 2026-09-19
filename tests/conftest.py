"""Shared pytest fixtures for the `nids` test suite.

`row_template`, `row_factory`, and `frame_factory` build synthetic raw-schema data
without touching `data/raw/`. `tmp_results_root` isolates filesystem writes under
`results/` to a temporary directory. `frozen_clock` supplies a fixed timestamp for
determinism tests.
"""

import matplotlib

matplotlib.use("Agg")  # Must run before any `matplotlib.pyplot` import anywhere in
# the suite (for example inside `nids.cleaning_report`), so figure-writing tests
# never depend on a display or a platform-specific GUI backend being available.

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from nids import paths
from nids.cleaning import CleaningResult, clean_partitions
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


@pytest.fixture
def raw_train(
    row_factory: Callable[..., dict[str, Any]],
    frame_factory: Callable[[list[Mapping[str, Any]]], pd.DataFrame],
) -> pd.DataFrame:
    """Engineered raw training partition for the data-cleaning capability.

    Builds exactly 200 feature/target-distinct rows so 1% is exactly 2 rows,
    with a `proto` composition that pins the rare-category boundary:

    | `proto` | rows | share  | outcome after cleaning |
    |---------|------|--------|-------------------------|
    | `tcp`   | 150  | 75.0%  | kept                    |
    | `udp`   | 46   | 23.0%  | kept                    |
    | `arp`   | 2    | 1.0%   | kept — the boundary case|
    | `ospf`  | 1    | 0.5%   | grouped into `"other"`  |
    | `sctp`  | 1    | 0.5%   | grouped into `"other"`  |

    Six exact duplicates of one `arp` row are appended (raw `arp` share
    8/206 ~= 3.9%, kept on the raw frame but only 1.0% once deduplication
    removes them — the miniature of the real-data raw-vs-cleaned finding used
    by `test_grouper_fit_differs_between_raw_and_cleaned` in Phase 4).

    Two feature-identical pairs carry conflicting `attack_cat` values, for the
    label-noise diagnostic (`conflicting_feature_combinations == 2`,
    `rows_in_conflicting_combinations == 4`).

    46 rows (the `udp` group) carry `service == "-"`; every other row carries
    a normal service value, for the dash-mapping invariant.
    """
    rows: list[dict[str, Any]] = []
    next_id = 1

    def add_row(**overrides: Any) -> dict[str, Any]:
        nonlocal next_id
        base: dict[str, Any] = {"id": next_id, "stcpb": next_id, "dtcpb": next_id}
        base.update(overrides)
        row = row_factory(**base)
        next_id += 1
        rows.append(row)
        return row

    # 146 plain tcp rows + 4 conflict-pair tcp rows below = 150 total (75.0%).
    for i in range(146):
        add_row(proto="tcp", service="http", dur=1000.0 + i)

    # Two feature-identical pairs with conflicting attack_cat (label-noise
    # floor). Both members of a pair share every feature (proto/service/dur
    # and the row-template defaults) and differ only on the target.
    add_row(proto="tcp", service="http", dur=5000.0, attack_cat="Normal", label=0)
    add_row(proto="tcp", service="http", dur=5000.0, attack_cat="Generic", label=1)
    add_row(proto="tcp", service="http", dur=5001.0, attack_cat="Normal", label=0)
    add_row(proto="tcp", service="http", dur=5001.0, attack_cat="Exploits", label=1)

    # 46 udp rows (23.0%), carrying service == "-" for the dash-mapping test.
    for i in range(46):
        add_row(proto="udp", service="-", dur=2000.0 + i)

    # 2 arp rows (exactly 1.0%, the boundary). The first is duplicated below.
    arp_row_a = add_row(proto="arp", service="http", dur=3000.0)
    add_row(proto="arp", service="http", dur=3001.0)

    # 1 ospf row (0.5%), 1 sctp row (0.5%) -- both grouped into "other".
    add_row(proto="ospf", service="http", dur=4000.0)
    add_row(proto="sctp", service="http", dur=4001.0)

    assert len(rows) == 200

    # 6 exact duplicates of the first arp row. Only id/stcpb/dtcpb differ,
    # and all three are dropped in step 1, so these are true duplicates on
    # the post-drop, post-mapping row.
    for _ in range(6):
        duplicate = dict(arp_row_a)
        duplicate["id"] = next_id
        duplicate["stcpb"] = next_id
        duplicate["dtcpb"] = next_id
        next_id += 1
        rows.append(duplicate)

    assert len(rows) == 206
    return frame_factory(rows)


@pytest.fixture
def raw_test(
    row_factory: Callable[..., dict[str, Any]],
    frame_factory: Callable[[list[Mapping[str, Any]]], pd.DataFrame],
) -> pd.DataFrame:
    """Engineered raw testing partition for the data-cleaning capability.

    Twelve rows, each engineered for one assertion:

    - 3 rows match a training row on features, `attack_cat`, and `label`
      (removed under the default `"full_row"` leakage key; `removed == 3`).
    - 4 rows match a training row on features alone but carry a different
      `attack_cat`/`label` (retained under `"full_row"` and counted —
      `retained_contradictory == 4`; removed under `"features_only"`, giving
      `removed == 7` and `retained_contradictory == 0` there).
    - 2 internal-duplicate pairs (`count_test_internal_duplicates == 2`):
      one pair identical on every raw column, one pair identical on the 41
      kept columns but differing in the dropped `stcpb` column — proving
      duplicate detection depends on the column drop running first (step
      order).
    - 1 `proto == "gre"` row present only in test (unseen category, used by
      the Phase 4 preprocessing tests).

    None of these rows share a `proto` with any other engineered scenario in
    this fixture, so the counts above cannot cross-contaminate each other.
    """
    rows: list[dict[str, Any]] = []
    next_id = 1000

    def add_row(**overrides: Any) -> dict[str, Any]:
        nonlocal next_id
        base: dict[str, Any] = {"id": next_id, "stcpb": next_id, "dtcpb": next_id}
        base.update(overrides)
        row = row_factory(**base)
        next_id += 1
        rows.append(row)
        return row

    # 3 rows matching train exactly on features + attack_cat + label.
    for i in range(3):
        add_row(proto="tcp", service="http", dur=1000.0 + i, attack_cat="Normal", label=0)

    # 4 rows matching train on features alone, contradicting attack_cat/label.
    for i in range(3, 7):
        add_row(
            proto="tcp", service="http", dur=1000.0 + i, attack_cat="Backdoor", label=1
        )

    # Internal-duplicate pair A: identical on every raw column, including
    # stcpb, so it is a duplicate under any column selection.
    add_row(proto="dns", service="-", dur=9000.1, attack_cat="Normal", label=0, stcpb=500)
    add_row(proto="dns", service="-", dur=9000.1, attack_cat="Normal", label=0, stcpb=500)

    # Internal-duplicate pair B: identical on the 41 kept columns, but the
    # two rows differ in `stcpb` -- dropped in step 1. A duplicate scan that
    # ran before the column drop would miss this pair.
    add_row(proto="ftp", service="-", dur=9001.1, attack_cat="Normal", label=0, stcpb=1)
    add_row(proto="ftp", service="-", dur=9001.1, attack_cat="Normal", label=0, stcpb=2)

    # Unseen-in-train category, present only in test.
    add_row(proto="gre", service="-", dur=9500.5, attack_cat="Normal", label=0)

    assert len(rows) == 12
    return frame_factory(rows)


@pytest.fixture
def cleaned(raw_train: pd.DataFrame, raw_test: pd.DataFrame) -> CleaningResult:
    """The `CleaningResult` from running `clean_partitions` on the engineered
    `raw_train`/`raw_test` fixtures, using the default `"full_row"` leakage
    comparison key.
    """
    return clean_partitions(raw_train, raw_test)
