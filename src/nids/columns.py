"""Frozen column allow-lists for the UNSW-NB15 partitions.

This module is the single place a raw or derived column name is written in the
package. Every other module imports these names instead of repeating them, so
`attack_cat`/`label` cannot reach the feature matrix through a skipped drop step, and
the rare-category grouping and leakage-key columns stay consistent across slices.
"""

from typing import Final, Literal

LeakageKey = Literal["full_row", "features_only"]
"""Row-comparison strategy used when checking test rows against train rows."""

RAW_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "stcpb",
    "dtcpb",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "is_ftp_login",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
    "attack_cat",
    "label",
)
"""The 45 UNSW-NB15 CSV columns, in the exact order they appear in the raw files."""

DROPPED_COLUMNS: Final[tuple[str, ...]] = ("id", "stcpb", "dtcpb", "is_ftp_login")
"""Columns removed before any feature or target selection happens."""

TARGET_COLUMNS: Final[tuple[str, ...]] = ("attack_cat", "label")
"""The two supervised targets. Never a feature, under any toggle."""

FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "dur",
    "proto",
    "service",
    "state",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "swin",
    "dwin",
    "tcprtt",
    "synack",
    "ackdat",
    "smean",
    "dmean",
    "trans_depth",
    "response_body_len",
    "ct_srv_src",
    "ct_state_ttl",
    "ct_dst_ltm",
    "ct_src_dport_ltm",
    "ct_dst_sport_ltm",
    "ct_dst_src_ltm",
    "ct_ftp_cmd",
    "ct_flw_http_mthd",
    "ct_src_ltm",
    "ct_srv_dst",
    "is_sm_ips_ports",
)
"""The 39-name modelling feature allow-list: RAW_COLUMNS minus DROPPED_COLUMNS minus
TARGET_COLUMNS, written explicitly rather than derived so no future edit can widen it
by subtraction."""

KEPT_COLUMNS: Final[tuple[str, ...]] = FEATURE_COLUMNS + TARGET_COLUMNS
"""The 41-name canonical post-drop column order: features first, then both targets."""

CATEGORICAL_COLUMNS: Final[tuple[str, ...]] = ("proto", "service", "state")
"""The three categorical feature columns."""

RARE_GROUPED_COLUMNS: Final[tuple[str, ...]] = ("proto", "service", "state")
"""Columns fed into RareCategoryGrouper. D1 resolved: `service` is rare-grouped too."""

SKEWED_COLUMNS: Final[tuple[str, ...]] = (
    "dur",
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "rate",
    "sload",
    "dload",
    "sloss",
    "dloss",
    "sinpkt",
    "dinpkt",
    "sjit",
    "djit",
    "response_body_len",
)
"""The 15 volumetric/rate/timing feature columns routed through `log1p` before
scaling. Frozen by design (Decision 6); never computed at runtime."""

TTL_SHORTCUT_COLUMNS: Final[tuple[str, ...]] = ("sttl", "ct_state_ttl")
"""The known testbed shortcut pair. Excluded from the feature set by the TTL toggle,
but always present in `leakage_key_columns()`, which takes no TTL parameter."""

SERVICE_DASH: Final[str] = "-"
SERVICE_NONE: Final[str] = "none"
OTHER_CATEGORY: Final[str] = "other"
RARE_CATEGORY_THRESHOLD: Final[float] = 0.01


def feature_columns(include_ttl: bool = True) -> list[str]:
    """Return the modelling feature allow-list, optionally without the TTL shortcut pair.

    Args:
        include_ttl: When False, `sttl` and `ct_state_ttl` are excluded from the
            returned list.

    Returns:
        The feature column names, in FEATURE_COLUMNS order. `attack_cat` and `label`
        never appear, under either toggle state.
    """
    if include_ttl:
        return list(FEATURE_COLUMNS)
    return [column for column in FEATURE_COLUMNS if column not in TTL_SHORTCUT_COLUMNS]


def leakage_key_columns(key: LeakageKey = "full_row") -> list[str]:
    """Return the row-comparison key. `full_row` is FEATURE_COLUMNS + TARGET_COLUMNS.

    This function takes no TTL parameter and never will: the comparison key is a
    dataset-identity question, not a modelling exclusion, so it always includes
    TTL_SHORTCUT_COLUMNS regardless of any feature-selection toggle used elsewhere.

    Args:
        key: `"full_row"` compares on features and both targets; `"features_only"`
            compares on features alone.

    Returns:
        The column names making up the comparison key.

    Raises:
        ValueError: `key` is not one of the two supported LeakageKey values.
    """
    if key == "full_row":
        return list(FEATURE_COLUMNS) + list(TARGET_COLUMNS)
    if key == "features_only":
        return list(FEATURE_COLUMNS)
    raise ValueError(f"Unsupported leakage comparison key: {key!r}.")


def numeric_feature_columns(include_ttl: bool = True) -> list[str]:
    """Selected features that are neither categorical nor skewed.

    Args:
        include_ttl: Forwarded to `feature_columns` before routing.

    Returns:
        The plain-numeric feature column names, in FEATURE_COLUMNS order.
    """
    selected = feature_columns(include_ttl)
    return [
        column
        for column in selected
        if column not in CATEGORICAL_COLUMNS and column not in SKEWED_COLUMNS
    ]


def skewed_feature_columns(include_ttl: bool = True) -> list[str]:
    """Selected features in SKEWED_COLUMNS, in FEATURE_COLUMNS order.

    Args:
        include_ttl: Forwarded to `feature_columns` before routing.

    Returns:
        The skewed feature column names, in FEATURE_COLUMNS order.
    """
    selected = feature_columns(include_ttl)
    return [column for column in selected if column in SKEWED_COLUMNS]
