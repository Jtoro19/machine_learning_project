"""Deterministic, seeded stratified subsampling bounded at 10,000 rows.

Gives the Gaussian Process classifier, spectral clustering, and t-SNE
notebooks a single way to reduce a cleaned partition to a workable size while
keeping every `attack_cat` class visible (AGENTS.md: "Gaussian Process,
spectral clustering and t-SNE run on a stratified subsample of at most 10000
rows").

Allocation uses the largest-remainder method with an explicit,
data-independent tie-break (design Decision 8): a floor pass, then
proportional distribution of the remaining budget by `floor(exact)`, then
residual distribution by largest fractional remainder, ties broken by class
label ascending. There is no randomness in *how many* rows each class gets;
randomness is used only to choose *which* rows are drawn within a class.
"""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClassAllocation:
    """One class's realized row allocation, plus the proportional reference.

    Attributes:
        label: The `attack_cat` value this allocation describes.
        available: Rows of this class present in the input.
        allocated: Rows actually drawn for this class.
        proportional: What a `floor=0` run would have allocated this class.
        floor_applied: `allocated > proportional` -- whether the floor (or
            the "class smaller than floor" rule) changed this class's
            allocation relative to pure proportional allocation.
        population_share: `available / total_rows` of the input.
        sample_share: `allocated / total_rows` of the output.
    """

    label: str
    available: int
    allocated: int
    proportional: int
    floor_applied: bool
    population_share: float
    sample_share: float


@dataclass(frozen=True, slots=True)
class SubsampleResult:
    """The subsampled frame plus every class's realized allocation.

    Attributes:
        data: The subsampled frame. Row order is preserved from the input
            (ascending by selected position), never reordered by class.
        allocations: One `ClassAllocation` per class present in the input,
            sorted by class label ascending.
        stratify_by: The column that was stratified on.
        requested_max_rows: The `max_rows` argument this result was built
            with.
        floor: The `floor` argument this result was built with.
        seed: The `seed` argument this result was built with.
        total_rows: `len(data)`.
    """

    data: pd.DataFrame
    allocations: tuple[ClassAllocation, ...]
    stratify_by: str
    requested_max_rows: int
    floor: int
    seed: int
    total_rows: int

    @property
    def floored_classes(self) -> tuple[str, ...]:
        """Labels whose allocation was raised above pure proportional."""
        return tuple(allocation.label for allocation in self.allocations if allocation.floor_applied)

    def to_frame(self) -> pd.DataFrame:
        """One row per class, ready for `ResultsWriter.add_table`.

        Returns:
            A `DataFrame` with one row per `ClassAllocation`, columns
            matching its field names, in `self.allocations` order (class
            label ascending).
        """
        return pd.DataFrame([asdict(allocation) for allocation in self.allocations])


def stratified_subsample(
    df: pd.DataFrame,
    *,
    stratify_by: str = "attack_cat",
    max_rows: int = 10_000,
    floor: int = 50,
    seed: int = 42,
) -> SubsampleResult:
    """Select at most `max_rows` rows from `df`, stratified by `stratify_by`.

    Args:
        df: The input partition. Its row order MUST already be deterministic
            (`clean_partitions`'s terminal `reset_index(drop=True)` is the
            precondition this relies on) so that seed-based reproducibility
            is not silently broken by a non-deterministic upstream order.
        stratify_by: The column to stratify on.
        max_rows: The output row cap. Must be positive. When `len(df) <=
            max_rows`, the entire input is returned unchanged, every class's
            `floor_applied` is `False`, and no random number is consumed.
        floor: The minimum rows guaranteed per class when the input has at
            least that many rows for the class. Must be non-negative.
            `floor=0` reproduces pure proportional allocation.
        seed: Seed for the row-selection RNG. Iteration is in sorted class
            order with one `numpy.random.default_rng(seed)` created once, so
            two calls with identical arguments return identical rows.

    Returns:
        A `SubsampleResult` with the subsampled frame and every class's
        realized allocation.

    Raises:
        ValueError: `max_rows <= 0`, `floor < 0`, `df` is empty, or the floor
            is infeasible for this `max_rows` (`sum(min(n_c, floor)) >
            max_rows`).
        KeyError: `stratify_by` is not a column of `df`.
    """
    if max_rows <= 0:
        raise ValueError(f"max_rows must be positive, got {max_rows!r}.")
    if floor < 0:
        raise ValueError(f"floor must be non-negative, got {floor!r}.")
    if df.empty:
        raise ValueError("stratified_subsample received an empty DataFrame.")
    if stratify_by not in df.columns:
        raise KeyError(f"stratify_by column {stratify_by!r} not found in df.columns.")

    # Round-5 review finding 3: labels were stringified for `labels_sorted`/diagnostics
    # but then compared directly against `df[stratify_by]` for the row mask below --
    # for a non-string (e.g. int64) `stratify_by` column, `df[stratify_by] == "0"` is
    # always False, `available` silently becomes all-zero, and `stratified_subsample`
    # returns an EMPTY frame with no exception. Fixed: `label_to_value` keeps the
    # ACTUAL unique value (original dtype) for every mask comparison; only the
    # diagnostics label text (`ClassAllocation.label`, `labels_sorted`'s ordering) is
    # stringified. Sorting by the stringified representation preserves the existing
    # deterministic ordering exactly as before this fix.
    label_to_value = {str(value): value for value in df[stratify_by].unique()}
    labels_sorted = sorted(label_to_value)
    available = {
        label: int((df[stratify_by] == label_to_value[label]).sum()) for label in labels_sorted
    }
    population_rows = int(len(df))

    if population_rows <= max_rows:
        allocations = tuple(
            ClassAllocation(
                label=label,
                available=available[label],
                allocated=available[label],
                proportional=available[label],
                floor_applied=False,
                population_share=available[label] / population_rows,
                sample_share=available[label] / population_rows,
            )
            for label in labels_sorted
        )
        return SubsampleResult(
            data=df.reset_index(drop=True),
            allocations=allocations,
            stratify_by=stratify_by,
            requested_max_rows=max_rows,
            floor=floor,
            seed=seed,
            total_rows=population_rows,
        )

    allocated = _largest_remainder_allocation(available, labels_sorted, max_rows, floor)
    proportional = _largest_remainder_allocation(available, labels_sorted, max_rows, 0)

    rng = np.random.default_rng(seed)
    chosen_positions: list[np.ndarray] = []
    for label in labels_sorted:
        positions = np.flatnonzero((df[stratify_by] == label_to_value[label]).to_numpy())
        chosen_positions.append(rng.choice(positions, size=allocated[label], replace=False))

    sorted_positions = np.sort(np.concatenate(chosen_positions))
    subsample = df.iloc[sorted_positions].reset_index(drop=True)

    allocations = tuple(
        ClassAllocation(
            label=label,
            available=available[label],
            allocated=allocated[label],
            proportional=proportional[label],
            floor_applied=allocated[label] > proportional[label],
            population_share=available[label] / population_rows,
            sample_share=allocated[label] / max_rows,
        )
        for label in labels_sorted
    )

    return SubsampleResult(
        data=subsample,
        allocations=allocations,
        stratify_by=stratify_by,
        requested_max_rows=max_rows,
        floor=floor,
        seed=seed,
        total_rows=len(subsample),
    )


def _largest_remainder_allocation(
    available: Mapping[str, int],
    labels_sorted: Sequence[str],
    max_rows: int,
    floor: int,
) -> dict[str, int]:
    """Largest-remainder allocation of `max_rows` across classes.

    Implements steps 2-4 of design Decision 8's algorithm: a floor
    feasibility check, proportional distribution of the remaining budget by
    `floor(exact)`, then residual distribution by largest fractional
    remainder with ties broken by class label ascending. `floor(exact)` can
    only under-allocate, so the residual is always `>= 0` and the sum always
    lands on exactly `max_rows`, with no overshoot-correction pass needed.

    Args:
        available: Per-class row counts in the input.
        labels_sorted: Class labels in ascending order -- both the iteration
            order and the tie-break order.
        max_rows: The total allocation budget. Must not exceed
            `sum(available.values())` (the caller only reaches this helper
            when the input exceeds the cap).
        floor: The per-class minimum baseline.

    Returns:
        Per-class allocated row counts, summing exactly to `max_rows`.

    Raises:
        ValueError: the floor is infeasible for this `max_rows`
            (`sum(min(n_c, floor)) > max_rows`).
    """
    base = {label: min(available[label], floor) for label in labels_sorted}
    base_total = sum(base.values())
    if base_total > max_rows:
        raise ValueError(
            f"floor={floor} is infeasible for max_rows={max_rows}: the sum of "
            f"per-class floors ({base_total}) exceeds the cap."
        )

    remainder_budget = max_rows - base_total
    remainder_pool = {label: available[label] - base[label] for label in labels_sorted}
    pool_total = sum(remainder_pool.values())

    exact: dict[str, float] = dict.fromkeys(labels_sorted, 0.0)
    allocated = dict(base)
    if remainder_budget > 0 and pool_total > 0:
        for label in labels_sorted:
            exact_c = remainder_budget * remainder_pool[label] / pool_total
            exact[label] = exact_c
            allocated[label] += min(int(exact_c), remainder_pool[label])

    deficit = max_rows - sum(allocated.values())
    if deficit > 0:
        candidates = [label for label in labels_sorted if allocated[label] < available[label]]
        ranked = sorted(candidates, key=lambda label: (-(exact[label] - int(exact[label])), label))
        for label in ranked[:deficit]:
            allocated[label] += 1

    # Round-5 review finding 3: the mismatched-dtype bug this guards against made
    # `available` silently all-zero, which starves the deficit pass above (every
    # `allocated[label] < available[label]` comparison is `0 < 0`, i.e. False) and
    # this function returned an allocation summing to 0 instead of `max_rows`, with
    # no exception -- the root cause of `stratified_subsample` silently returning an
    # empty frame. Fail loudly instead of ever returning a short allocation again.
    total_allocated = sum(allocated.values())
    if total_allocated != max_rows:
        raise ValueError(
            f"Internal allocation error: allocated rows ({total_allocated}) do not "
            f"sum to max_rows ({max_rows}); available={available!r}. This indicates "
            "`available` under-counts the input (e.g. a stratify column compared "
            "against mismatched label types)."
        )

    return allocated
