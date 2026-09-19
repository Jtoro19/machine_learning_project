# Design: Project Foundation

> **Read first.** This document fixes every architectural decision for `project-foundation` down to
> function signatures, so `sdd-tasks` splits work and `sdd-apply` writes code without making further
> architectural judgment calls. It adds no scope: everything here serves the proposal's eight
> deliverables. Three places where the design goes beyond the proposal's literal wording are marked
> **D1**, **D2**, **D3** and collected in [Deviations needing a yes/no](#deviations-needing-a-yesno).

---

## Technical Approach

One installable package, `nids`, built as a **layered chain with no back-edges**. Each layer imports
only from layers below it, which is what makes the proposal's five sequential commits revertible in
reverse order.

```
                                cleaning_report.py        <- slice 5 (composition + I/O)
                                        |
                +-----------------------+-----------------------+
                |                       |                       |
           results.py            data.py (facade)         (matplotlib)   <- slice 5 / 3
                |                       |
                |          +------------+------------+
                |          |                         |
                |      cleaning.py              sampling.py              <- slice 3 / 4
                |          |                         |
                |      loading.py   transformers.py  |  preprocessing.py <- slice 2 / 4
                |          |             |           |        |
                |      validation.py     |           |        |
                |          |             |           |        |
                +----- paths.py -------- columns.py -+--------+          <- slice 1
```

Four architectural commitments carry the whole change:

| # | Commitment | What it buys |
|---|---|---|
| 1 | **Unfitted operations and fitted operations live in different modules.** `cleaning.py` is pure `DataFrame -> DataFrame`; `transformers.py` / `preprocessing.py` hold every data-derived parameter. | The AGENTS.md blocking rule ("any `fit` on test data") becomes a *module-boundary* question a reviewer answers by reading imports, not by tracing dataflow. |
| 2 | **Diagnostics are return values, not side effects.** Every cleaning step returns `(frame, diagnostics)`; nothing logs, nothing prints, nothing recomputes. | The cleaning report cannot drift from the cleaning code, because it never touches a DataFrame. |
| 3 | **Column sets are frozen allow-lists in one module.** `columns.py` is the only place a column name is written. | `attack_cat`/`label` cannot reach the feature matrix through a skipped drop step. |
| 4 | **Determinism is designed in, not discovered.** Fixed iteration orders, explicit sort keys, atomic writes, one timestamp field in one file. | `results/` is committed; a nondeterministic writer would produce a dirty diff on every run forever. |

Relationship to `sdd-spec`: this document is the HOW. The concurrently-written specs under
`openspec/changes/project-foundation/specs/` own the Given/When/Then behavioral contracts. Where the
two describe the same rule (step order, leakage key, subsample floor), the proposal is the shared
source and neither document overrides it.

---

## Architecture Decisions

### Decision 1: Repository-root resolution by marker search, not by relative offset

**Choice.** `paths.repo_root()` walks **upward** from `Path(__file__).resolve()` looking for a
directory that contains **both** `pyproject.toml` and `src/nids`. If that fails it repeats the walk
from `Path.cwd().resolve()`. If both fail it raises `RepoRootNotFoundError` with an actionable
message. An environment variable `NIDS_REPO_ROOT` short-circuits the search but is validated against
the same two markers before it is trusted. Nothing is cached.

**Failure mode being designed against.** The naive form is `Path(__file__).parents[2]`. Under the
editable install this happens to be right. Under a **non-editable wheel install** `__file__` lives in
`.venv/lib/python3.14/site-packages/nids/paths.py`, and `parents[2]` is
`.venv/lib/python3.14/` — so `raw_data_dir()` silently returns a path that does not exist and
`results_dir("data_cleaning")` silently **creates a results tree inside site-packages**. The run
appears to succeed and writes nothing the repository can see. That is the specific bug this decision
exists to make impossible: a marker search either finds a directory that really is this repository,
or it raises.

**Behavior per install mode:**

| Situation | What `repo_root()` does |
|---|---|
| Editable install (`uv sync` with `[build-system]` present), imported from anywhere | Hatchling's editable hook points `nids.__file__` at the real `src/nids/paths.py` in the worktree. The upward walk finds the repo root on the third parent. Correct. |
| Notebook launched via `uv run jupyter lab` from the repo root, kernel cwd = `notebooks/` | `__file__` still resolves in the worktree, so the `__file__` walk succeeds before cwd is ever consulted. The cwd fallback is not needed — but it would also succeed, since `notebooks/` is inside the repo. |
| Non-editable wheel, `.venv` inside the repo | The `__file__` walk passes through `.venv/` and finds the repo root anyway. Correct, if slightly by luck. |
| Non-editable wheel, venv outside the repo, cwd outside the repo | Both walks fail. `RepoRootNotFoundError` names the two markers, the two start directories that were tried, and the `NIDS_REPO_ROOT` escape hatch. Loud, not silent. |

**Alternatives considered.**
`Path(__file__).parents[2]` — rejected for the silent-wrong-path failure above.
`importlib.resources` / package data — rejected: `data/raw/` and `results/` are repository
artifacts, not package data; shipping them inside the wheel would violate "raw data is never
modified" and bloat the distribution.
`git rev-parse --show-toplevel` — rejected: spawns a subprocess, requires git on PATH, and fails in a
source tarball with no `.git`.
`functools.lru_cache` on `repo_root()` — rejected: the search is a handful of `stat` calls, and a
cache would make `monkeypatch.setenv("NIDS_REPO_ROOT", ...)` order-dependent across tests.

**Marker choice rationale.** `pyproject.toml` alone matches any Python project above this one in the
filesystem. `.git` is absent from a tarball export. `pyproject.toml` **and** `src/nids` together
match exactly this repository and nothing in `site-packages`.

---

### Decision 2: Cleaning steps return `(frame, diagnostics)`; diagnostic-only steps return no frame

**Choice.** Each of the six ordered operations is a module-level function. Mutating steps return a
tuple `(new_frame, DiagnosticsDataclass)`. The two **diagnostic** steps
(`count_test_internal_duplicates`, `count_label_conflicts`) return **only** a dataclass — no
DataFrame at all. `clean_partitions()` runs them in order and stores what they returned into a single
frozen `CleaningResult`.

**Rationale (two separate wins).**

1. *Counts are by-products.* A count exists because the step that produced it had the before-frame
   and the after-frame in hand. No consumer re-derives it. The enforceable rule:
   **`cleaning_report.py` contains no `len()`, no `.shape`, no `value_counts()`, no `duplicated()`,
   and no `groupby()` over a partition frame.** Every number it writes comes from `CleaningResult`
   or from `RareCategoryGrouper.category_counts_`. An AST test asserts this (see Testing Strategy).
2. *The API makes the wrong move unspellable.* `count_test_internal_duplicates(test)` returns an
   `InternalDuplicateDiagnostics`. A caller who wanted to drop those rows has nothing to assign — the
   function hands back no frame. Likewise `drop_test_leakage(train, test, ...)` returns only a test
   frame, so it cannot shrink train even by a typo at the call site.

**Alternatives considered.**
A mutating `Cleaner` class accumulating state — rejected: hidden order dependencies between method
calls, and the object could be half-applied.
Logging counts and parsing the log in the report — rejected: string round-trip, and the numbers stop
being testable.
Returning a plain `dict[str, int]` instead of dataclasses — rejected: no type checking, no attribute
autocomplete, and `result["removed"]` vs `result["rows_removed"]` becomes a runtime `KeyError` in a
notebook cell.

---

### Decision 3: `RareCategoryGrouper` is fitted on the **cleaned** training partition

**Choice.** The grouper is a scikit-learn estimator with fitted state (`frequent_categories_`,
`category_counts_`). It MUST be fitted on the **cleaned** train partition — the output of
`clean_partitions()` — never on the raw CSV frame.

**Rationale, with the measured evidence.** Deduplication does not merely shrink the train partition;
it **reshapes the categorical distributions the 1% threshold is computed from.**

| Evidence | Raw train (175,341 rows) | Cleaned train | Consequence |
|---|---|---|---|
| `state == "INT"` | 82,275 rows (46.9%) | 19,726 rows (18.3%) | The distribution is not a scaled copy of itself. |
| `proto` values below the 1% line | 128 of 133 | **130 of 133** | **Two `proto` values flip from "kept as themselves" to "grouped into `other`" purely because of when the threshold is computed.** |

Fitting on raw counts therefore produces a *different, wrong* category set: two protocols survive
one-hot encoding as their own columns that the cleaned-train rule says should be folded into
`"other"`, and the resulting design matrix has two spurious near-empty columns. This is not a
rounding difference; it changes the feature space.

**How the design enforces it.** `preprocessing.py` exposes **no function that takes a DataFrame at
all** (see Decision 5), so the factory cannot fit anything. The only frames available to a notebook
are `load_raw_*()` output and `CleaningResult.train`/`.test`. `build_preprocessor()`'s docstring
names `CleaningResult.train` as the required fit input, and a unit test proves the two fits differ
(see Testing Strategy, `test_grouper_fit_differs_between_raw_and_cleaned`).

**Alternative considered and rejected.** Tagging cleaned frames with a private attribute
(`df.attrs["_nids_cleaned"] = True`) and checking it in `fit` — rejected: `DataFrame.attrs` does not
survive most pandas operations, `clone()` and `ColumnTransformer` slicing would drop it, and a
false-negative guard that blocks a legitimate fit is worse than a documented convention plus a test.

---

### Decision 4: The leakage comparison uses `MultiIndex.isin`, never a merge

**Choice.** Row-set membership between partitions is computed as

```python
train_key = pd.MultiIndex.from_frame(train[key_columns])
test_key  = pd.MultiIndex.from_frame(test[key_columns])
removed_mask = test_key.isin(train_key)          # np.ndarray[bool], len == len(test)
```

**Rationale — correctness, not just speed.** `pd.merge(test, train, on=key_columns,
indicator=True)` is a **many-to-many join**. Under `comparison_key="features_only"` the train
partition's *feature* keys are **not unique** (deduplication used the full 41-column row, so two
train rows can share features and differ on `attack_cat`). A merge would therefore emit one output
row per matching train row and silently multiply the test frame. `MultiIndex.isin` answers the
membership question directly, allocates one boolean array, and is immune to key multiplicity.

**Float-equality caveat, handled explicitly.** Tuple hashing uses exact float equality. Both
partitions are parsed by the same private `_read_partition()` with the same dtype map, so identical
CSV text produces bit-identical floats. `NaN != NaN` would break membership — the dataset has zero
NaN, and `validation.py` asserts it, so the precondition is checked rather than assumed. This is
recorded here so nobody "improves" the loader with a per-partition dtype override later.

**Alternatives considered.**
`pd.util.hash_pandas_object` — rejected: 64-bit hashing over 175k x 82k comparisons carries a
non-zero collision probability, and a collision would silently delete a legitimate test row.
Row-wise `apply` with tuple construction — rejected: minutes instead of seconds, no benefit.

---

### Decision 5: `preprocessing.py` accepts no DataFrame parameter anywhere

**Choice.** The module's entire public surface is:

```python
def build_preprocessor(include_ttl_features: bool = True) -> Pipeline: ...
def build_preprocessor_pair() -> dict[str, Pipeline]: ...
```

One `bool`, and one no-argument function. **No function in `preprocessing.py` takes data.**

**Rationale.** The proposal's requirement is "`preprocessing.py` exposes no function that takes a
test partition." Taking *no* partition is strictly stronger and much easier to verify: a reviewer
reads two signatures instead of auditing parameter names. A notebook that wants to fit must write
`preprocessor.fit(cleaned.train[feature_columns()])` **itself**, naming the frame at the call site —
which is precisely the line the AGENTS.md review checklist looks for. `fit(test)` becomes a visible,
deliberate act in the notebook, not something that can hide inside a helper.

`remainder="drop"` on the `ColumnTransformer` is the second half of this: even if a caller passes the
full 41-column cleaned frame, `attack_cat` and `label` are dropped rather than passed through.

---

### Decision 6: `log1p` sits **before** `StandardScaler`, inside the pipeline

**Choice.** The skewed branch is `FunctionTransformer(np.log1p, inverse_func=np.expm1) ->
StandardScaler()`, in that order, inside the `ColumnTransformer`.

**Rationale.** The reverse order is not merely suboptimal, it **produces NaN**. `StandardScaler`
centres the data, so roughly half the values become negative; `np.log1p(x)` for `x < -1` is NaN, and
NaN propagates silently through the design matrix into every downstream model. `log1p` first
compresses the 3.3-to-76.3 skew range; the scaler then standardises the compressed values, which is
the only ordering with a defined domain.

`FunctionTransformer` rather than a manual pre-step: the proposal's §5 row 8 requires composability —
one `fit(X_train)` / `transform(X_test)` pair instead of a pre-step three notebooks can each forget.
`inverse_func=np.expm1` is supplied so `inverse_transform` works for plotting in original units.

`SKEWED_COLUMNS` is a **frozen constant**, not computed at runtime. A runtime skew measurement would
make column routing data-derived — i.e. fitted state hiding outside the estimator — and would change
the feature space between train and test. A unit test asserts every listed column is non-negative,
which is `log1p`'s domain precondition.

---

### Decision 7: `OneHotEncoder(handle_unknown="ignore")` as unreachable belt-and-braces

**Choice.** `OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float64)`,
downstream of the grouper.

**Rationale.** After `RareCategoryGrouper.transform`, every value is either a fitted frequent
category or the literal `"other"` — the encoder's unknown path is unreachable by construction.
`"ignore"` is chosen anyway so that a future bug produces an all-zero row instead of taking a whole
notebook down mid-run. A test asserts the path stays unreachable: a test-only `proto` value must
produce `proto_other == 1`, **not** an all-zero one-hot block. That test is the evidence that
`"ignore"` is never actually exercised.

**Alternatives considered.**
`handle_unknown="error"` — rejected: turns a cosmetic edge case into a hard notebook failure, and the
grouper already guarantees correctness.
`handle_unknown="infrequent_if_exist"` with `min_frequency=0.01` — rejected: that is scikit-learn's
*own* rare-category grouping. Running it alongside `RareCategoryGrouper` means two mechanisms with
two thresholds interacting invisibly, and the fitted rare set would no longer be inspectable as a
single attribute the cleaning report can print.
`sparse_output=True` — rejected: the encoded block is about 16 columns; t-SNE, spectral clustering
and Gaussian Processes all densify anyway, so sparsity buys nothing and complicates
`ColumnTransformer` output handling.

---

### Decision 8: Subsample allocation uses the largest-remainder method with an explicit tie-break

**Choice.** Floor pass, then proportional distribution of the remaining budget by
`floor(exact)`, then residual distribution by largest fractional remainder with ties broken by class
label ascending. No randomness in allocation; randomness only in *which* rows are drawn.

**Rationale.** Round-half-up does not sum to the cap — it can overshoot or undershoot by several
rows, which then needs an ad-hoc correction pass whose behavior is unspecified. Largest-remainder
(Hare-Niemeyer) lands on the cap **by construction**, and its only ambiguity — ties — is resolved by a
stated, data-independent rule. Determinism is the requirement here, and an algorithm with one
documented tie-break is verifiable; a rounding rule plus a fix-up loop is not.

Full algorithm in [Subsample allocation](#7-subsample-allocation-samplingpy).

**Alternatives considered.**
`df.groupby(col).sample(n=..., random_state=42)` — rejected: group iteration order depends on
`sort=` and on whether the column is `category` or `str`, and pandas does not document the order in
which the per-group RNG is consumed. That is exactly the "silent drift across reruns despite seed 42"
risk the proposal flagged.
`train_test_split(..., stratify=...)` — rejected: cannot express a per-class floor at all.

---

### Decision 9: One timestamp, in one file, with an override

**Choice.** `generated_at` appears **only** in `manifest.json`. No table, no figure, no `counts.json`
carries a timestamp. Its value resolves in this order: explicit constructor argument ->
`SOURCE_DATE_EPOCH` environment variable (UTC epoch seconds) -> `datetime.now(UTC)`, always
serialised at second precision as `%Y-%m-%dT%H:%M:%SZ`.

**Rationale.** `results/` is committed. `generated_at` changes on every run *by definition* — that is
what the field means — so the design's job is to confine the churn, not to eliminate it. Confined as
above, a rerun over unchanged inputs produces **a one-line diff in exactly one file**, and
`SOURCE_DATE_EPOCH=0 python -m nids.cleaning_report` produces a **byte-identical tree**, which is how
the determinism success criterion is actually verified. The determinism test pops `generated_at`
before comparing manifests.

Two non-obvious churn sources are closed in the same decision:

| Source | Fix |
|---|---|
| matplotlib writes a `Software: Matplotlib version …` and a `Creation Time` PNG text chunk by default, so two identical runs produce different bytes | `fig.savefig(..., metadata={"Software": None, "Creation Time": None})` |
| pandas `value_counts()` tie ordering is not guaranteed stable across versions | every table is sorted by an explicit, total key before writing |

---

### Decision 10: Explicit string dtype at load time; never branch on `.dtype`

**Choice.** `loading.py` declares one constant, `STRING_DTYPE = "str"`, and applies it to
`proto`, `service`, `state`, `attack_cat` in a single private `_read_partition()`. Every category
comparison in the codebase uses Python `str` literals through `==`, `.isin({...})` or `.where(...)`.
**No module inspects or compares `.dtype`.** Numeric columns are left to pandas' inference;
`dtype_backend="pyarrow"` is **not** used.

**Rationale.** `pyarrow` is a declared dependency and pandas here is **3.0.6**, so PDEP-14 is live:
`read_csv` yields the new `str` dtype (pyarrow-backed storage, NaN missing semantics), **not**
`object`. The design must not assume pandas 2.x. The portability argument is that `"str"` resolves to
the pandas-3 `str` dtype and, on any pandas 2.x fallback, to `object` — and the operations the code
actually performs behave identically on both:

| Operation | `object` | pandas-3 `str` | Used for |
|---|---|---|---|
| `ser == "-"` | bool ndarray-backed Series | bool Series (NaN semantics) | the `"-"` detection count |
| `ser.where(ser != "-", "none")` | element replacement | element replacement | the `"-" -> "none"` mapping |
| `ser.isin(frozenset_of_str)` | identical | identical | rare-category grouping |
| `ser.value_counts()` | identical counts | identical counts | the 1% threshold |
| `pd.MultiIndex.from_frame` tuple hashing | Python `str` in tuples | Python `str` in tuples | dedup and leakage keys |

**The `"-" -> "none"` mapping uses `.where`, not `.replace`.** `Series.replace` has carried
regex-interpretation and downcasting behavior changes across pandas majors; `.where(cond, other)` has
no such history and no regex semantics. `.str.replace` is rejected outright — it is substring
replacement and would corrupt any value containing a hyphen.

**`.astype("category")` is forbidden before the grouper.** Under `category` dtype,
`.where(cond, "other")` raises when `"other"` is not among the declared categories, and
`value_counts()` includes zero-count unused categories — which changes the denominator's category set
and therefore the 1% decision. Stated here so it is not "optimised in" later.

**`ct_ftp_cmd` is deliberately left to inference.** In some UNSW-NB15 distributions that column
contains blank strings, which would make it `object` and silently break `StandardScaler`.
`validation.py` asserts it is numeric and names the file if it is not, rather than the loader forcing
a dtype that would raise an opaque parse error.

**Copy-on-Write.** pandas 3 enforces CoW, so chained assignment is a silent no-op. Every cleaning
function is therefore **pure**: it builds a new frame with `.assign()` / `.loc[mask]` / `.drop()` and
returns it. No function mutates its argument. This is a correctness requirement under CoW, not a
style preference.

---

### Decision 11: `nids/__init__.py` stays empty of heavy imports; the façade lives in `nids/data.py`

**Choice.** `__init__.py` holds the package docstring and `__all__: list[str] = []` — no re-exports
of anything that imports pandas, scikit-learn or matplotlib. The one-call convenience entry point
lives in a thin module `nids/data.py`. **(Deviation D3.)**

**Rationale.** The proposal's success narrative says the first notebook cell is
`from nids.data import load_clean_partitions`, but the proposal's module table lists neither
`data.py` nor that function. This design closes the gap in favour of the narrative, because the
narrative is the user-facing contract. `data.py` is ~30 lines:

```python
def load_clean_partitions(
    *,
    comparison_key: LeakageKey = "full_row",
    raw_dir: Path | None = None,
) -> CleaningResult:
```

It is the **only** module that both reads files and cleans, which keeps `cleaning.py` pure
(frames in, frames out) and therefore trivially unit-testable with synthetic fixtures.

Keeping `__init__.py` import-free means `from nids.paths import repo_root` costs no sklearn import,
and `import nids` in a notebook does not pause for seconds.

---

## Module-by-Module Design

Every signature below is the complete public surface of its module. Every function gets a docstring
(summary, `Args`, `Returns`, `Raises`) and pytest coverage, per AGENTS.md. Shared type aliases live
in `columns.py`.

### 1. `src/nids/paths.py` (slice 1)

```python
from pathlib import Path
import re

REPO_MARKERS: tuple[str, ...] = ("pyproject.toml", "src/nids")
REPO_ROOT_ENV_VAR: str = "NIDS_REPO_ROOT"
NOTEBOOK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


class RepoRootNotFoundError(RuntimeError):
    """Raised when no ancestor directory carries every repository marker."""


def find_repo_root(start: Path) -> Path | None:
    """Walk upward from `start` returning the first directory carrying every REPO_MARKER."""


def repo_root() -> Path:
    """Resolve the repository root. Raises RepoRootNotFoundError if it cannot be located."""


def raw_data_dir() -> Path:
    """Return `<repo_root>/data/raw`. Never creates it; this tree is read-only."""


def results_root() -> Path:
    """Return `<repo_root>/results`. Never creates it."""


def results_dir(notebook_id: str, *, create: bool = True) -> Path:
    """Return `<repo_root>/results/<notebook_id>`, creating it and `tables/`, `figures/`.

    Raises:
        ValueError: `notebook_id` does not match NOTEBOOK_ID_PATTERN.
        RuntimeError: the resolved directory escapes `results_root()`.
    """
```

Notes for `sdd-apply`:
- `results_dir` validates `notebook_id` **before** any `mkdir`, and re-checks
  `resolved.is_relative_to(results_root().resolve())` **after** resolution. The pattern already
  excludes `/`, `\`, `.` and `..`; the post-check is the defence-in-depth layer.
- `raw_data_dir()` deliberately has no `create` parameter. Nothing in this package may create,
  write to, or modify anything under `data/raw/`.

### 2. `src/nids/columns.py` (slice 1)

```python
from typing import Final, Literal

LeakageKey = Literal["full_row", "features_only"]

RAW_COLUMNS: Final[tuple[str, ...]]          # the 45 CSV columns, in file order
DROPPED_COLUMNS: Final[tuple[str, ...]]      # ("id", "stcpb", "dtcpb", "is_ftp_login")
TARGET_COLUMNS: Final[tuple[str, ...]]       # ("attack_cat", "label")
FEATURE_COLUMNS: Final[tuple[str, ...]]      # 39 names, explicit allow-list
KEPT_COLUMNS: Final[tuple[str, ...]]         # FEATURE_COLUMNS + TARGET_COLUMNS -> 41
CATEGORICAL_COLUMNS: Final[tuple[str, ...]]  # ("proto", "service", "state")
RARE_GROUPED_COLUMNS: Final[tuple[str, ...]] # ("proto", "service", "state")  -- see D1
SKEWED_COLUMNS: Final[tuple[str, ...]]       # 15 volumetric/rate/timing names
TTL_SHORTCUT_COLUMNS: Final[tuple[str, ...]] # ("sttl", "ct_state_ttl")
SERVICE_DASH: Final[str] = "-"
SERVICE_NONE: Final[str] = "none"
OTHER_CATEGORY: Final[str] = "other"
RARE_CATEGORY_THRESHOLD: Final[float] = 0.01


def feature_columns(include_ttl: bool = True) -> list[str]:
    """Return the modelling feature allow-list, optionally without the TTL shortcut pair."""


def leakage_key_columns(key: LeakageKey = "full_row") -> list[str]:
    """Return the row-comparison key. `full_row` is FEATURE_COLUMNS + TARGET_COLUMNS."""


def numeric_feature_columns(include_ttl: bool = True) -> list[str]:
    """Selected features that are neither categorical nor skewed."""


def skewed_feature_columns(include_ttl: bool = True) -> list[str]:
    """Selected features in SKEWED_COLUMNS, in FEATURE_COLUMNS order."""
```

**Concrete column partition (39 features, pairwise disjoint, exhaustive):**

| Group | Count | Members |
|---|---|---|
| Categorical | 3 | `proto`, `service`, `state` |
| Skewed (`log1p` then scale) | 15 | `dur`, `spkts`, `dpkts`, `sbytes`, `dbytes`, `rate`, `sload`, `dload`, `sloss`, `dloss`, `sinpkt`, `dinpkt`, `sjit`, `djit`, `response_body_len` |
| Plain numeric (scale only) | 21 | `sttl`, `dttl`, `swin`, `dwin`, `tcprtt`, `synack`, `ackdat`, `smean`, `dmean`, `trans_depth`, `ct_srv_src`, `ct_state_ttl`, `ct_dst_ltm`, `ct_src_dport_ltm`, `ct_dst_sport_ltm`, `ct_dst_src_ltm`, `ct_ftp_cmd`, `ct_flw_http_mthd`, `ct_src_ltm`, `ct_srv_dst`, `is_sm_ips_ports` |

**Critical asymmetry, stated so it cannot be "unified" later.** `feature_columns(include_ttl)`
carries the TTL toggle. `leakage_key_columns()` **does not and must never**. The TTL pair is a
*modelling* exclusion; the comparison key is a *dataset identity* question. Dropping `sttl` from the
comparison key would change which rows count as duplicates. A test asserts
`set(TTL_SHORTCUT_COLUMNS) <= set(leakage_key_columns("features_only"))` under every call.

`SKEWED_COLUMNS` selection rule, recorded for auditability: non-negative-by-construction
volumetric, rate and timing columns whose measured right skew on the raw train partition exceeds
1.0. The list is **frozen** (see Decision 6); the cleaning report is free to print measured skews so
the list stays auditable, but the list itself is never computed.

### 3. `src/nids/loading.py` (slice 2)

```python
from pathlib import Path
import pandas as pd

RAW_TRAIN_FILENAME: Final[str] = "UNSW_NB15_training-set.csv"
RAW_TEST_FILENAME: Final[str] = "UNSW_NB15_testing-set.csv"
STRING_DTYPE: Final[str] = "str"
STRING_COLUMNS: Final[tuple[str, ...]] = ("proto", "service", "state", "attack_cat")


class RawSchemaError(ValueError):
    """Raised when a raw CSV's column list does not match RAW_COLUMNS exactly."""


def raw_data_present(raw_dir: Path | None = None) -> bool:
    """True when both raw partition files exist. Used by `pytest.mark.skipif`."""


def load_raw_train(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read the training partition read-only, with the declared string dtypes."""


def load_raw_test(raw_dir: Path | None = None) -> pd.DataFrame:
    """Read the testing partition read-only, with the declared string dtypes."""


def _read_partition(path: Path) -> pd.DataFrame:
    """Single parse implementation. Both partitions MUST go through this function."""
```

`_read_partition` is the reason both partitions are dtype-identical, which is the precondition for
the `MultiIndex` equality in Decision 4. It performs exactly:

```python
frame = pd.read_csv(path, dtype={c: STRING_DTYPE for c in STRING_COLUMNS})
if tuple(frame.columns) != RAW_COLUMNS:
    raise RawSchemaError(...)   # names the file, the missing and the unexpected columns
return frame
```

`"-"` is **not** added to `na_values` and `keep_default_na` is left at its default (`"-"` is not in
pandas' default NA set), so `service == "-"` arrives as the literal two-character string the mapping
expects.

### 4. `src/nids/validation.py` (slice 2)

```python
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PartitionExpectation:
    filename: str
    rows: int
    columns: int


RAW_EXPECTATIONS: Final[tuple[PartitionExpectation, ...]] = (
    PartitionExpectation(RAW_TRAIN_FILENAME, 175_341, 45),
    PartitionExpectation(RAW_TEST_FILENAME, 82_332, 45),
)


@dataclass(frozen=True, slots=True)
class PartitionCheck:
    filename: str
    exists: bool
    rows: int | None
    columns: int | None
    expected_rows: int
    expected_columns: int
    columns_match: bool
    numeric_dtypes_ok: bool
    null_cells: int | None
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool: ...


@dataclass(frozen=True, slots=True)
class ValidationReport:
    checks: tuple[PartitionCheck, ...]

    @property
    def ok(self) -> bool: ...
    def render(self) -> str: ...


def validate_raw_data(raw_dir: Path | None = None) -> ValidationReport:
    """Check presence, shape, column names, numeric dtypes and null count. Reads only."""


def download_instructions(raw_dir: Path | None = None) -> str:
    """Manual UNSW-NB15 acquisition instructions, naming the two expected filenames."""


def main(argv: Sequence[str] | None = None) -> int:
    """Print the report; on failure also print download instructions. Returns 0 or 1."""


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
```

**`175_341`, `82_332` and `45` are the only dataset row/column literals permitted anywhere in
`src/nids`.** They are properties of the unmodified input, not of the cleaning, and they live in this
module alone. The AST hygiene test enforces exactly that (see Testing Strategy).

`main()` **never downloads anything**. It prints instructions and exits non-zero — the proposal's
out-of-scope rule and the host restriction, made structural by there being no network import in the
package at all.

### 5. `src/nids/cleaning.py` (slice 3)

Pure functions only. **No class, no `fit`, no file I/O, no mutation of any argument.**

```python
from dataclasses import dataclass
from collections.abc import Mapping
import pandas as pd


@dataclass(frozen=True, slots=True)
class DropColumnsDiagnostics:
    dropped: tuple[str, ...]
    columns_before: int
    columns_after: int


@dataclass(frozen=True, slots=True)
class ServiceDiagnostics:
    partition: str                 # "train" | "test"
    dash_rows_before: int
    none_rows_after: int
    replacement_label: str


@dataclass(frozen=True, slots=True)
class DeduplicationDiagnostics:
    rows_before: int
    rows_after: int
    duplicate_rows_dropped: int
    state_counts_before: Mapping[str, int]
    state_counts_after: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class InternalDuplicateDiagnostics:
    rows: int
    duplicate_rows: int
    duplicate_share: float


@dataclass(frozen=True, slots=True)
class LeakageDiagnostics:
    comparison_key: LeakageKey
    rows_before: int
    rows_after: int
    removed: int
    retained_contradictory: int
    retained_contradictory_share: float


@dataclass(frozen=True, slots=True)
class LabelConflictDiagnostics:
    conflicting_feature_combinations: int
    rows_in_conflicting_combinations: int


@dataclass(frozen=True, slots=True)
class ClassDistributionDiagnostics:
    train_before: Mapping[str, int]
    train_after: Mapping[str, int]
    test_before: Mapping[str, int]
    test_after: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class StepRowCounts:
    step: int
    operation: str
    partition: str
    rows_before: int
    rows_after: int
    rows_removed: int


@dataclass(frozen=True, slots=True)
class CleaningResult:
    train: pd.DataFrame
    test: pd.DataFrame
    drop_columns: DropColumnsDiagnostics
    service_train: ServiceDiagnostics
    service_test: ServiceDiagnostics
    deduplication: DeduplicationDiagnostics
    test_internal_duplicates: InternalDuplicateDiagnostics
    leakage: LeakageDiagnostics
    label_conflicts: LabelConflictDiagnostics
    class_distribution: ClassDistributionDiagnostics

    def row_count_timeline(self) -> list[StepRowCounts]:
        """Project the six ordered steps as table rows.

        MUST read only the diagnostics fields. MUST NOT reference `self.train`
        or `self.test`.
        """


def drop_unused_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, DropColumnsDiagnostics]: ...
def normalize_service(df: pd.DataFrame, *, partition: str) -> tuple[pd.DataFrame, ServiceDiagnostics]: ...
def drop_train_duplicates(train: pd.DataFrame) -> tuple[pd.DataFrame, DeduplicationDiagnostics]: ...
def count_test_internal_duplicates(test: pd.DataFrame) -> InternalDuplicateDiagnostics: ...
def drop_test_leakage(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    comparison_key: LeakageKey = "full_row",
) -> tuple[pd.DataFrame, LeakageDiagnostics]: ...
def count_label_conflicts(train: pd.DataFrame) -> LabelConflictDiagnostics: ...
def clean_partitions(
    train_raw: pd.DataFrame,
    test_raw: pd.DataFrame,
    *,
    comparison_key: LeakageKey = "full_row",
) -> CleaningResult: ...


def _row_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.MultiIndex:
    """Single construction point for every row-comparison key."""
```

**Implementation specifics `sdd-apply` must follow.**

| Step | Implementation |
|---|---|
| 1 `drop_unused_columns` | `df.drop(columns=list(DROPPED_COLUMNS))`, then reindex to `KEPT_COLUMNS` so column order is canonical regardless of input order. Raises `KeyError` naming any missing column. |
| 2 `normalize_service` | `dash = int((df["service"] == SERVICE_DASH).sum())`, then `df.assign(service=df["service"].where(df["service"] != SERVICE_DASH, SERVICE_NONE))`. `none_rows_after` is recounted on the output so the invariant "`none_after >= dash_before`" is a real measurement, not an assumption. |
| 3 `drop_train_duplicates` | `mask = train.duplicated(subset=list(KEPT_COLUMNS), keep="first")`; `train.loc[~mask].reset_index(drop=True)`. `state_counts_*` from `value_counts()` on input and output. `keep="first"` for determinism given CSV row order. |
| 4 `count_test_internal_duplicates` | `int(test.duplicated(subset=list(KEPT_COLUMNS), keep="first").sum())`. Returns **no frame**. |
| 5 `drop_test_leakage` | See the one-pass computation below. |
| 6 `count_label_conflicts` | `codes, _ = pd.factorize(_row_key(train, FEATURE_COLUMNS))`; build `DataFrame({"g": codes, "attack_cat": train["attack_cat"].to_numpy()}).drop_duplicates()`; `sizes = .groupby("g", sort=False).size()`; conflicting `= int((sizes > 1).sum())`. A single integer groupby, not a 39-key groupby. |

**Step 5, one pass, producing both numbers:**

```python
key_cols      = leakage_key_columns(comparison_key)
removed_mask  = _row_key(test, key_cols).isin(_row_key(train, key_cols))
test_clean    = test.loc[~removed_mask].reset_index(drop=True)

feature_mask  = _row_key(test_clean, FEATURE_COLUMNS).isin(_row_key(train, FEATURE_COLUMNS))
retained_contradictory = int(feature_mask.sum())
```

Why the arithmetic identity holds, and why it is a strong test with no magic numbers: the set of test
rows whose *features* match a train row partitions exactly into (rows that also match on
`attack_cat`+`label`) and (rows that do not). Under `"full_row"` the first part is `removed` and the
second survives into `test_clean` as `retained_contradictory`. Under `"features_only"` both parts are
removed, so `removed(features_only) == removed(full_row) + retained_contradictory(full_row)` and
`retained_contradictory(features_only) == 0`. Both facts fall out of the two masks above; neither is
computed by a separate pass.

**`clean_partitions` composition** — this is the only place the order is written:

```
train_raw, test_raw
    |
    |-- capture class_distribution.train_before / .test_before      (attack_cat value_counts)
    |
 [1] drop_unused_columns(train)  -> train1   DropColumnsDiagnostics
     drop_unused_columns(test)   -> test1    (asserted identical to train's)
    |
 [2] normalize_service(train1, partition="train") -> train2  ServiceDiagnostics
     normalize_service(test1,  partition="test")  -> test2   ServiceDiagnostics
    |
 [3] drop_train_duplicates(train2)             -> train3  DeduplicationDiagnostics
    |                                                      (test2 untouched)
 [4] count_test_internal_duplicates(test2)     ->         InternalDuplicateDiagnostics
    |                                                      (no frame returned)
 [5] drop_test_leakage(train3, test2, key=...) -> test3   LeakageDiagnostics
    |                                                      (train3 untouched)
 [6] count_label_conflicts(train3)             ->         LabelConflictDiagnostics
    |
    |-- capture class_distribution.train_after / .test_after
    |
    +-> CleaningResult(train=train3.reset_index(drop=True),
                       test=test3.reset_index(drop=True), ...)
```

The terminal `reset_index(drop=True)` on both partitions is **the mitigation for the subsample
determinism risk**: `stratified_subsample` relies on a deterministic positional order, and this is
where that order is established.

### 6. `src/nids/transformers.py` (slice 4)

```python
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Fold categories below a train-frequency threshold into a single label.

    MUST be fitted on the CLEANED training partition. Fitting on the raw partition
    learns a different category set (see design Decision 3).
    """

    def __init__(
        self,
        threshold: float = RARE_CATEGORY_THRESHOLD,
        other_label: str = OTHER_CATEGORY,
    ) -> None:
        self.threshold = threshold
        self.other_label = other_label

    # fitted attributes, trailing underscore per scikit-learn convention
    frequent_categories_: dict[str, frozenset[str]]
    category_counts_: dict[str, dict[str, int]]
    n_rows_fit_: int
    feature_names_in_: np.ndarray
    n_features_in_: int

    def fit(self, X: pd.DataFrame, y: object = None) -> "RareCategoryGrouper": ...
    def transform(self, X: pd.DataFrame) -> pd.DataFrame: ...
    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> np.ndarray: ...
```

**`fit` contract.**
- `X` MUST be a `pd.DataFrame`; anything else raises `TypeError` naming the received type.
- `0.0 <= threshold < 1.0`, else `ValueError`.
- `self.n_rows_fit_ = len(X)`; `ValueError` on an empty frame (the threshold denominator would be 0).
- Per column: `counts = X[col].value_counts()`; store the **full** mapping in `category_counts_[col]`
  (the cleaning report needs the rare ones too, with their counts);
  `frequent_categories_[col] = frozenset(counts.index[counts / n_rows_fit_ >= threshold])`.
- **Boundary rule: `>=` keeps.** A category at *exactly* 1.0% is kept as itself; strictly below 1% is
  grouped. This matches AGENTS.md's "under 1 percent" wording, and the synthetic fixture pins the
  boundary at exactly 2 rows out of 200.
- If `other_label` appears among the fitted categories, raise `ValueError`. A real `"other"`
  category would silently merge with the grouped bucket and the report would be wrong. Two lines,
  and it converts a silent data-corruption path into a startup error.
- Sets `feature_names_in_` (from `X.columns`) and `n_features_in_`.

**`transform` contract.**
- `check_is_fitted(self)` first.
- Raises `ValueError` if `tuple(X.columns) != tuple(self.feature_names_in_)`.
- For each column:
  `out[col] = X[col].where(X[col].isin(self.frequent_categories_[col]), self.other_label).astype(STRING_DTYPE)`
- **Rare-in-train and unseen-in-train travel the same code path.** There is no `if unseen` branch,
  because a value absent from `frequent_categories_` is absent for the same reason whether it was
  rare in train or never seen at all. The two behaviors therefore cannot diverge under a later edit.
- Returns a new DataFrame with the same index and column order. Never mutates `X`.
- The explicit `.astype(STRING_DTYPE)` makes the output dtype invariant regardless of whether the
  input arrived as `object` or pandas-3 `str`.

**`get_feature_names_out`** is one-to-one: returns `feature_names_in_` when `input_features is None`,
otherwise validates equality and echoes the argument. Implemented **without**
`sklearn.utils.validation._check_feature_names_in` — that helper is private, and a minor scikit-learn
upgrade removing or renaming it would break the build for no benefit.

**scikit-learn compliance scope.** `check_estimator` is **not** run. The grouper is DataFrame-only by
design (it needs column names to key `frequent_categories_`), and `check_estimator` requires ndarray
support. Instead the tests verify the contract that actually matters: `clone()` round-trip,
`get_params`/`set_params`, `fit_transform == fit().transform()`, `check_is_fitted` behavior before
fit, and correct composition inside `Pipeline` and `ColumnTransformer`.

### 7. Subsample allocation (`src/nids/sampling.py`, slice 4)

```python
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class ClassAllocation:
    label: str
    available: int            # rows of this class in the input
    allocated: int            # rows actually drawn
    proportional: int         # what floor=0 would have allocated
    floor_applied: bool       # allocated > proportional
    population_share: float
    sample_share: float


@dataclass(frozen=True, slots=True)
class SubsampleResult:
    data: pd.DataFrame
    allocations: tuple[ClassAllocation, ...]
    stratify_by: str
    requested_max_rows: int
    floor: int
    seed: int
    total_rows: int

    @property
    def floored_classes(self) -> tuple[str, ...]: ...
    def to_frame(self) -> pd.DataFrame:
        """One row per class, ready for ResultsWriter.add_table."""


def stratified_subsample(
    df: pd.DataFrame,
    *,
    stratify_by: str = "attack_cat",
    max_rows: int = 10_000,
    floor: int = 50,
    seed: int = 42,
) -> SubsampleResult: ...
```

**Allocation algorithm, step by step.** Let classes be `c` with available counts `n_c`,
`N = sum(n_c)`, cap `M = max_rows`, floor `F`.

| Step | Rule |
|---|---|
| **0. Guards** | `M <= 0`, `F < 0`, empty `df`, or `stratify_by` not in `df.columns` -> `ValueError`/`KeyError` with a message naming the offending value. |
| **1. Already under the cap** | If `N <= M`: allocate `a_c = n_c` for every class, `proportional_c = n_c`, `floor_applied = False`, **consume no RNG**, and return `df.reset_index(drop=True)` unchanged. A cap is a cap, not a resample. |
| **2. Floor feasibility** | `base_c = min(n_c, F)`. If `sum(base_c) > M` -> `ValueError` naming `F`, the class count and `M`. Truncating silently would break the floor contract without telling anyone. (10 classes x 50 = 500 << 10,000.) |
| **3. Proportional distribution of the remainder** | `R = M - sum(base_c)`; `r_c = n_c - base_c`; `exact_c = R * r_c / sum(r_c)`; `a_c = base_c + min(floor(exact_c), r_c)`. |
| **4. Residual by largest remainder** | `D = M - sum(a_c)` (>= 0 by construction). Rank unsaturated classes (`a_c < n_c`) by fractional remainder `exact_c - floor(exact_c)` **descending**, ties broken by **class label ascending**. Give one row each, cycling, until `D == 0`. No RNG. |
| **5. Reference allocation** | Re-run steps 3-4 with `F = 0` to obtain `proportional_c`. Set `floor_applied = a_c > proportional_c`. |
| **6. Row selection** | `rng = np.random.default_rng(seed)`, created **once**. Iterate classes in **sorted label order**. Per class: `positions = np.flatnonzero((df[stratify_by] == c).to_numpy())`; `chosen = rng.choice(positions, size=a_c, replace=False)`. |
| **7. Output ordering** | Concatenate all `chosen`, `np.sort(...)` ascending, `df.iloc[sorted_positions].reset_index(drop=True)`. |

**Rounding rule and why the total still lands on `M`.** Step 3 uses `floor`, which can only
under-allocate, so `D >= 0` always; step 4 hands out exactly `D` rows. The sum is therefore `M`
exactly, with no correction pass and no possibility of overshoot. Round-half-up would allow both
signs of error and need an unspecified fix-up. `floor_applied` is defined by **comparison against the
proportional reference**, not by "`n_c < F`", because the disclosure consumers need is "this class is
over-represented relative to the population", which is what `a_c > proportional_c` states exactly.

**Three determinism preconditions, each satisfied by design:**
1. *Class iteration order* — sorted labels, never `groupby` order (Decision 8).
2. *Within-class row order* — the input's positional order, made deterministic by
   `clean_partitions`' terminal `reset_index(drop=True)`.
3. *RNG consumption order* — one `default_rng(seed)` drawn in a fixed class sequence.

Step 7's ascending sort means the returned frame preserves the input's row order, so the result is
independent of iteration order *at the frame level* and two runs are byte-identical.

**`floor=0`** makes `base_c = 0`, so steps 3-4 are pure proportional largest-remainder and step 5's
reference allocation is identical — `floor_applied` is `False` everywhere. The proposal's
"`floor=0` reproduces pure proportional" requirement holds structurally, not by a special case.

### 8. `src/nids/preprocessing.py` (slice 4)

```python
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def build_preprocessor(include_ttl_features: bool = True) -> Pipeline:
    """Return an UNFITTED preprocessing pipeline.

    Fit on the CLEANED training partition only:
        pipe = build_preprocessor()
        pipe.fit(cleaned.train[feature_columns()])
        X_test = pipe.transform(cleaned.test[feature_columns()])
    """


def build_preprocessor_pair() -> dict[str, Pipeline]:
    """Return {"with_ttl": ..., "without_ttl": ...} for AGENTS.md's with/without reporting."""
```

Assembled shape:

```python
categorical = [c for c in RARE_GROUPED_COLUMNS if c in selected]
skewed      = skewed_feature_columns(include_ttl_features)
numeric     = numeric_feature_columns(include_ttl_features)
# runtime invariant: the three are pairwise disjoint and their union == selected

ColumnTransformer(
    transformers=[
        ("categorical", Pipeline([
            ("group_rare", RareCategoryGrouper()),
            ("onehot", OneHotEncoder(handle_unknown="ignore",
                                     sparse_output=False,
                                     dtype=np.float64)),
        ]), categorical),
        ("skewed", Pipeline([
            ("log1p", FunctionTransformer(np.log1p, inverse_func=np.expm1,
                                          feature_names_out="one-to-one",
                                          validate=False)),
            ("scale", StandardScaler()),
        ]), skewed),
        ("numeric", StandardScaler(), numeric),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
```

wrapped as `Pipeline([("features", column_transformer)])` so a notebook can append a model step and
so the returned object always answers `.fit`, `.transform` and `.get_feature_names_out` uniformly.

| Setting | Why |
|---|---|
| `remainder="drop"` | Structural leak guard: `attack_cat`/`label` are dropped even if the caller passes the full 41-column frame. |
| `verbose_feature_names_out=False` | Readable output names (`proto_tcp`, `sbytes`) instead of `categorical__proto_tcp`. scikit-learn raises on collisions, so this also acts as a name-uniqueness check. |
| `sparse_output=False` | Output is roughly 21 + 15 + ~16 one-hot = **~52 dense columns**. GP, t-SNE and spectral clustering densify anyway. |
| No `set_output` call in the factory | Default numpy output. A notebook opts in with `pipe.set_output(transform="pandas")`; a test asserts that opt-in produces the expected column names. Keeping it out of the factory avoids a global-ish setting that surprises a caller who wanted arrays. |
| Runtime disjointness assertion | Catches a bad edit to `columns.py` at construction time rather than producing a silently-narrower matrix. |

**Where the `sttl` / `ct_state_ttl` with-and-without reporting applies** (`openspec/config.yaml`
`rules.design` requires this call-out):

| Location | What the TTL toggle does |
|---|---|
| `columns.TTL_SHORTCUT_COLUMNS` | The single definition of the pair. Nothing else names `sttl` or `ct_state_ttl`. |
| `columns.feature_columns(include_ttl)` | Filters the 39-name allow-list down to 37. This is a filter of an allow-list, **not** a set difference. |
| `preprocessing.build_preprocessor(include_ttl_features)` | Both TTL columns are plain numeric, so exclusion simply removes them from the `numeric` branch. No fitted value changes shape or meaning. |
| `preprocessing.build_preprocessor_pair()` | Returns both variants keyed by name, so a future notebook reports with-and-without by **iterating a dict** rather than copy-pasting a cell. |
| `columns.leakage_key_columns()` | **The toggle does NOT apply here.** Dataset identity always includes both columns. |
| This change's scope | No supervised result is produced here. The pair exists so the three future notebooks cannot forget the AGENTS.md requirement; enforcement of "both numbers are reported" belongs to those notebooks' changes. |

### 9. `src/nids/results.py` (slice 5)

```python
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:                      # keep matplotlib out of import time
    from matplotlib.figure import Figure

SCHEMA_VERSION: Final[str] = "1.0"
ARTIFACT_NAME_PATTERN: Final[re.Pattern[str]] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    path: str          # POSIX, relative to the manifest's own folder
    title: str
    type: str          # "csv" | "json" | "png"
    description: str


@dataclass(frozen=True, slots=True)
class MetricEntry:
    name: str
    value: int | float | str
    description: str


@dataclass(frozen=True, slots=True)
class NoteEntry:
    id: str
    text: str


class ResultsWriter:
    def __init__(
        self,
        notebook_id: str,
        *,
        root: Path | None = None,
        generated_at: datetime | None = None,
        prune_undeclared: bool = True,
    ) -> None: ...

    @property
    def directory(self) -> Path: ...

    def add_table(self, frame: pd.DataFrame, *, name: str, title: str,
                  description: str, sort_by: Sequence[str] | None = None) -> Path: ...
    def add_json_table(self, payload: Any, *, name: str, title: str,
                       description: str) -> Path: ...
    def add_figure(self, figure: "Figure", *, name: str, title: str,
                   description: str) -> Path: ...
    def add_metric(self, name: str, value: int | float | str, description: str) -> None: ...
    def add_note(self, note_id: str, text: str) -> None: ...
    def write_json(self, payload: Any, *, name: str) -> Path: ...   # unregistered sidecar
    def write_manifest(self) -> Path: ...
    def close(self) -> Path: ...
    def __enter__(self) -> "ResultsWriter": ...
    def __exit__(self, *exc: object) -> None: ...   # writes the manifest on clean exit only
```

**`manifest.json` shape** (`notes` is the additive field, **D2**):

```json
{
  "schema_version": "1.0",
  "notebook_id": "data_cleaning",
  "generated_at": "2026-09-19T12:00:00Z",
  "tables":  [{"path": "tables/x.csv", "title": "...", "type": "csv", "description": "..."}],
  "figures": [{"path": "figures/y.png", "title": "...", "type": "png", "description": "..."}],
  "metrics": [{"name": "...", "value": 0, "description": "..."}],
  "notes":   [{"id": "retained_error_floor", "text": "..."}]
}
```

**Design points, each answering a specific failure:**

| Concern | Design response |
|---|---|
| Path traversal via `notebook_id` or artifact name | Both validated against `^[a-z][a-z0-9_]{0,63}$` **before** any filesystem call. Callers pass a bare name; the writer appends `.csv`/`.json`/`.png`. A caller therefore cannot supply an extension, a separator or a `..`. |
| Silent overwrite of a previous artifact | Registering the same `name` twice raises `ValueError`. |
| Partial writes on a crash | Every artifact is written to `<path>.tmp` in the same directory and then `os.replace`d — atomic on POSIX and Windows. **The manifest is written last**, so "manifest exists" implies "every declared artifact exists". |
| A manifest declaring a file that is not there | `write_manifest` stats every declared relative path and raises before writing if one is missing. The success criterion becomes a runtime guarantee, not only a test. |
| Windows/Linux path drift in a committed file | Manifest paths are built with `PurePosixPath`, never `os.sep`. |
| Orphan files from a previous schema accumulating in a committed tree | `prune_undeclared=True` deletes, on `close()`, only files directly under this notebook's own `tables/` and `figures/` with a registered extension that the manifest does not declare. Each pruned path is reported in the return of `close()`'s log line. **Rejected alternative:** `rmtree` of the whole directory, which would also delete `manifest.json` mid-run and any hand-added note. |
| Non-deterministic CSV bytes | `to_csv(index=False, lineterminator="\n", float_format="%.6f", encoding="utf-8")`; optional `sort_by` applied with `kind="stable"`. Callers must `reset_index()` — `index=False` is unconditional. |
| Non-deterministic JSON bytes | `json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False)` plus a trailing newline, written with `newline="\n"`. |
| Non-deterministic PNG bytes | `figure.savefig(tmp, format="png", dpi=150, metadata={"Software": None, "Creation Time": None})`. Without the `metadata` override matplotlib stamps the version and the wall-clock time into PNG text chunks and **every rerun is a binary diff**. |
| Entry ordering churn | `tables`, `figures`, `metrics`, `notes` are emitted in **registration order**, which is stable across reruns of a deterministic producer and matches the narrative a human reads. Sorting by name was rejected: renaming one title would reshuffle unrelated entries. |
| `generated_at` churn | See Decision 9. One field, one file, `SOURCE_DATE_EPOCH` override, popped before comparison in the determinism test. |

**Rerun contract, stated plainly:** with `SOURCE_DATE_EPOCH` set, two runs over identical inputs
produce a **byte-identical** `results/<id>/` tree. Without it, they differ in exactly one line of one
file: `manifest.json`'s `generated_at`.

### 10. `src/nids/cleaning_report.py` (slice 5)

```python
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

NOTEBOOK_ID: Final[str] = "data_cleaning"


def build_cleaning_report(
    *,
    comparison_key: LeakageKey = "full_row",
    raw_dir: Path | None = None,
    results_root: Path | None = None,
    generated_at: datetime | None = None,
) -> Path:
    """Run the cleaning steps and write results/data_cleaning/. Returns the directory."""


def main(argv: Sequence[str] | None = None) -> int:
    """argparse entry point: --comparison-key {full_row,features_only}. Returns 0 or 1."""


if __name__ == "__main__":       # pragma: no cover
    raise SystemExit(main())
```

**`main` runs `validation.validate_raw_data()` first** and returns `1` with the download
instructions when it fails, so the report can never half-run against a missing or truncated CSV.
`cleaning_report` depends on `validation`; never the reverse.

**The enforceable rule of this module** (Decision 2): it contains no `len()`, `.shape`,
`value_counts()`, `duplicated()` or `groupby()` applied to a partition frame. Every number comes from
`CleaningResult` or from `RareCategoryGrouper.category_counts_`.

**Artifacts written:**

| Artifact | Source | Sort key |
|---|---|---|
| `tables/row_counts_before_after.csv` | `CleaningResult.row_count_timeline()` | execution order (step number) — deliberately not re-sorted |
| `tables/attack_cat_distribution.csv` | `CleaningResult.class_distribution` | `train_rows_after` desc, then `attack_cat` asc |
| `tables/state_distribution.csv` | `DeduplicationDiagnostics.state_counts_before/after` | `rows_before` desc, then `state` asc |
| `tables/rare_categories.csv` | `RareCategoryGrouper.category_counts_` + `frequent_categories_`, fitted on `result.train[RARE_GROUPED_COLUMNS]` | `column` asc, `rows` desc, `category` asc |
| `figures/class_balance_before_after.png` | `class_distribution` | grouped bars, **log y-scale** (Worms ~127 against Normal ~51,890 is unreadable linearly). English title and axis labels. |
| `counts.json` | `{e.name: e.value for e in metrics}` plus `notes` | unregistered sidecar |
| `manifest.json` | the writer | last |

**`counts.json` and `manifest.json.metrics` are emitted from the same `MetricEntry` list**, so the
human-readable mirror cannot drift from the machine-readable source of truth.

Fitting the grouper here is not a detour — it is the only way `tables/rare_categories.csv` can be a
truthful record of the fitted grouping state, and it means the table is produced from a grouper
fitted on the **cleaned** train partition, which is the visible proof of Decision 3.

**Metrics emitted** (names fixed here so downstream skills can rely on them):
`train_duplicate_rows_dropped`, `test_internal_duplicate_rows`, `test_leakage_rows_dropped`,
`test_leakage_comparison_key` (string), `test_contradictory_rows_retained`,
`test_contradictory_share_of_retained`, `label_noise_floor_combinations`,
`label_noise_floor_rows`, `service_dash_rows_mapped`, `raw_null_cells`,
`train_rows_final`, `test_rows_final`.

**Notes emitted** (both mandated by the proposal, both also repeated in the README):
`benchmark_non_comparability` and `retained_error_floor`.

### 11. `src/nids/data.py` (slice 3, **D3**)

```python
def load_clean_partitions(
    *,
    comparison_key: LeakageKey = "full_row",
    raw_dir: Path | None = None,
) -> CleaningResult:
    """Load both raw partitions and run the ordered cleaning steps."""
    return clean_partitions(
        load_raw_train(raw_dir=raw_dir),
        load_raw_test(raw_dir=raw_dir),
        comparison_key=comparison_key,
    )
```

The proposal's success narrative promises exactly this import path. Keeping it in its own module is
what lets `cleaning.py` stay pure and file-system-free.

---

## Structural Leak Prevention

The AGENTS.md blocking rule — "any `fit` or `fit_transform` on test data" — is enforced by four
structural choices, not by discipline. Each row states what it prevents.

| # | Structural choice | What becomes impossible | How it is checked |
|---|---|---|---|
| 1 | `cleaning.py` defines no class, no `fit`, no `fit_transform`, and never calls one | A data-derived parameter (mode, mean, category set) cannot be computed in the module that runs on both partitions. Every cleaning operation is, by construction, safe to apply to test. | AST test walks `cleaning.py`: no `FunctionDef` named `fit`/`fit_transform`, no `Attribute` call with those names. |
| 2 | `preprocessing.py` takes no DataFrame parameter in any public function (Decision 5) | No helper can hide a `fit(test)`. A notebook that fits must name the frame at the call site, which is the exact line review greps for. | AST test: every public function's parameters are annotated `bool` or absent; no parameter named `X`, `test`, `X_test`, `df`, or `test_df`. |
| 3 | `feature_columns()` is an **explicit allow-list**, never `set(all) - set(targets)` | A skipped or reordered drop step cannot reintroduce `attack_cat`/`label` into the feature matrix. A set difference would silently include the target the moment step 1 moved. | Test asserts neither target is in `feature_columns(True)` or `feature_columns(False)`, and that `len(...)` is 39 / 37. |
| 4 | `ColumnTransformer(remainder="drop")` | Even passing the whole 41-column cleaned frame to `fit`, the targets are dropped rather than passed through. Belt to #3's braces. | Test fits on the full `CleaningResult.train` and asserts `get_feature_names_out()` contains neither target. |

Two further guards that are not leak prevention but sit on the same boundary:

- `drop_test_leakage(train, test, ...)` returns **only a test frame**, so the direction-critical step
  cannot shrink train even through a mistaken tuple unpack. Tested by asserting
  `len(train)` is unchanged.
- `count_test_internal_duplicates` and `count_label_conflicts` return **no frame at all**, so a
  diagnostic cannot become a mutation.

And the deliberate asymmetry, restated because it is the one place `attack_cat`/`label` are used
legitimately: the targets **are** part of the cleaning row-comparison key (`leakage_key_columns`,
`columns.py`) and **are not** part of the feature matrix (`feature_columns`, same module, different
function, different call sites). Two different jobs on the same columns, kept apart by naming and
covered by the tests in rows 3 and 4.

---

## Data Flow

```
data/raw/*.csv   (READ ONLY — nothing in this package ever writes here)
      |
      v
 loading._read_partition ------------ one parse, one dtype map, both partitions
      |
      v
 data.load_clean_partitions
      |
      v
 cleaning.clean_partitions  [1 drop cols][2 service][3 train dedup]
                            [4 test-dup count][5 test leakage][6 label conflicts]
      |                                                    |
      | CleaningResult.train / .test                       | every diagnostic dataclass
      |                                                    |
      +----------------+--------------------+              |
      |                |                    |              |
      v                v                    v              v
 preprocessing    sampling             transformers   cleaning_report
 .build_          .stratified_         .RareCategory        |
  preprocessor     subsample            Grouper             |
      |                |                    |               v
      | .fit(TRAIN)    | seed 42            | .fit(CLEANED  results.ResultsWriter
      | .transform(    | floor 50           |   TRAIN)            |
      |   TEST)        |                    |                     v
      v                v                    v          results/data_cleaning/
   design matrix   SubsampleResult   frequent_          manifest.json (written LAST)
   (~52 cols)      + realized counts  categories_       counts.json
                   + floor flags                        tables/*.csv
                                                        figures/*.png
```

The single arrow that matters for review: **`.fit` is only ever drawn from the TRAIN branch.** The
test frame reaches `preprocessing` through `.transform` alone, and it reaches `sampling`,
`transformers` and `cleaning_report` not at all.

---

## File Changes

| File | Action | Description |
|---|---|---|
| `pyproject.toml` | Modify | Add `[build-system]` (hatchling), `[tool.hatch.build.targets.wheel] packages = ["src/nids"]`, `[tool.pytest.ini_options]` (`testpaths`, `markers = ["slow: ..."]`, `addopts = ["-m", "not slow"]`). Dependencies unchanged. |
| `src/nids/__init__.py` | Create | Docstring + `__all__: list[str] = []`. No heavy imports (Decision 11). |
| `src/nids/paths.py` | Create | Marker-based repo-root resolution and the three directory accessors. |
| `src/nids/columns.py` | Create | Every column constant and the four selector functions. |
| `src/nids/loading.py` | Create | One `_read_partition`, two public loaders, `raw_data_present`. |
| `src/nids/validation.py` | Create | Shape/schema/dtype/null checks, download instructions, `main()`. |
| `src/nids/cleaning.py` | Create | Six ordered pure operations, nine diagnostics dataclasses, `clean_partitions`. |
| `src/nids/data.py` | Create | `load_clean_partitions` façade (**D3**). |
| `src/nids/transformers.py` | Create | `RareCategoryGrouper`. |
| `src/nids/preprocessing.py` | Create | `build_preprocessor`, `build_preprocessor_pair`. |
| `src/nids/sampling.py` | Create | `stratified_subsample`, `ClassAllocation`, `SubsampleResult`. |
| `src/nids/results.py` | Create | `ResultsWriter`, entry dataclasses, manifest emission. |
| `src/nids/cleaning_report.py` | Create | Report composition and `main()`. |
| `tests/conftest.py` | Create | Row/frame factories and the engineered synthetic partitions. |
| `tests/test_paths.py` … `tests/test_cleaning_report.py` | Create | One module per `src/nids` module (11 files). |
| `tests/test_leak_safety.py` | Create | AST guards for leak-prevention rows 1 and 2. |
| `tests/test_repo_hygiene.py` | Create | AST guard for "no hardcoded cleaned counts" and for the `cleaning_report` no-recompute rule. |
| `notebooks/.gitkeep`, `skills/.gitkeep`, `dashboard/.gitkeep`, `results/.gitkeep` | Create | Empty directories the proposal requires. |
| `results/data_cleaning/**` | Create (generated, committed) | Output of `python -m nids.cleaning_report`. |
| `README.md` | Create | Setup, manual acquisition, `uv run jupyter lab`, both mandatory notes, the subsample floor caveat. |
| `.gitignore` | **Unchanged** | Already ignores `data/raw/` and does not ignore `results/`. |
| `data/raw/**` | **Untouched** | Read-only, byte-identical before and after. |

---

## Testing Strategy

`strict_tdd: false` — tests are written alongside implementation, not before it. Every public
function in `src/nids` gets coverage. Runner: `pytest`.

### Fixture architecture (`tests/conftest.py`)

```python
@pytest.fixture(scope="session")
def row_template() -> dict[str, Any]:
    """One valid 45-column row with neutral defaults for every column."""

@pytest.fixture
def row_factory(row_template) -> Callable[..., dict[str, Any]]:
    """row_factory(proto="tcp", service="-", attack_cat="Normal") -> full 45-column dict."""

@pytest.fixture
def frame_factory(row_factory) -> Callable[[Sequence[Mapping[str, Any]]], pd.DataFrame]:
    """Build a DataFrame with columns == RAW_COLUMNS in order and the loader's dtypes."""

@pytest.fixture
def raw_train(frame_factory) -> pd.DataFrame:    # engineered, see below
@pytest.fixture
def raw_test(frame_factory) -> pd.DataFrame:     # engineered, see below
@pytest.fixture
def cleaned(raw_train, raw_test) -> CleaningResult
@pytest.fixture
def tmp_results_root(tmp_path) -> Path
@pytest.fixture
def frozen_clock() -> datetime                   # fixed generated_at for writer tests
```

`frame_factory` applies the **same** dtype map `loading._read_partition` uses, so a fixture frame is
dtype-indistinguishable from a parsed one. Without that, every pandas-3 dtype behavior the design
relies on would go untested.

### Controlling the 1% threshold precisely

The engineered train fixture is built so the **cleaned** partition has **exactly 200 rows**, making
1% exactly **2 rows**. That is the whole trick: a category with 2 rows sits at exactly the boundary
and must be **kept** (`>=`); a category with 1 row sits at 0.5% and must be **grouped**.

`proto` composition of the cleaned train fixture (`state` and `service` follow the same pattern):

| `proto` value | Rows | Share | Expected outcome |
|---|---|---|---|
| `tcp` | 150 | 75.0% | kept |
| `udp` | 46 | 23.0% | kept |
| `arp` | 2 | **exactly 1.0%** | **kept** — the boundary case |
| `ospf` | 1 | 0.5% | grouped into `other` |
| `sctp` | 1 | 0.5% | grouped into `other` |
| **total** | **200** | | |

**The duplicates are engineered to carry `proto == "arp"`.** The raw train fixture is those 200
distinct rows plus, say, 6 exact duplicates all with `proto == "arp"`. On the **raw** frame `arp` is
8/206 = 3.9% and would be **kept**; on the **cleaned** frame it is exactly 1.0% — and one extra
duplicate would push it below. This makes `test_grouper_fit_differs_between_raw_and_cleaned` a direct
miniature of the real-data finding (128 vs 130 `proto` values below the line) rather than an
abstract assertion.

### Engineered edge cases, and which assertion each one exists for

| Fixture feature | Exercises |
|---|---|
| A test-frame pair identical on the 41 kept columns but **differing on `stcpb`** | Step order. Under the wrong order (compare before dropping) they are not duplicates; under the correct order they are. |
| `service == "-"` on a known number of rows in both partitions | The `"-" -> "none"` mapping and the count invariant. |
| 6 exact duplicate train rows, all `proto == "arp"` | Deduplication count; the raw-vs-cleaned grouper fit difference. |
| 2 duplicate pairs **inside** the test frame | `count_test_internal_duplicates` returns 2 and the test row count is unchanged. |
| 3 test rows matching train on features **and** label | `removed == 3` under `full_row`. |
| 4 test rows matching train on features, **different** `attack_cat`/`label` | `retained_contradictory == 4` under `full_row`; `removed == 7` under `features_only`; identity `3 + 4 == 7`; `retained_contradictory == 0` under `features_only`. |
| `proto == "gre"` present only in the test frame | Unseen-in-train maps to `other`; `proto_other == 1`, not an all-zero one-hot block. |
| 2 train feature combinations each carrying 2 distinct `attack_cat` | `conflicting_feature_combinations == 2`. |
| A 1,000-row, 10-class sampling frame with a 3-row class and a 700-row class | Floor smaller than a class; `max_rows=100`, `floor=5`; total lands exactly on 100. |

### Test matrix

| Layer | What to test | Approach |
|---|---|---|
| Unit — `paths` | Marker search succeeds from a nested temp tree; fails loudly outside one; `NIDS_REPO_ROOT` honoured only when it carries the markers; `results_dir` rejects `../x`, `/abs`, `Bad-Id`, and creates `tables/`+`figures/`; `raw_data_dir` has no create path. | `tmp_path` fake repos, `monkeypatch.chdir`, `monkeypatch.setenv`. |
| Unit — `columns` | 45/41/39/37 counts; targets absent from `feature_columns` under both toggles; TTL pair present in every `leakage_key_columns` result; the three routing groups are disjoint and exhaustive; every `SKEWED_COLUMNS` member is non-negative in the fixture. | Pure assertions, no data. |
| Unit — `loading` | Both partitions parse to identical dtypes; `"-"` survives as a literal; `RawSchemaError` on a reordered/missing column; `raw_data_present` on a temp dir. | `tmp_path` CSVs written by the test. |
| Unit — `validation` | Missing file -> `ok is False`, instructions printed, `main() == 1`; wrong shape reported per-partition; `main() == 0` on a synthetic well-formed pair. | `tmp_path` + `capsys`. |
| Unit — `cleaning` | Every row in the engineered-edge-case table above; `clean_partitions` output has 41 columns; train count strictly decreases on dedup and is unchanged by leakage; index is a clean `RangeIndex`; **`row_count_timeline()` returns the diagnostic values when `CleaningResult` is built with empty frames** (proves no second pass). | Synthetic fixtures only. |
| Unit — `transformers` | Boundary at exactly 1% kept; 0.5% grouped; unseen -> `other`; grouper fitted on frame A applied to frame B yields A's category set; `clone()`/`get_params` round-trip; `ValueError` when `other_label` is a real category; `TypeError` on ndarray input; `fit_transform == fit().transform()`; **raw-vs-cleaned fit produces different `frequent_categories_`**. | Synthetic fixtures. |
| Unit — `sampling` | Two calls with seed 42 return identical row content; total == `max_rows`; a class smaller than the floor gets all its rows; `floor=0` matches an independently computed proportional largest-remainder allocation; `floor_applied` true exactly where `allocated > proportional`; `N <= max_rows` returns the input unchanged; infeasible floor raises. | Synthetic 1,000-row frame. |
| Unit — `results` | Manifest round-trips through `json.load`; every declared path exists; duplicate `name` raises; traversal names rejected; manifest written last (simulate a mid-run exception and assert no `manifest.json`); two runs with a fixed `generated_at` produce byte-identical files; `prune_undeclared` removes only orphans with registered extensions. | `tmp_results_root`, `frozen_clock`, an Agg-backend figure. |
| Integration — `preprocessing` | `fit` on the cleaned fixture train, `transform` on the cleaned fixture test: shapes agree; `get_feature_names_out()` excludes both targets; `include_ttl_features=False` output lacks `sttl`/`ct_state_ttl` and has exactly two fewer columns; no NaN in the output (the `log1p`-before-scale ordering proof); `set_output(transform="pandas")` yields the expected column names. | Synthetic fixtures + the real pipeline. |
| Integration — `cleaning_report` | `build_cleaning_report` against synthetic partitions in `tmp_results_root` produces manifest + counts + 4 tables + 1 figure; `counts.json` keys equal the manifest metric names; both notes present; a second run with the same `generated_at` is byte-identical. | Monkeypatched `raw_dir`, `Agg` backend. |
| Repo hygiene — `test_repo_hygiene` | AST walk of `src/nids/**` and `tests/**`: no integer literal `>= 10_000` outside a small allow-list (`175341`, `82332` in `validation.py`; `10000` in `sampling.py`); `cleaning_report.py` contains no `len`/`value_counts`/`duplicated`/`groupby`/`.shape` on a frame. | `ast.parse` + `ast.walk`. Regex was rejected: it matches comments and version strings. |
| Repo hygiene — `test_leak_safety` | AST walk: `cleaning.py` defines and calls no `fit`/`fit_transform`; no public function in `preprocessing.py` has a data parameter. | `ast.parse` + `ast.walk`. |
| Real data (skipped when absent) | `test_validation.py::test_raw_partitions_match_expected_shape` and `::test_raw_columns_match_allow_list` — `175,341x45` and `82,332x45`. Legitimate literals: properties of the unmodified input. | `@pytest.mark.skipif(not raw_data_present(), reason="data/raw/ absent")`. |
| Real data, slow (opt-in) | `test_cleaning_report.py::test_end_to_end_on_real_partitions` — runs the whole report and asserts the arithmetic identity on real data. | `@pytest.mark.slow` **and** `skipif`. Excluded from the default run by `addopts = ["-m", "not slow"]`; run with `uv run pytest -m slow`. |

**Why the "no hardcoded counts" test is structural rather than a literal blacklist.** Listing the
forbidden cleaned counts would put those exact numbers into `tests/` — the thing the rule prohibits.
Scanning for *any* integer literal at or above 10,000 with a tiny, justified allow-list is
self-limiting: a future contributor who hardcodes 107,740 trips it without the test ever naming
107,740.

---

## Threat Matrix

The standard matrix covers routing, shell, subprocess, VCS/PR automation and executable-file
classification. **Every standard row is `N/A` for this change.**

| Boundary | Applicability | Reason |
|---|---|---|
| Documentation-like paths | **N/A** | No file is classified or executed by extension or name. The package reads two fixed CSV filenames and writes `.csv`/`.json`/`.png` under a validated directory. |
| Git repository selection | **N/A** | No git invocation. `repo_root()` uses marker files, explicitly rejecting `git rev-parse` (Decision 1). No commit is authorized by this design. |
| Commit state | **N/A** | No VCS automation of any kind. |
| Push state | **N/A** | No VCS automation of any kind. |
| PR commands | **N/A** | No PR automation. Delivery is five manually authorized commits. |

No `subprocess`, no `shell=True`, no network import, no dynamic import, and no `eval`/`exec` appears
anywhere in `src/nids`. The two `main()` entry points parse `sys.argv` with `argparse` and return an
exit code; they spawn nothing.

Two project-specific boundaries **are** applicable and each carries a planned test. These extend the
matrix rather than replacing it:

| Boundary | Adversarial case | Design response | Planned test |
|---|---|---|---|
| **Results write path** | `notebook_id` or artifact name containing `..`, `/`, `\`, a leading `/`, or an extension | `^[a-z][a-z0-9_]{0,63}$` validated before any filesystem call; the writer appends the extension; `resolved.is_relative_to(results_root())` re-checked after resolution | `test_paths.py::test_results_dir_rejects_traversal`, `test_results.py::test_artifact_name_rejects_traversal` |
| **`data/raw/` read-only** | Any code path that could open a raw file for writing or create a directory under `data/raw/` | `raw_data_dir()` has no `create` parameter; loading uses `pd.read_csv` only; `validation.main()` prints instructions instead of downloading | `test_repo_hygiene.py::test_no_write_mode_open_under_raw` — AST walk asserting no `open(..., "w"/"a"/"x")`, no `to_csv`, no `mkdir` whose target derives from `raw_data_dir()` |

---

## Migration / Rollout

**No migration required.** The change is purely additive: new directories, new files, three new
`pyproject.toml` tables, and no existing code to alter because none exists. `data/raw/` is read-only
throughout and byte-identical before and after.

Rollout is the proposal's five sequential commits on `main`, each individually authorized by the
user. The layering diagram at the top of this document is what makes reverse-order `git revert`
safe: every slice imports only from earlier slices, so the dependency chain has no back-edges.

---

## Slice-by-Slice Implementation Order

| Slice | Modules | Tests | Independently reviewable against | Est. lines |
|---|---|---|---|---|
| **1 — packaging** | `pyproject.toml` tables, `src/nids/__init__.py`, `paths.py`, `columns.py`, `notebooks/` `skills/` `dashboard/` `results/` `.gitkeep`, README skeleton | `test_paths.py`, `test_columns.py`, `test_repo_hygiene.py` (initial allow-list) | `uv sync` then `uv run python -c "import nids"` succeeds; `repo_root()` resolves and raises loudly outside a repo; `feature_columns()` is 39 / 37 and excludes both targets; the three routing groups are disjoint and exhaustive. **No data needed.** | ~300 |
| **2 — data access** | `loading.py`, `validation.py` | `test_loading.py`, `test_validation.py`, `conftest.py` row/frame factories | `python -m nids.validation` prints a green report with `data/raw/` present and download instructions with exit 1 when absent; both partitions parse to identical dtypes; `"-"` survives parsing. The factories land here because `loading.py` defines the dtype contract they must mirror. | ~250 |
| **3 — cleaning** | `cleaning.py`, `data.py` (**D3**) | `test_cleaning.py`, engineered fixtures in `conftest.py`, `test_leak_safety.py` (cleaning half) | The step order (verified by the `stcpb`-differing fixture); leakage direction (train unchanged); the `3 + 4 == 7` arithmetic identity; `retained_contradictory == 0` under `features_only`; `row_count_timeline()` reads only diagnostics. **Review the order and the key before any number.** | ~360 |
| **4 — fitted state** | `transformers.py`, `preprocessing.py`, `sampling.py` | `test_transformers.py`, `test_preprocessing.py`, `test_sampling.py`, `test_leak_safety.py` (preprocessing half) | **The compliance-critical slice — every fitted parameter in the project lives here.** The 1% boundary at exactly 2/200; raw-vs-cleaned fit difference; unseen -> `other`; no NaN in the design matrix (the `log1p` ordering proof); the TTL pair; allocation determinism and the `floor=0` equivalence. | ~380 |
| **5 — outputs** | `results.py`, `cleaning_report.py`, README completion, generated `results/data_cleaning/` | `test_results.py`, `test_cleaning_report.py`, hygiene rule for `cleaning_report` | Manifest round-trip and every declared path existing; manifest written last; `SOURCE_DATE_EPOCH` byte-identical rerun; `counts.json` keys matching manifest metric names; both mandatory notes present; `prune_undeclared` removing only orphans. | ~340 |

Forecast: ~1,650 authored changed lines against a 400-line review budget, which is why the work is
sliced. Slicing is settled (five commits on `main`), so no further delivery decision is needed before
apply.

---

## Deviations needing a yes/no

Three points where this design goes beyond the proposal's literal wording. Each is small, each is
reversible by a one-line change, and each is flagged rather than absorbed.

| ID | Deviation | Rationale | If rejected |
|---|---|---|---|
| **D1** | `RARE_GROUPED_COLUMNS` includes **`service`**, not only `proto` and `state` as the proposal's §5 row 7 states. | Without it, an unseen `service` value in test hits `OneHotEncoder(handle_unknown="ignore")` and becomes an **all-zero row** — a silent hole in the exact place the proposal worried about. `service` gets the identical treatment `proto` gets. AGENTS.md is not violated: after the mapping, `"none"` is ~53.7% of cleaned train, so the `"-"` category is never grouped and is never imputed with the mode. | Change one constant: `RARE_GROUPED_COLUMNS = ("proto", "state")` and route `service` directly to the encoder. |
| **D2** | `manifest.json` gains an optional top-level `notes: [{id, text}]` array, at `schema_version` `"1.0"`. | The proposal requires two written notes in the cleaning report but gives them no home in the schema. Putting them only in `counts.json` means a dashboard reading manifests never sees the error-floor warning. Nothing consumes this schema yet, so there is no compatibility break to guard. | Drop the `notes` key from the manifest; keep it in `counts.json` only. |
| **D3** | An eleventh module, `src/nids/data.py`, holding `load_clean_partitions`. | The proposal's success narrative promises `from nids.data import load_clean_partitions`, but its module table lists neither. This closes the gap in favour of the user-facing narrative and keeps `cleaning.py` free of file I/O. | Move `load_clean_partitions` into `cleaning.py` and accept the `loading` import there, or into `__init__.py` and accept eager pandas import. |

---

## Open Questions

- [ ] **D1 / D2 / D3** above — three yes/no answers, each with a stated fallback. None blocks
      `sdd-tasks`; only D1 and D3 affect slices 3 and 4, and both fallbacks are single-line changes.
- [ ] **Leakage comparison key** — carried forward from the proposal, unchanged and unresolved by
      design: the default `"full_row"` retains roughly 4,294 feature-identical, label-contradictory
      test rows, a near-certain ~5.5% error floor on the retained test set. This design makes the key
      a parameter with both values first-class and reports both numbers, so flipping it is the whole
      remedy. **This is the top item for the user's design review.**
- [ ] `SKEWED_COLUMNS` holds 15 names selected as non-negative volumetric/rate/timing columns with
      raw-train skew above 1.0. The list is frozen by design (Decision 6). If the design review wants
      a different membership, it is a one-constant edit in `columns.py` and changes no logic.

Nothing above blocks `sdd-tasks` or `sdd-apply`. Every open item has a stated default and a one-line
reversal.
