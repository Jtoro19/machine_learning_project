# Design: Notebook 01 — EDA, Dimensionality Reduction and Clustering

> **Read first.** This document fixes every technical decision for the notebook down to cell order,
> function shape, artifact name, figure axis and English label, so `sdd-tasks` splits work and
> `sdd-apply` writes cells without making further architectural judgment calls.
> It adds no scope beyond `proposal.md` and contradicts none of it. Where it goes beyond the
> proposal's literal wording, the refinement is listed in
> [Refinements to the proposal](#refinements-to-the-proposal) with its reason.
>
> **One provisional input**: `notebook_id` is `eda_reduction_clustering` under proposal **Q1 option A**,
> still awaiting the user's ruling. See [Q1 status](#q1-status-provisional).

---

## Quick path

1. Author `notebooks/01_eda_reduction_clustering.ipynb` as nbformat v4.5 JSON, ~73 cells, hand-assigned cell ids (Decision 1).
2. Validate the JSON, then execute it in one pass with an explicit unlimited cell timeout (see [Build and execution procedure](#build-and-execution-procedure)).
3. Read the printed findings, write the section 6 narrative markdown, execute once more. That second pass is the committed state.
4. Verify: 17 tables, 16 figures, 29 metrics, 6 notes in `results/eda_reduction_clustering/manifest.json`; both runtime assertion cells passed; the `rg` guards return nothing.

---

## Technical Approach

The notebook is a **thin consumer** of the frozen `nids` API. Every operation that touches data either
comes from `nids` or is arithmetic the notebook performs on an already-transformed matrix. Four
structural commitments carry the whole change:

| # | Commitment | What it buys |
|---|---|---|
| 1 | **One code path, two variant spaces.** A frozen `VariantSpace` record is built once per TTL variant; every downstream computation is a loop over `spaces.items()`. | Dual TTL reporting cannot drift between variants, because there is only one implementation. Roughly 600 lines of duplicated cell code never exist. |
| 2 | **Long-format tables with a `variant` column; figures overlay or panel both variants.** | 17 table names and 16 figure names, none variant-suffixed. The manifest stays half the size, and the with/without comparison lives inside one sortable file. |
| 3 | **Every number that reaches `results/` passes through one recorder.** `record_metric` / `record_note` write to an ordered dict *and* to `ResultsWriter`, and `counts.json` is emitted from that same dict. | `counts.json` cannot drift from `manifest.json.metrics`, mirroring `cleaning_report.py`'s guarantee. |
| 4 | **Determinism is designed in, then honestly bounded.** Fixed seeds, an explicit PCA sign convention, a deterministic `eps` rule, one execution pass. And a plain statement of the one thing that is *not* fixed: inline PNG blobs. | `results/` is byte-stable across reruns. The `.ipynb` is not, and the design says so instead of implying otherwise. |

Relationship to the other artifacts: `proposal.md` is the WHY and the inventory; the concurrently
written specs under this change's `specs/` own the Given/When/Then contracts; this document is the
HOW. Where two of them describe the same rule, the proposal is the shared source and neither
overrides it.

### Q1 status (provisional)

Verified against committed source (`src/nids/paths.py:19`, commit `d71e306`):

```python
NOTEBOOK_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
```

`results_dir("01_eda_reduction_clustering")` raises `ValueError` before any filesystem call, because
a leading digit is rejected. `ResultsWriter` validates artifact names against the same pattern
(`ARTIFACT_NAME_PATTERN`), so no table, figure, metric or note name may begin with a digit either.

**This design is written under option A**, pending the user's ruling:

| Item | Value under option A |
|---|---|
| Notebook file | `notebooks/01_eda_reduction_clustering.ipynb` — unchanged, execution order preserved |
| `NOTEBOOK_ID` constant | `"eda_reduction_clustering"` |
| Results folder | `results/eda_reduction_clustering/` |
| Precedent | consistent with the existing `data_cleaning` producer, which also carries no numeric prefix |

If the user rules otherwise, the only change to this design is the value of one notebook constant
(`NOTEBOOK_ID`) and one folder path. Nothing else in this document depends on it.

---

## Architecture Decisions

### Decision 1: The notebook is authored as nbformat v4.5 JSON, with hand-assigned cell ids

**Choice.** `notebooks/01_eda_reduction_clustering.ipynb` is written directly as an nbformat v4.5 JSON
document. Every cell carries an explicit, human-chosen `id`. No conversion step, no generator script
committed to the repository.

Skeleton:

```json
{
  "cells": [ /* ~73 cells */ ],
  "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
```

Cell shapes — markdown cells carry **no** `outputs` and **no** `execution_count`:

```json
{"cell_type": "markdown", "id": "s0_00_title", "metadata": {}, "source": ["# ...\n"]}
{"cell_type": "code", "execution_count": null, "id": "s0_01_imports", "metadata": {}, "outputs": [], "source": ["import ...\n"]}
```

**Cell id convention**: `s<section>_<two-digit ordinal>_<slug>`, e.g. `s0_01_imports`,
`s3_07_fig_pca_scree`. The nbformat 4.5 schema constrains ids to `^[a-zA-Z0-9-_]+$`, 1–64 characters,
and requires them to be unique within the document. These ids are written once and never regenerated,
so they contribute zero diff churn across re-executions.

`"nbformat_minor": 5` is load-bearing: at minor 4 the `id` field is stripped on write, and the
stability guarantee disappears.

**Alternatives considered.**

| Rejected | Failure mode |
|---|---|
| Author a `.py` percent-script and convert with **jupytext** | Adds an undeclared dependency, which the change explicitly forbids. Worse, `jupytext --to notebook` mints a **fresh random cell id on every conversion**, so every regeneration is a whole-file `id` diff — the exact churn the proposal's F4 exists to prevent. |
| Build programmatically with **`nbformat.v4.new_code_cell(...)`** from a committed generator script | Creates two sources of truth. The committed artifact is the executed `.ipynb`, but the reviewable source would be the generator, and the two drift the moment anyone edits the notebook in JupyterLab. `new_code_cell()` without an explicit `id=` also calls `random_cell_id()`, reintroducing the churn. |
| Author interactively in JupyterLab and commit what comes out | Interactive re-running leaves arbitrary, non-sequential `execution_count` values and rewrites a scattered subset of cells per partial run. This is the single largest source of avoidable notebook churn (proposal F4.4). |

**Permitted mechanical aid.** `sdd-apply` MAY use a throwaway assembly script **in the session
scratchpad** to reduce JSON-escaping risk, provided the script is never written into the repository
and the committed `.ipynb` remains the single source of truth. That is a typing aid, not an
architecture.

**Escaping discipline.** `source` is a JSON array of strings, one per source line, each ending in
`\n` except optionally the last. Backslashes and quotes inside Python string literals must be
JSON-escaped. A `nbformat.validate` gate (see [Build and execution procedure](#build-and-execution-procedure))
catches a malformed document before the kernel is ever started.

---

### Decision 2: One code path, two variant spaces — a `VariantSpace` record, not duplicated cells

**Choice.** A frozen dataclass is defined once in the setup section and instantiated twice. Every
analysis cell from section 2 onward is a loop over `spaces.items()`.

```python
VARIANTS: dict[str, bool] = {"with_ttl": True, "without_ttl": False}

@dataclass(frozen=True)
class VariantSpace:
    key: str                     # "with_ttl" | "without_ttl"
    include_ttl: bool
    columns: list[str]           # feature_columns(include_ttl)
    pipeline: Pipeline           # FITTED on train[columns]
    names: list[str]             # pipeline.get_feature_names_out().tolist()
    design: np.ndarray           # (n_rows, ~52) transformed matrix
    modelled: pd.DataFrame       # numeric + skewed block of `design`, by name — VIF / correlation input
    blocks: dict[str, list[str]] # "numeric" | "skewed" | "onehot" -> column names in `names`
```

Construction (one cell, one function, called twice):

```python
spaces = {key: build_variant_space(key, include_ttl) for key, include_ttl in VARIANTS.items()}
```

**Rationale.** The proposal estimates ~1,900 authored lines against a 400-line budget. Literal
duplication would add roughly 600 more and, far worse, would make the two variants independently
editable — a change applied to the `with_ttl` cell and forgotten in the `without_ttl` cell produces
two numbers that look like a finding and are actually a bug. A single parameterised path makes
"both variants were computed identically" structural rather than a claim.

**Alternatives considered.**

| Rejected | Why |
|---|---|
| Literal duplication of every analysis cell | ~600 extra lines, two independently editable code paths, and a class of silent divergence bugs that reads as a substantive finding. |
| `build_preprocessor_pair()` alone, with ad-hoc per-variant locals | The pair factory returns only the two pipelines. Everything downstream (fitted names, design matrix, block map, modelled frame, PCA, scores, labels) still needs a per-variant home. An unstructured set of `*_with_ttl` / `*_without_ttl` locals is what the dataclass replaces. |
| A single wide frame with a `variant` column carried through every computation | PCA, K-Means and DBSCAN each need their own matrix; interleaving two of them in one frame means slicing at every call site instead of once. |

`build_preprocessor_pair()` **is** used — it is exactly the "iterate a dict rather than copy-paste a
cell" affordance the foundation design describes — and `VariantSpace` is the notebook-side record
that carries everything else derived from each pipeline.

**Artifact-name legality under two variants.** Names stay distinct and manifest-legal without a
combinatorial explosion:

| Artifact kind | Variant handling | Name shape |
|---|---|---|
| Tables | long format, `variant` column with values `with_ttl` / `without_ttl` | one name, no suffix |
| Figures with shared axes (scree, cumulative, elbow, silhouette, CH, VIF bars, k-distance) | both variants overlaid as two series in one axes | one name, no suffix |
| Figures with variant-specific coordinate spaces (PC1/PC2 scatter, top loadings) | side-by-side panels within one figure | one name, no suffix |
| Metrics | one registration per variant | `<stem>_with_ttl` / `<stem>_without_ttl` |

`"with_ttl"` sorts before `"without_ttl"` under plain ascending string order (`_` = 0x5F < `o` = 0x6F),
so an ascending `sort_by` on `variant` yields the natural reading order with no special handling.

---

### Decision 3: `ResultsWriter` is opened in setup and closed in the final cell, not held in a `with` block

**Choice.** `writer = ResultsWriter(NOTEBOOK_ID)` in section 0; `writer.close()` in the last code
cell of section 6. No `with` statement.

**Rationale.** A `with` block cannot span notebook cells, and collapsing the whole analysis into one
cell would destroy section-by-section reviewability — which is precisely what makes a ~1,900-line
single-slice PR tolerable (proposal, Effort Estimate).

The guarantee the proposal wanted from the context manager is preserved exactly:

- `ResultsWriter` writes `manifest.json` **only** in `close()` / `__exit__`, and every artifact is
  written to `<path>.tmp` then `os.replace`d.
- `jupyter nbconvert --execute` **stops at the first erroring cell** unless `--allow-errors` is
  passed, and this design never passes it.
- Therefore a failed run never reaches the close cell, never writes a manifest, and leaves the
  unambiguous "no manifest ⇒ incomplete folder" signal the rollback plan relies on.

**Alternatives considered.** A single mega-cell wrapped in `with` — rejected for reviewability.
A `try/finally` spread across cells — impossible; `finally` cannot span cells either.

---

### Decision 4: Subsample rows are located by an explicit `row_position` column, not by index

**Choice.** The subsampler is called on a **two-column index frame**, never on the partition itself:

```python
index_frame = pd.DataFrame({
    "attack_cat": train["attack_cat"].to_numpy(),
    "row_position": np.arange(len(train), dtype=np.int64),
})
sub_default      = stratified_subsample(index_frame, max_rows=10_000, floor=50, seed=42)
sub_proportional = stratified_subsample(index_frame, max_rows=10_000, floor=0,  seed=42)
positions        = sub_default.data["row_position"].to_numpy()
```

Downstream, subsample rows are recovered positionally: `scores_90[positions]`,
`kmeans.labels_[positions]`, `sub_default.data["attack_cat"]`.

**The problem this solves.** `SubsampleResult.data` is built as
`df.iloc[sorted_positions].reset_index(drop=True)` (foundation design §7, step 7). The terminal
`reset_index(drop=True)` **discards** the original index, and `SubsampleResult` exposes no accessor
for `sorted_positions`. So there is no supported way to ask "which rows of the cleaned partition were
drawn?" — which the notebook must know, because K-Means is fitted on the **full** partition while
silhouette is evaluated on the **sample**, using the sample rows' labels *from that full-partition fit*
(proposal F2).

**Why this is safe and does not change the sample.** Allocation depends only on the `attack_cat`
counts and on positional row order; row selection is `rng.choice` over positional indices in sorted
class order. Adding a column changes neither. `cleaned.train` carries a terminal
`reset_index(drop=True)`, so its positional order equals `np.arange(len(train))` exactly — the same
property the foundation names as the subsample determinism mitigation. The index frame therefore
produces a **bit-identical selection** to subsampling the full partition.

**Why the index frame, and not `train.assign(row_position=...)`.** The index frame contains only the
stratification key and a positional integer. It is structurally impossible for it to carry a feature
into anything, and impossible for `row_position` to reach a design matrix, because every pipeline call
site slices `train[feature_columns(...)]` explicitly. A guard assertion pins this
(`"row_position" not in feature_columns(True)`).

**Finding for `project-foundation` (report, do not patch).** `SubsampleResult` has no
`selected_positions` field. Any consumer that fits on a population and evaluates on a sample must
re-derive the mapping. The proposal's rule applies: *"If this notebook needs something the API does
not offer, that is a finding to report, not a patch to apply."* A one-line
`selected_positions: np.ndarray` field on the dataclass would remove the workaround. `src/nids/` is
concurrently owned and out of scope here; this is recorded for the foundation's backlog.

**Alternatives considered.** Re-deriving the drawn rows by reimplementing the largest-remainder
allocation in the notebook — rejected outright; it duplicates tested `nids` logic, which the change
forbids. Merging the subsample back onto the partition on all 41 columns — rejected: the partition is
deduplicated on the full row, so the join is technically unique, but it is an expensive 41-key
`MultiIndex` operation to answer a question an integer column answers exactly.

---

### Decision 5: VIF by an explicit `LinearRegression` loop on the modelled numeric block

**Choice.** The helper the proposal sketched in F1, with three guards, one status column and a fixed
input space.

```python
def variance_inflation_factors(frame: pd.DataFrame) -> pd.DataFrame:
    """VIF = 1 / (1 - R2) per column, regressed on every other column."""
    spread = frame.std(ddof=0)
    constant = [c for c in frame.columns if not np.isfinite(spread[c]) or spread[c] <= 0.0]
    usable = [c for c in frame.columns if c not in constant]

    rows = [
        {"feature": c, "r_squared": float("nan"), "vif": float("nan"),
         "status": "zero_variance"}
        for c in constant
    ]
    design = frame[usable]
    for column in usable:
        others = design.drop(columns=[column])          # notebook-local frame, not a partition
        target = design[column]
        r2 = float(LinearRegression().fit(others, target).score(others, target))
        infinite = (not np.isfinite(r2)) or r2 >= 1.0
        rows.append({
            "feature": column,
            "r_squared": r2,
            "vif": float("inf") if infinite else 1.0 / (1.0 - r2),
            "status": "perfect_collinearity" if infinite else "ok",
        })
    return pd.DataFrame(rows)
```

**Three guards, each closing a real failure.**

| Guard | Failure it closes |
|---|---|
| `r2 >= 1.0` (or non-finite) → `inf`, never divide | Perfectly collinear columns exist in this dataset's neighbourhood (`swin`/`dwin`, the `ct_*` family). A `ZeroDivisionError` in section 2 kills the whole `nbconvert` run; an honest `inf` in a table does not. |
| zero-variance columns dropped **before** the loop, reported with `NaN` and `status="zero_variance"` | `is_sm_ips_ports` is near-constant on this partition. Regressing *on* a constant column yields a meaningless `R²` and can make the design rank-deficient for every other feature too. |
| `status` column | The proposal's F1 requires "a reason". Three values only: `ok`, `zero_variance`, `perfect_collinearity`. |

**Input space — fixed, not guessed.** VIF runs on `space.modelled`: the **numeric + skewed block of
the fitted design matrix**, selected by name from `pipeline.get_feature_names_out()`. Because
`verbose_feature_names_out=False`, those names are the original column names, so the selection is
`[n for n in space.names if n in set(skewed_feature_columns(v)) | set(numeric_feature_columns(v))]`.

- **One-hot columns are excluded by that selection**, exactly as F1 requires: VIF over a dummy block
  is dominated by the mutual-exclusivity constraint and reports structural, not substantive,
  collinearity.
- `log1p` **is** applied (it materially changes linear relationships among the volumetric features,
  and the modelled form is the space PCA and the clustering actually see). Standardisation is present
  but irrelevant — VIF is scale-invariant.
- Column counts: **36** with TTL (15 skewed + 21 plain numeric), **34** without.

**Cost on ~36 features.** Each fit is an OLS solve of an `n × 35` design, `n ≈ 108,000`:
roughly `n·p² ≈ 1.3 × 10⁸` flops, times 36 features, times 2 variants ≈ **10¹⁰ flops** on BLAS —
seconds to low minutes. Transient allocation is one `108k × 35` float64 copy (~30 MB) per iteration,
released each pass. Both are comfortably inside a one-shot `nbconvert` run.

**Alternatives considered.**

| Rejected | Why |
|---|---|
| `statsmodels.stats.outliers_influence.variance_inflation_factor` | Adds a permanent dependency plus `patsy` for one column of arithmetic (proposal Q3/F1). It also requires the caller to add a constant column manually — routinely forgotten, silently inflating every value. `LinearRegression` fits an intercept by default. |
| The exact identity `vif = diag(inv(corr(X)))` | Mathematically the same quantity in one 36×36 inversion, i.e. instant. Rejected because it raises `LinAlgError` under exact singularity, turning the very case the `inf` guard exists for into a dead notebook. The loop degrades gracefully; the inversion does not. |
| Computing VIF on raw (un-`log1p`ed) columns | Would describe a space nothing downstream uses, and would disagree with the correlation matrices (Decision 7) and with PCA. |

**`inf` must not reach JSON as a float.** `json.dumps(float("inf"))` emits the bare token `Infinity`,
which is not valid JSON and would make `manifest.json` / `counts.json` unparseable by a strict reader.
`MetricEntry.value` is typed `int | float | str`, so the rule is:

> `max_vif_{variant}` is registered as the **string** `"inf"` when any feature's VIF is infinite, and
> as the maximum finite float otherwise. The `variance_inflation_factors` table still carries `inf`
> in its `vif` column, where CSV renders it as the text `inf` with no ambiguity.

---

### Decision 6: PCA determinism — `svd_solver="full"` plus a notebook-level sign convention

**Choice.**

```python
pca = PCA(n_components=None, svd_solver="full", random_state=42).fit(space.design)
scores = pca.transform(space.design)
components, scores = fix_component_signs(pca.components_, scores)
```

with an explicit, version-independent sign rule:

```python
def fix_component_signs(components: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Force each component's largest-magnitude loading to be positive.

    Ties on magnitude are broken by the smallest feature index, because
    np.argmax returns the first maximum.
    """
    components = components.copy()
    scores = scores.copy()
    pivots = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), pivots])
    signs[signs == 0.0] = 1.0
    components *= signs[:, None]
    scores *= signs[None, :]
    return components, scores
```

**The exact rule, stated in words:** for component *i*, let *j* be the feature index with the largest
absolute loading (first index wins on ties). If `components_[i, j] < 0`, negate row *i* of
`components_` **and** column *i* of the scores. Zero pivots (a degenerate all-zero trailing component)
are treated as positive.

**Rationale.** `svd_solver="full"` is chosen over `"auto"` because `"auto"` may select the randomized
solver on larger inputs, and `"full"` is exactly deterministic. But `"full"` only pins the *subspace*;
the **sign** of each singular vector is chosen by LAPACK and then post-processed by scikit-learn's
private `svd_flip`, whose `u_based_decision` default has changed across releases. Relying on it makes
the loadings table and the PC1/PC2 scatter hostage to a scikit-learn minor upgrade. The rule above is
computed from the loadings the notebook already holds, so it is stable across LAPACK builds, BLAS
vendors and scikit-learn versions.

**What the flip does and does not change.** It changes only the sign of loadings and of score
coordinates. Explained variance, pairwise distances in PCA space, K-Means inertia, silhouette, DBSCAN
and every cluster assignment are sign-invariant — so the convention exists purely to stop the loadings
table and the scatter figure from mirroring between runs.

**Why scores are indexed, never re-transformed.** `scores` is computed once on the full design matrix
and sign-fixed once; the subsample's scores are the positional row subset `scores[positions]`.
Re-calling `pca.transform` on a subsample would return *unflipped* coordinates and silently disagree
with the flipped loadings. Indexing makes the inconsistency unrepresentable.

`random_state=42` is passed for uniformity with the rest of the notebook and is documented in the cell
as **inert under `svd_solver="full"`** — scikit-learn only consumes it for the randomized solver.
Stating that is more honest than passing it silently and implying it does work.

**Component-count derivation.**

```python
cumulative = np.cumsum(pca.explained_variance_ratio_)
n90 = int(np.searchsorted(cumulative, 0.90, side="left") + 1)
n95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
```

`searchsorted` on a monotone non-decreasing array is exact and avoids the float edge cases of a
boolean `argmax` idiom.

---

### Decision 7: Correlation matrices are computed on the modelled form, with-TTL only

**Choice.** `pearson_correlation_matrix` and `spearman_correlation_matrix` are computed on
`spaces["with_ttl"].modelled` — the 36 numeric + skewed columns **after** `log1p` and standardisation
— and are emitted once, not per variant.

**Rationale, two parts.**

*Why the modelled form.* VIF is explicitly specified on the modelled form (proposal F1), and a reader
who compares "these two features are correlated at 0.97" against "this feature has VIF 40" must be
looking at the same space. Standardisation does not change Pearson at all (it is location- and
scale-invariant), so the only substantive difference from raw units is `log1p` — which is exactly the
transform that makes the volumetric features' linear relationships meaningful. The notebook states
this and adds the corollary worth knowing: **Spearman is identical in either form**, because `log1p`
is strictly increasing and rank correlation is invariant under monotone transforms. Pearson is not.

*Why one variant.* The without-TTL matrix is the with-TTL matrix with two rows and two columns
deleted; every remaining cell is numerically identical because Pearson and Spearman are pairwise. A
second heatmap would carry zero new information. VIF is the opposite case — it is multivariate, so
removing two columns changes every other feature's value — which is why VIF *is* dual-reported. The
asymmetry is itself a section 2 finding and the notebook says so in prose.

**Alternatives considered.** Raw-unit matrices — rejected: they would describe a space nothing
downstream uses and would disagree with VIF and PCA. Per-variant matrices — rejected: two extra
figures and two extra 36×36 CSVs to show identical numbers.

---

### Decision 8: Selected `k` is `argmax` silhouette; section 5 reuses the sweep's fitted estimator

**Choice.**

```python
K_RANGE = range(2, 13)                       # k = 2 ... 12
kmeans_fits[(variant, k)] = KMeans(n_clusters=k, n_init=10, random_state=42).fit(scores_90)
selected_k[variant] = min(
    K_RANGE, key=lambda k: (-sweep[(variant, k)].silhouette, k)
)                                            # argmax silhouette, ties -> smaller k
final_labels[variant] = kmeans_fits[(variant, selected_k[variant])].labels_
```

**Where each metric is computed — the asymmetry matters and is disclosed.**

| Metric | Computed on | Why |
|---|---|---|
| `inertia` | **full** cleaned partition (`kmeans.inertia_`) | Free — it is the fitted attribute. |
| `calinski_harabasz` | **full** cleaned partition | `O(n·d)`. There is no reason to sample, so it is a population fact. |
| `silhouette` | **`floor=50` stratified subsample**, labels taken from the full-partition fit | `O(n²)` pairwise distances: ~1.2 × 10¹⁰ pairs on 108k rows. `silhouette_score`'s chunking bounds memory, not time (proposal F2). |

The `kmeans_k_sweep_metrics` manifest description states this split explicitly, and the
`silhouette_is_subsample_estimate` note repeats it with the sample size, stratification key and floor.

**Why argmax silhouette and not an inertia knee.** Silhouette is the only one of the three metrics
with an absolute scale and a defined optimum. Inertia decreases monotonically in `k`, so selecting
from it requires a knee heuristic whose answer moves when the k-range endpoints move — an unstable
rule for a committed artifact. Calinski-Harabasz has an optimum but is strongly biased toward small
`k` on elongated clusters. `argmax` over a bounded metric with a stated tie-break is deterministic and
one line.

**The honest caveat, and its mitigation.** Because silhouette is measured on the `floor=50` sample,
the selection inherits that sample's class-rebalanced composition. That is exactly why, **at the
selected `k` only**, silhouette is recomputed on the `floor=0` proportional subsample
(`sub_proportional`), which is population-representative in class mix. Both numbers are registered
(`kmeans_silhouette_at_selected_k_{v}`, `kmeans_silhouette_floor0_sensitivity_{v}`). If they disagree
materially, that disagreement is a section 6 finding — not a number to quietly prefer.

**Section 5 refits nothing.** All 22 fitted estimators are retained in `kmeans_fits`, so profiling
reads `kmeans_fits[(v, selected_k[v])].labels_` directly. Memory is ~22 × 108k int32 ≈ 10 MB. A refit
would waste minutes and, if any parameter drifted between the two call sites, would silently produce
labels that disagree with the sweep the notebook just reported.

**Degenerate-case guard.** If `len(np.unique(labels[positions])) < 2` for some `k`, `silhouette_score`
raises. The sweep records `NaN` for that cell instead and continues. On 10,000 rows and `k ≥ 2` this
is unreachable; the guard exists so an unreachable case cannot end the run.

**Runtime expectation for the sweep — stated as an estimate, with the arithmetic.** One Lloyd
iteration costs `O(n·k·d)` ≈ `108,000 × 12 × 15` ≈ 2 × 10⁷ flops; at ~50 iterations × `n_init=10`
that is ~10¹⁰ flops per `k`, over 11 values of `k` and 2 variants. On scikit-learn's threaded Cython
kernel this lands at roughly **3–12 minutes total**. Two process rules follow:

- If the sweep exceeds **~15 minutes**, `sdd-apply` reports it as a finding. It does **not** switch to
  `MiniBatchKMeans`, which optimises a different objective and would make the reported inertia
  non-comparable to everything else in the table.
- The measured wall time is reported in the apply report, **never registered as a metric**. A
  wall-clock number in a committed `results/` file is nondeterministic churn by construction.

---

### Decision 9: DBSCAN `eps` from a normalised-axis chord rule, with three degenerate-case guards

**Choice.** DBSCAN runs on `scores_90[positions]` — the same ≤10,000-row `floor=50` stratified
subsample, in the same PCA space, per variant. `min_samples = 2 × n_components_90[variant]`
(the Sander et al. heuristic for `d`-dimensional data), registered as a metric per variant rather
than buried in a call.

**The `eps` rule, written as implementable code.**

```python
k = min_samples
distances, _ = NearestNeighbors(n_neighbors=k).fit(sample).kneighbors(sample)
kdist = np.sort(distances[:, k - 1])                 # ascending, length m; column 0 is the self-distance
m = kdist.size

x = np.arange(m, dtype=np.float64) / (m - 1)         # normalise both axes to [0, 1]
span = kdist[-1] - kdist[0]
y = (kdist - kdist[0]) / span

dx, dy = x[-1] - x[0], y[-1] - y[0]                  # chord from the first to the last point
perpendicular = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / np.hypot(dx, dy)
eps = float(kdist[int(np.argmax(perpendicular))])    # np.argmax -> first maximum, deterministic tie-break
```

**Why both axes are normalised to [0, 1] first.** On raw axes the index runs 0…10⁴ while the distance
runs 0…~10, so the chord is nearly horizontal and "maximum perpendicular distance" silently degenerates
into "maximum vertical gap" with an arbitrary implicit weighting. Normalising makes the rule
scale-free and matches the geometric intent the kneedle method formalises. It changes the selected
point, so it is stated rather than left implicit.

**Three guards.**

| Condition | Behaviour |
|---|---|
| `span == 0` (flat k-distance curve — every point equidistant) | `eps = float(np.median(kdist))`; the `dbscan_eps_not_transferable` note records that the chord rule degenerated. Division by zero is impossible. |
| Selected `eps == 0` (the knee falls inside a block of exact duplicate rows in PCA space) | `eps = float(kdist[kdist > 0].min())`. `sklearn.cluster.DBSCAN` raises `ValueError` for `eps <= 0`, so this is a real failure path: the partition is deduplicated on the full 41-column row, but two rows differing **only** in `attack_cat`/`label` are distinct rows with an identical ~52-column design vector. |
| No positive k-distance at all | DBSCAN is skipped for that variant; `dbscan_cluster_count_{v}` and `dbscan_noise_fraction_{v}` are registered as `NaN` and the note says so. Unreachable in practice; present so it cannot end the run. |

**Why DBSCAN is sampled at all, and what that costs.** On ~108,000 rows in ~10 PCA dimensions the
tree-based neighbour search degrades (ball-tree performance falls off well before ten dimensions),
and memory is driven by neighbourhood *size*, not row count — this partition has large near-duplicate
dense regions, so an `eps` slightly too large produces neighbour lists that can exhaust memory
mid-notebook. That failure mode is unbounded and would take the whole run down. A 10,000-row bound
makes the cost predictable.

The price is disclosed in two mandatory notes: `dbscan_is_subsample_property` (the noise fraction,
cluster count and cluster sizes are properties of that subsample, not of the 108k-row partition) and
`dbscan_eps_not_transferable` (local density scales with `n`; an `eps` calibrated on 10,000 rows is
too large for the full partition and must not be lifted into another notebook).

The selected value is drawn on `dbscan_k_distance_plot` as a horizontal reference line, so a reader
can disagree with it from the picture.

---

### Decision 10: Figure legibility — opaque white canvas, one frozen class palette, symmetric diverging limits

**Choice**, set once in the setup cell and never overridden per figure:

```python
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.dpi": 100,              # inline blob size fixed
    "figure.facecolor": "white",    # opaque, never transparent
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "font.family": "DejaVu Sans",   # ships with matplotlib; pins text metrics
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.frameon": True,
})
CLASS_ORDER  = sorted(train["attack_cat"].unique().tolist())
CLASS_COLORS = {c: plt.get_cmap("tab10")(i) for i, c in enumerate(CLASS_ORDER)}
```

**Light-and-dark readability, stated honestly.** The thing that breaks a PNG in dark mode is a
*transparent* background: dark text lands on a dark canvas and disappears. An **opaque white canvas
with near-black text is legible in both contexts** — merely bright in dark mode. That is the
achievable answer in matplotlib without shipping two renderings of every figure, and shipping two
would double the 16-figure inventory for a cosmetic gain. So: opaque white, always.

**`attack_cat` at 10 classes.** `tab10` has exactly ten categorically distinct, deuteranopia-tolerant
colours. The mapping is built **once** from the sorted class list, so a class keeps the same colour in
every figure that shows it — the distribution bars, the boxplots and the PC1/PC2 scatter all agree.
A guard (`assert len(CLASS_ORDER) <= 10`, else fall back to `tab20`) keeps a surprise eleventh class
from silently recycling a colour. Legends sit outside the axes with `markerscale=4` so a 5-point
scatter marker is identifiable in the legend.

**Diverging heatmaps centred exactly at 0.** `cmap="RdBu_r"`, `vmin=-1`, `vmax=1`, `center=0`. The
symmetric limits are what guarantee 0 maps to the palette's midpoint exactly — `center=0` alone would
still be honoured, but symmetric limits make it arithmetically unavoidable and make the two heatmaps
directly comparable. Red/blue is CVD-tolerant (it is red/green that fails). The 36 × 36 grid is
**not** annotated: 1,296 printed numbers are unreadable and would bloat the PNG; the numbers live in
the CSV, which is what the CSV is for. Colorbar labels are spelled out
("Pearson correlation coefficient", "Spearman rank correlation coefficient").

**Overplotting order on the scatter is deterministic.** Classes are drawn in **descending class size,
ties broken by name ascending** (`sorted(CLASS_ORDER, key=lambda c: (-counts[c], c))`), so minority
attack families land on top and stay visible, and the draw order is reproducible rather than dependent
on `value_counts()` tie behaviour.

**Boxplots omit individual outliers.** `showfliers=False, whis=(1, 99)`: 108,000 rows × 6 features ×
10 classes of outlier dots would be an unreadable figure and a multi-megabyte PNG. Whiskers at the
1st and 99th percentiles convey the tail; the figure caption and the markdown say so explicitly
rather than letting a reader assume Tukey whiskers.

**Zeros on a log axis.** Several key features contain exact zeros, which a `log` axis silently drops —
data loss inside a figure. Every such axis uses `symlog` with `linthresh=1` and is labelled
"(symlog scale, linear below 1)".

---

### Decision 11: `counts.json` is emitted from the same ordered dict that feeds `add_metric`

**Choice.** Two thin recorders, defined once:

```python
METRICS: dict[str, tuple[int | float | str, str]] = {}
NOTES: dict[str, str] = {}

def record_metric(name: str, value: int | float | str, description: str) -> None:
    if name in METRICS:
        raise ValueError(f"Metric {name!r} already registered.")
    METRICS[name] = (value, description)
    writer.add_metric(name, value, description)

def record_note(note_id: str, text: str) -> None:
    if note_id in NOTES:
        raise ValueError(f"Note {note_id!r} already registered.")
    NOTES[note_id] = text
    writer.add_note(note_id, text)
```

and, in the final cell, before `close()`:

```python
writer.write_json({"metrics": {k: v for k, (v, _) in METRICS.items()}, "notes": NOTES}, name="counts")
```

**Rationale.** `ResultsWriter` exposes no public accessor for its registered metric list, so a
notebook that builds `counts.json` independently would be maintaining a second hand-typed copy that
drifts on the first edit. Emitting both from one ordered dict reproduces exactly the guarantee
`cleaning_report.py` has — *"`counts.json` and `manifest.json.metrics` are emitted from the same
`MetricEntry` list, so the human-readable mirror cannot drift from the machine-readable source of
truth"* — and the two producer folders then read the same way, which is the whole point of the output
contract generalising. The duplicate-name check is a cheap local echo of the writer's own
`ValueError`, raised at the call site where the name is visible.

---

### Decision 12: Metric inventory made symmetric across variants — 29 metrics, not ~20

**Choice.** Every metric whose value depends on the TTL toggle is registered twice, with a
`_with_ttl` / `_without_ttl` suffix. Full inventory in
[ResultsWriter registration plan](#resultswriter-registration-plan).

**Rationale.** The proposal's illustrative list registers the DBSCAN family
(`dbscan_eps_with_ttl`, `dbscan_noise_fraction_with_ttl`, `dbscan_cluster_count_with_ttl`), the
silhouette-at-selected-k pair and `onehot_share_of_total_variance_with_ttl` for the **with-TTL variant
only**, while `dbscan_min_samples` carries no suffix at all — but `min_samples = 2 × n_components_90`
and `n_components_90` differs between variants, so an unsuffixed name cannot hold both values
truthfully. Leaving the without-TTL DBSCAN parameters unregistered would also contradict AGENTS.md's
hard rule that results are reported with and without the TTL shortcut pair.

Symmetry costs 11 extra `add_metric` calls (~55 lines) and buys a metric surface where every name has
a variant twin. Count moves from the proposal's "~20" to **29** (7 variant-independent + 11 × 2).

One metric named in the proposal is **deliberately not registered**: the design-matrix column count.
It belongs in `pca_variance_by_feature_block` as an `n_columns` column, where it sits next to the
variance share it explains, rather than floating as a standalone number.

---

### Decision 13: Section 6 is finalised by a second execution, and the committed state is still one clean pass

**Choice — the authoring loop `sdd-apply` follows:**

1. Author all ~73 cells. Section 6's narrative markdown cells are written with the *questions* and the
   *rules of interpretation*, not yet with numbers. Section 6 also contains a code cell that **prints**
   a compact findings frame assembled from `METRICS`.
2. `nbformat.validate`, then execute once.
3. Read the printed findings. Rewrite section 6's narrative markdown cells with the observed numbers
   and the actual conclusions.
4. Execute once more. **This is the committed state.**

**Why this still satisfies "a single `nbconvert` pass renumbering 1…N".** Markdown cells carry no
`execution_count`. Editing them between two full `--execute --inplace` passes leaves the final file
with code-cell execution counts `1…N` in order, which is exactly the invariant proposal F4.4 names.
The rule being protected is "never commit the output of interactive partial re-runs", and this loop
never produces one.

**Why not generate section 6 from a template.** An f-string narrative rendered through
`display(Markdown(...))` would be single-pass and could not drift from the numbers — but "plain-language
conclusions" is a named deliverable, and template prose is not a conclusion. The compromise keeps both:
the *mechanical* summary is generated and cannot drift; the *judgment* is authored and reviewed as
prose.

**Cost, stated.** Step 4 rewrites every inline PNG blob. That is unavoidable — see
[What is honestly not fixed](#what-is-honestly-not-fixed) — and the cost of one extra rewrite of a
file that is being created in this same commit is zero.

---

## The `nids` API surface consumed

Signatures below are taken from `openspec/changes/project-foundation/design.md`; the two slice-1
modules were additionally read from committed source (`src/nids/paths.py`, `src/nids/columns.py` at
`d71e306`) and match. **`sdd-apply` MUST re-verify all five consumed signatures against committed
source before writing its first cell** — slices 3–5 are still uncommitted and concurrently owned.

| Call | Exact signature | Returns | How the notebook threads it |
|---|---|---|---|
| `load_clean_partitions` | `(*, comparison_key: LeakageKey = "full_row", raw_dir: Path \| None = None) -> CleaningResult` | frozen dataclass with `.train`, `.test` and nine diagnostics | Called **once**, defaults. `train = cleaned.train`, then `del cleaned` — see the test-partition guarantee below. |
| `feature_columns` | `(include_ttl: bool = True) -> list[str]` | 39 names (`True`) / 37 names (`False`), in `FEATURE_COLUMNS` order | The only source of the feature allow-list. Called with both toggles; the result becomes `VariantSpace.columns`. |
| `numeric_feature_columns` / `skewed_feature_columns` | `(include_ttl: bool = True) -> list[str]` | 21 / 15 with TTL; 19 / 15 without | Builds `VariantSpace.blocks`, which drives the VIF input selection, the F6 variance decomposition and the loading-bar colouring. |
| `build_preprocessor` | `(include_ttl_features: bool = True) -> Pipeline` | an **unfitted** `Pipeline([("features", ColumnTransformer(...))])` | Obtained via `build_preprocessor_pair()`; each is fitted as `pipe.fit(train[feature_columns(v)])` — the frame is named at the call site, which is the line the AGENTS.md review checklist greps for. |
| `build_preprocessor_pair` | `() -> dict[str, Pipeline]` | `{"with_ttl": Pipeline, "without_ttl": Pipeline}` | Supplies both unfitted pipelines with keys that match `VARIANTS` exactly. |
| `stratified_subsample` | `(df, *, stratify_by="attack_cat", max_rows=10_000, floor=50, seed=42) -> SubsampleResult` | frozen dataclass: `.data` (reset index), `.allocations`, `.floored_classes`, `.to_frame()` | Called **twice** on `index_frame` (Decision 4): `floor=50` for silhouette / DBSCAN / the scatter, `floor=0` for the sensitivity silhouette. |
| `ResultsWriter` | `(notebook_id, *, root=None, generated_at=None, prune_undeclared=True)` | writer object; `directory` property | Opened in setup, closed last (Decision 3). |
| `ResultsWriter.add_table` | `(frame, *, name, title, description, sort_by=None) -> Path` | the written path | 17 calls. `sort_by` is **always** supplied with an ascending total key. |
| `ResultsWriter.add_figure` | `(figure, *, name, title, description) -> Path` | the written path | 16 calls. PNG at dpi 150 with `metadata={"Software": None, "Creation Time": None}`. |
| `ResultsWriter.add_metric` / `add_note` | `(name, value, description) -> None` / `(note_id, text) -> None` | `None` | Reached only through `record_metric` / `record_note` (Decision 11). |
| `ResultsWriter.write_json` | `(payload, *, name) -> Path` | the written path | One call: the unregistered `counts.json` sidecar. |
| `ResultsWriter.close` | `() -> Path` | the manifest path | One call, last code cell. Writes `manifest.json` and prunes undeclared files. |
| `validate_raw_data` | `(raw_dir: Path \| None = None) -> ValidationReport` | report with `.ok` and `.render()` | First data cell. `assert report.ok` so a truncated or replaced CSV fails at cell 5 rather than halfway through section 3. |

**What the notebook is forbidden to contain**, because `nids` already owns it — enforced by the
proposal's `rg` success criterion: `pd.read_csv`, `.duplicated(`, `.drop(columns=` **on a partition
frame**, `StandardScaler(`, `OneHotEncoder(`, `np.log1p` on a partition column, any `"-"` → `"none"`
mapping, any rare-category threshold, any `to_csv` / `savefig` outside `ResultsWriter`.

> **One deliberate, narrow exception to the `drop(columns=` grep**: the VIF helper calls
> `design.drop(columns=[column])` on a **notebook-local transformed frame**, never on a partition.
> The grep as written in the proposal's Success Criteria would flag it. `sdd-apply` must either scope
> the grep to exclude the VIF cell or rephrase the VIF loop as a positional selection
> (`design[[c for c in usable if c != column]]`). **This design chooses the rephrasing**, so the grep
> stays literal and the criterion needs no footnote.

---

## Data Flow

```
data/raw/*.csv                    READ ONLY — reached only through nids.validation / nids.data
      |
      v
nids.validation.validate_raw_data() --> assert report.ok        [cell s0_05]
      |
      v
nids.data.load_clean_partitions()  -> CleaningResult
      |
      |  train = cleaned.train ; del cleaned                    [cell s0_06]
      |  (the test partition becomes unreachable from this kernel)
      v
   train  (~108k rows x 41 columns)
      |
      +--> train["attack_cat"] --------------------------+  descriptive tables, bars,
      |                                                  |  boxplot grouping, scatter colour,
      |                                                  |  cluster composition   (POST HOC ONLY)
      |                                                  |
      +--> index_frame(attack_cat, row_position)         |
      |        |                                         |
      |        +--> stratified_subsample(floor=50)  -> positions, sub_default
      |        +--> stratified_subsample(floor=0)   -> sub_proportional
      |                                                  |
      +--> train[feature_columns(v)]                     |
               |                                         |
               v                                         |
        build_preprocessor_pair()[v].fit(...)            |
               |                                         |
               v                                         |
        VariantSpace(design ~52 cols, names, blocks, modelled 36/34 cols)
               |                                         |
      +--------+---------+-----------------+             |
      |                  |                 |             |
      v                  v                 v             |
  correlation          VIF          PCA(full) -> sign fix -> scores
  (with_ttl only)   (both)                |               |
      |                  |                 |             |
      |                  |            scores[:, :n90]    |
      |                  |                 |             |
      |                  |     +-----------+---------+   |
      |                  |     |                     |   |
      |                  |     v                     v   |
      |                  |  KMeans k=2..12      scores[positions]
      |                  |  (FULL partition)         |   |
      |                  |     |                     v   |
      |                  |     |            silhouette / DBSCAN  (SUBSAMPLE)
      |                  |     v                         |
      |                  |  final_labels[v] ------> cluster profiling <-+
      |                  |                                 |
      v                  v                                 v
                    nids.results.ResultsWriter
                              |
                              v
              results/eda_reduction_clustering/
                  tables/*.csv   (17)
                  figures/*.png  (16)
                  counts.json    (sidecar, from METRICS)
                  manifest.json  (written LAST, by close())
```

The two arrows that matter for review: **`.fit` is only ever drawn from `train`**, and **`attack_cat`
never enters any arrow that leads into a `.fit`**.

---

## Cell-by-cell outline

73 cells. `md` = markdown, `code` = code. "Registers" names the `ResultsWriter` artifacts the cell
produces. This outline is the spine `sdd-apply` follows; cell ids are final.

### Section 0 — Setup and guards (14 cells)

| id | type | Purpose | `nids` API | Registers |
|---|---|---|---|---|
| `s0_00_title` | md | Title, the three questions the notebook answers, how to re-run it, the `results/` folder name | — | — |
| `s0_01_imports` | code | `dataclasses`, `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, the five sklearn imports, the `nids` imports | all module imports | — |
| `s0_02_style` | code | `%matplotlib inline`, `sns.set_theme`, `plt.rcParams` block, `pd.set_option` display limits (Decision 10) | — | — |
| `s0_03_constants` | code | `NOTEBOOK_ID`, `SEED = 42`, `VARIANTS`, `K_RANGE = range(2, 13)`, `VARIANCE_THRESHOLDS = (0.90, 0.95)`, `KEY_FEATURES`, `MAX_ROWS`, `FLOOR` | — | — |
| `s0_04_data_guard_md` | md | Why raw validation runs before anything else | — | — |
| `s0_05_validate` | code | `report = validate_raw_data()`; `print(report.render())`; `assert report.ok` | `validation.validate_raw_data` | — |
| `s0_06_load` | code | `cleaned = load_clean_partitions()`; `train = cleaned.train`; `del cleaned`; `assert "cleaned" not in globals()` | `data.load_clean_partitions` | — |
| `s0_07_boundary_md` | md | The `attack_cat` boundary: four legitimate uses, three structural guarantees | — | — |
| `s0_08_assert_labels` | code | **Runtime assertion cell 1** (see [The `attack_cat` boundary](#the-attack_cat-boundary)) | `columns.feature_columns` | — |
| `s0_09_helpers` | code | `VariantSpace`, `build_variant_space`, `variance_inflation_factors`, `fix_component_signs`, `select_eps`, `record_metric`, `record_note`, `new_figure` | `columns.*` | — |
| `s0_10_spaces` | code | `pipelines = build_preprocessor_pair()`; fit each on `train[feature_columns(v)]`; build `spaces` | `preprocessing.build_preprocessor_pair` | — |
| `s0_11_assert_matrix` | code | **Runtime assertion cell 2** | — | — |
| `s0_12_subsamples` | code | `index_frame`, `sub_default`, `sub_proportional`, `positions` (Decision 4) | `sampling.stratified_subsample` | — |
| `s0_13_writer` | code | `writer = ResultsWriter(NOTEBOOK_ID)`; print `writer.directory`; register the four shape metrics and the two sample-size metrics | `results.ResultsWriter` | metrics `train_rows`, `attack_cat_classes`, `feature_columns_with_ttl`, `feature_columns_without_ttl`, `silhouette_sample_rows`, `silhouette_sensitivity_sample_rows` |

### Section 1 — Descriptive statistics (13 cells)

| id | type | Purpose | Registers |
|---|---|---|---|
| `s1_00_header` | md | What section 1 establishes; why it overlaps `results/data_cleaning/` deliberately (each producer folder must be interpretable in isolation) | — |
| `s1_01_shape` | code | Column inventory with dtype, non-null count and role | table `partition_shape_and_dtypes` |
| `s1_02_numeric_summary` | code | `describe` extended with median, skew and `mean_median_ratio` (raw units) | table `numeric_summary_mean_vs_median` |
| `s1_03_categorical` | code | `value_counts` for `proto`, `service`, `state`, long format | table `categorical_value_counts` |
| `s1_04_attack_cat` | code | Class balance of the cleaned training partition | table `attack_cat_distribution` |
| `s1_05_imbalance_md` | md | Reading the imbalance; why the bar chart is on a log axis | — |
| `s1_06_fig_attack_bars` | code | Sorted horizontal bars, log x | figure `attack_cat_distribution_bars` |
| `s1_07_skew_md` | md | Right skew, the `log1p` rationale, and that the ECDF is invariant under `log1p` | — |
| `s1_08_fig_hist_raw` | code | 2 × 3 histogram grid, raw units, log y | figure `key_feature_histograms_raw` |
| `s1_09_fig_hist_log1p` | code | Same grid after `log1p` (computed on a notebook-local array, never on a partition column) | figure `key_feature_histograms_log1p` |
| `s1_10_fig_ecdf` | code | Two panels: the same six ECDFs under raw (symlog) and `log1p` (linear) x-axes | figure `key_feature_ecdf_raw_vs_log1p` |
| `s1_11_fig_boxplots` | code | 3 × 2 horizontal boxplot grid by `attack_cat`, symlog x, `showfliers=False`, `whis=(1, 99)` | figure `key_feature_boxplots_by_attack_cat` |
| `s1_12_findings` | md | What section 1 establishes, in prose | — |

> **`KEY_FEATURES` is a frozen constant**, not a runtime selection:
> `("dur", "sbytes", "dbytes", "rate", "sload", "dload")`. All six are in `SKEWED_COLUMNS`, hence
> non-negative, hence inside `log1p`'s domain. Freezing it mirrors the foundation's Decision 6
> reasoning: a runtime "top-6 by measured skew" selection would make the figure inventory data-derived
> and could reshuffle panels between runs.

### Section 2 — Correlation and collinearity (9 cells)

| id | type | Purpose | Registers |
|---|---|---|---|
| `s2_00_header` | md | Scope; **why the matrices are single-variant and VIF is dual-variant** (Decision 7) | — |
| `s2_01_matrices` | code | Pearson and Spearman on `spaces["with_ttl"].modelled`; both 36 × 36 | tables `pearson_correlation_matrix`, `spearman_correlation_matrix` |
| `s2_02_fig_pearson` | code | Diverging heatmap, `RdBu_r`, `vmin=-1`, `vmax=1`, `center=0`, unannotated | figure `pearson_correlation_heatmap` |
| `s2_03_fig_spearman` | code | Same palette and limits | figure `spearman_correlation_heatmap` |
| `s2_04_pairs` | code | Upper-triangle pairs with \|r\| > 0.9 under either method; `method_flagged` ∈ {`pearson`, `spearman`, `both`} | table `high_correlation_pairs`; metric `correlated_pairs_above_0_9` |
| `s2_05_vif_md` | md | VIF definition, the three guards, why one-hot columns are excluded, why the modelled form | — |
| `s2_06_vif` | code | `variance_inflation_factors` over both variants, concatenated with a `variant` column | table `variance_inflation_factors`; metrics `max_vif_with_ttl`, `max_vif_without_ttl` |
| `s2_07_fig_vif` | code | Grouped bars per feature, both variants, log y, infinite values clipped and hatched | figure `variance_inflation_factors_bars` |
| `s2_08_findings` | md | Which feature families are redundant; how the TTL toggle moved every other feature's VIF | — |

### Section 3 — Dimensionality reduction (12 cells)

| id | type | Purpose | Registers |
|---|---|---|---|
| `s3_00_header` | md | What PCA is being asked; the design-matrix width per variant | — |
| `s3_01_fit` | code | `PCA(svd_solver="full", random_state=42)` per variant, sign fix, store `components`, `scores`, `n90`, `n95` (Decision 6) | — |
| `s3_02_explained` | code | Per-component explained and cumulative variance, both variants | table `pca_explained_variance` |
| `s3_03_thresholds` | code | Components reaching 90% and 95% | table `pca_components_for_variance`; metrics `pca_components_90_{v}`, `pca_components_95_{v}` |
| `s3_04_fig_scree` | code | Explained variance ratio per component, both variants overlaid | figure `pca_scree_plot` |
| `s3_05_fig_cumulative` | code | Cumulative curve with 90% and 95% horizontal reference lines, both variants | figure `pca_cumulative_variance` |
| `s3_06_loadings` | code | Top 15 loadings by \|value\| for PC1 and PC2, per variant | table `pca_top_loadings_pc1_pc2` |
| `s3_07_fig_loadings` | code | 2 × 2 panel grid of horizontal loading bars, coloured by feature block | figure `pca_top_loadings_pc1_pc2_bars` |
| `s3_08_onehot_md` | md | **F6**: the one-hot variance asymmetry, why it is measured rather than fixed | — |
| `s3_09_blocks` | code | Trace decomposition plus PC1/PC2 loading-square shares (see [F6 measurement](#f6-measurement-the-one-hot-variance-asymmetry)) | table `pca_variance_by_feature_block`; metrics `onehot_share_of_total_variance_{v}`; note `onehot_variance_asymmetry` |
| `s3_10_fig_scatter` | code | 1 × 2 panels (one per variant), subsample rows, coloured post hoc by `attack_cat` | figure `pca_scatter_pc1_pc2_by_attack_cat` |
| `s3_11_findings` | md | How many real degrees of freedom; how the TTL toggle changed the component count | — |

### Section 4 — Clustering on the reduced space (13 cells)

| id | type | Purpose | Registers |
|---|---|---|---|
| `s4_00_header` | md | The fit-on-full / evaluate-on-sample split, stated before any number appears | — |
| `s4_01_sweep` | code | `KMeans(n_clusters=k, n_init=10, random_state=42)` for `k = 2…12` × 2 variants on `scores[:, :n90]`; inertia and CH on the full partition, silhouette on `positions` | — |
| `s4_02_sweep_table` | code | The sweep, long format | table `kmeans_k_sweep_metrics`; note `silhouette_is_subsample_estimate` |
| `s4_03_fig_elbow` | code | Inertia vs `k`, both variants | figure `kmeans_elbow_inertia` |
| `s4_04_fig_silhouette` | code | Silhouette vs `k`, both variants, selected `k` marked | figure `kmeans_silhouette_by_k` |
| `s4_05_fig_ch` | code | Calinski-Harabasz vs `k`, both variants | figure `kmeans_calinski_harabasz_by_k` |
| `s4_06_select_k` | code | `argmax` silhouette with the tie-break; `floor=0` sensitivity silhouette at the selected `k` (Decision 8) | metrics `kmeans_selected_k_{v}`, `kmeans_silhouette_at_selected_k_{v}`, `kmeans_silhouette_floor0_sensitivity_{v}` |
| `s4_07_dbscan_md` | md | Why DBSCAN is sampled; the `eps` rule written out; `min_samples = 2 × n_components_90` | — |
| `s4_08_eps` | code | k-distance curve and `select_eps` per variant, with the three guards (Decision 9) | metrics `dbscan_eps_{v}`, `dbscan_min_samples_{v}` |
| `s4_09_fig_kdist` | code | Sorted k-distance curves, both variants, selected `eps` as a horizontal reference line | figure `dbscan_k_distance_plot` |
| `s4_10_dbscan` | code | `DBSCAN(eps=..., min_samples=...)` per variant on `scores[positions, :n90]`; cluster sizes including the `-1` noise label | table `dbscan_cluster_summary`; metrics `dbscan_cluster_count_{v}`, `dbscan_noise_fraction_{v}`; notes `dbscan_is_subsample_property`, `dbscan_eps_not_transferable` |
| `s4_11_allocation` | code | `sub_default.to_frame()` and `sub_proportional.to_frame()`, concatenated with a `sampler` column, `label` renamed `attack_cat` | table `clustering_sample_allocation` |
| `s4_12_findings` | md | Whether unsupervised structure exists; whether the two variants agree on `k` | — |

### Section 5 — Cluster profiling (5 cells)

| id | type | Purpose | Registers |
|---|---|---|---|
| `s5_00_header` | md | Profiling is post hoc; no metric is optimised against `attack_cat`; ARI/NMI are notebook 2's | — |
| `s5_01_labels` | code | `final_labels[v] = kmeans_fits[(v, selected_k[v])].labels_` — **reuse, no refit** (Decision 8) | — |
| `s5_02_profile` | code | Per-cluster mean of the **original, unscaled** numeric + skewed features against the global mean | table `cluster_profile_original_units` |
| `s5_03_composition` | code | `attack_cat` composition per cluster, `share_of_cluster` and `share_of_class`, all combinations including zeros | table `cluster_attack_cat_composition`; note `attack_cat_is_post_hoc_only` |
| `s5_04_findings` | md | What the clusters appear to represent, in original units | — |

### Section 6 — Conclusions and close (7 cells)

| id | type | Purpose | Registers |
|---|---|---|---|
| `s6_00_header` | md | How to read this section | — |
| `s6_01_summary` | code | A compact decisive-metrics frame assembled from `METRICS` and **printed only** (not registered) | note `ttl_shortcut_effect` |
| `s6_02_structure` | md | What structure exists — authored after the first execution (Decision 13) | — |
| `s6_03_clusters` | md | What the clusters appear to represent — authored after the first execution | — |
| `s6_04_limitations` | md | Subsample caveats, the one-hot asymmetry, the TTL effect, what this notebook does **not** claim | — |
| `s6_05_close` | code | `writer.write_json(..., name="counts")`; `manifest = writer.close()`; re-read the manifest and assert every declared path resolves | — |
| `s6_06_rerun` | md | The exact re-run command and what a clean re-run is expected to change (one line of `manifest.json`) | — |

---

## The `attack_cat` boundary

AGENTS.md's blocking rule is *"any feature derived from the target is a blocking issue."*
`attack_cat` legitimately appears four times in this notebook, so the boundary must be unambiguous.

### Four legitimate uses

| Use | Section | Why it is legitimate |
|---|---|---|
| Class-distribution table, bar chart, boxplot grouping | 1 | Describing the label is the point of a descriptive section. It is not fed to anything. |
| `stratify_by="attack_cat"` in `stratified_subsample` | 0, 4 | The label selects *which rows are evaluated*, never what the clustering learns. Disclosed; see the caveat below. |
| Colouring the PC1/PC2 scatter | 3 | Post-hoc colouring of an already-computed projection. PCA was fitted before the colour was chosen. |
| Cluster composition cross-tabulation | 5 | Post-hoc. No metric is optimised against it; ARI and NMI are explicitly notebook 2's. |

### Three structural guarantees

| Forbidden use | Structural guarantee |
|---|---|
| As an input feature | `feature_columns()` is an explicit **allow-list** of 39 names (read from committed `src/nids/columns.py`), not `set(all) − set(targets)`. A skipped drop cannot reintroduce a target. |
| Reaching the design matrix by accident | `ColumnTransformer(remainder="drop")` drops `attack_cat` and `label` even if the full 41-column frame is passed to `fit`. |
| As a fitting target | Nothing in this notebook calls `.fit(X, y)`. PCA, K-Means and DBSCAN are all fitted on `X` alone. |

### Two runtime assertion cells

These prove the boundary **inside the executed notebook**, so the evidence is the committed output
rather than a claim in prose.

**Cell `s0_08_assert_labels`** — the allow-list and the test-partition guarantee:

```python
for include_ttl in (True, False):
    selected = feature_columns(include_ttl)
    assert "attack_cat" not in selected and "label" not in selected
    assert "row_position" not in selected          # the Decision 4 index column cannot leak
assert len(feature_columns(True)) == 39 and len(feature_columns(False)) == 37
assert "cleaned" not in globals()                  # the test partition is unreachable from here
print("Label boundary and test-partition guards passed.")
```

**Cell `s0_11_assert_matrix`** — the fitted design matrix, both variants:

```python
for key, space in spaces.items():
    assert list(space.pipeline.feature_names_in_) == feature_columns(space.include_ttl)
    assert not ({"attack_cat", "label"} & set(space.names))
    assert space.design.shape[0] == len(train)
print("Fitted design matrices carry no target column, in either TTL variant.")
```

### The test partition is structurally unreachable, not merely unused

`s0_06_load` binds `train = cleaned.train` and then executes `del cleaned`. After that line, no later
cell can reach `CleaningResult.test` — the object is garbage and the name is gone. `s0_08` asserts it
(`"cleaned" not in globals()`). This is a stronger guarantee than the proposal's `rg` criterion, which
checks that `.test` is never *written*; this checks that it cannot be *reached*. Both hold: the
notebook never writes `cleaned.test`, `.test`, `test_df`, `X_test` or `load_raw_test` anywhere.

### The one caveat that must not be glossed

Because the subsample is stratified by `attack_cat` with a per-class floor, the rows silhouette and
DBSCAN are evaluated on were *selected* using label information. The clustering is still unsupervised
— no label enters any `fit` — but the sample is class-rebalanced, so minority attack families are
over-represented relative to the population and the noise fraction is conditioned on that. The
`floor=0` sensitivity run (Decision 8) is the label-neutral comparison, and it reuses the same tested
`nids` helper rather than introducing a second, untested sampler. Evidence lives in
`clustering_sample_allocation`, which carries the `proportional` column and the `floor_applied` flag
for both samplers side by side.

---

## F6 measurement: the one-hot variance asymmetry

`build_preprocessor`'s categorical branch is `RareCategoryGrouper → OneHotEncoder`, with **no
scaler**, while the numeric and skewed branches both end in `StandardScaler`. So in the ~52-column
design matrix, numeric columns carry variance 1.0 while a one-hot column carries `p(1−p) ≤ 0.25`, and
far less for a rare category. PCA maximises variance, so components are numeric-dominated **by
construction**, and PC1/PC2 loadings will be numeric-heavy for a structural reason rather than a
substantive one.

Scaling the one-hot block would require changing `build_preprocessor`, which is out of scope and
concurrently owned. The notebook therefore **measures** the asymmetry. A measured number a reader can
check beats a caveat a reader must trust.

### Exactly how it is computed

```python
rows = []
for key, space in spaces.items():
    frame = pd.DataFrame(space.design, columns=space.names)
    column_variance = frame.var(axis=0, ddof=1)          # ddof=1 matches PCA.explained_variance_
    total = float(column_variance.sum())

    # Auditability check: PCA is an orthogonal rotation, so it preserves the trace.
    assert np.isclose(total, float(pca_by_variant[key].explained_variance_.sum()), rtol=1e-9)

    pc1 = components_by_variant[key][0] ** 2             # unit-norm rows: each squares to 1.0
    pc2 = components_by_variant[key][1] ** 2
    name_index = {n: i for i, n in enumerate(space.names)}

    for block in ("numeric", "skewed", "onehot"):
        members = space.blocks[block]
        idx = [name_index[n] for n in members]
        rows.append({
            "variant": key,
            "block": block,
            "n_columns": len(members),
            "share_of_total_variance": float(column_variance.iloc[idx].sum() / total),
            "share_of_pc1_loading_sq": float(pc1[idx].sum()),
            "share_of_pc2_loading_sq": float(pc2[idx].sum()),
        })
```

### Why this is the right measurement, and why it is self-checking

| Property | Why it holds |
|---|---|
| The three shares are a **complete decomposition** | PCA centres but does not rescale, so `sum(explained_variance_) = trace(cov) = Σ_j Var(x_j)`. The block shares therefore sum to exactly 1.0 per variant. |
| It is **auditable inside the notebook** | The `np.isclose` assertion against `pca.explained_variance_.sum()` proves trace preservation at run time. Without it, "share of total variance" would be an assertion the reader has to accept. |
| It separates **cause** from **effect** | `share_of_total_variance` is the structural cause (what PCA is maximising). `share_of_pc1_loading_sq` / `share_of_pc2_loading_sq` are the visible effect (what the leading components are actually made of). Components are unit-norm, so each of those columns also sums to exactly 1.0 per variant — a second self-check a reader can run on the CSV. |
| It changes nothing | No `build_preprocessor` edit, no re-scaling, no new dependency. Six rows in one table. |

### What is registered

- Table `pca_variance_by_feature_block`: `variant, block, n_columns, share_of_total_variance,
  share_of_pc1_loading_sq, share_of_pc2_loading_sq` — 6 rows (2 variants × 3 blocks),
  `sort_by=["variant", "block"]`.
- Metrics `onehot_share_of_total_variance_with_ttl` and `..._without_ttl`.
- Note `onehot_variance_asymmetry`, which states the measured share, names the cause
  (categorical branch has no scaler), and says plainly what it means for loading interpretation:
  **a numeric-heavy PC1 is expected and is not evidence that the categorical features are
  uninformative.**
- Figure `pca_top_loadings_pc1_pc2_bars` colours its bars by block using the same `space.blocks`
  mapping, so the asymmetry is visible in the picture and quantified in the table.

**Expected magnitude, so an unexpected value is recognisable.** The one-hot block is three 1-of-K
groups whose per-group variance sums to `1 − Σ p_c² < 1`, against 36 standardised columns at variance
1.0 each. The one-hot share should therefore land **below ~8%**, most likely around 4–6%. A value far
outside that range is a finding to investigate, not a number to publish quietly.

---

## Figure specifications

All 16 figures. English titles and English axis labels are mandatory (AGENTS.md). Every figure is
written through `ResultsWriter.add_figure` at dpi 150 with matplotlib metadata suppressed; every
figure cell **ends with `plt.close(fig)`**, which returns `None` and therefore emits no duplicate
inline repr.

### Encoding, colour and scales

| # | `name` | Chart | x encoding | y encoding | Colour | Scales |
|---|---|---|---|---|---|---|
| 1 | `attack_cat_distribution_bars` | Horizontal bars, sorted by rows descending | Row count | Attack category | `CLASS_COLORS` per bar | x: log |
| 2 | `key_feature_histograms_raw` | 2 × 3 histogram grid, `bins=50` | Feature value | Row count | Single neutral steel-blue | x: linear, y: log |
| 3 | `key_feature_histograms_log1p` | 2 × 3 histogram grid, `bins=50` | `log1p(value)` | Row count | Single neutral steel-blue | both linear |
| 4 | `key_feature_ecdf_raw_vs_log1p` | 1 × 2 panels, 6 step curves each | left: raw value; right: `log1p(value)` | Cumulative proportion (0–1) | 6-colour `tab10` slice, one per key feature, shared legend | left x: symlog (`linthresh=1`); right x: linear |
| 5 | `key_feature_boxplots_by_attack_cat` | 3 × 2 horizontal boxplot grid, `showfliers=False`, `whis=(1, 99)` | Feature value | Attack category (`CLASS_ORDER`) | `CLASS_COLORS` per box | x: symlog (`linthresh=1`) |
| 6 | `pearson_correlation_heatmap` | 36 × 36 heatmap, no annotations | Feature | Feature | `RdBu_r`, `vmin=-1`, `vmax=1`, `center=0` | linear |
| 7 | `spearman_correlation_heatmap` | 36 × 36 heatmap, no annotations | Feature | Feature | `RdBu_r`, `vmin=-1`, `vmax=1`, `center=0` | linear |
| 8 | `variance_inflation_factors_bars` | Grouped vertical bars, 2 bars per feature | Feature | VIF | Two-colour variant pair; infinite bars hatched (`///`) with their own legend entry | y: log |
| 9 | `pca_scree_plot` | 2 overlaid line+marker series | Principal component number | Explained variance ratio | Two-colour variant pair | linear |
| 10 | `pca_cumulative_variance` | 2 overlaid step series + 2 horizontal reference lines | Number of components | Cumulative explained variance ratio | Two-colour variant pair; grey dashed reference lines at 0.90 / 0.95 | linear |
| 11 | `pca_top_loadings_pc1_pc2_bars` | 2 × 2 panels (variant × component), horizontal bars, top 15 by \|loading\| | Loading (signed, zero line drawn) | Feature | 3-colour block palette (`numeric`, `skewed`, `onehot`) | linear |
| 12 | `pca_scatter_pc1_pc2_by_attack_cat` | 1 × 2 panels (one per variant), scatter, `s=5`, `alpha=0.4`, `linewidths=0` | PC1 score | PC2 score | `CLASS_COLORS`; drawn largest class first, ties by name; legend outside with `markerscale=4` | linear |
| 13 | `kmeans_elbow_inertia` | 2 overlaid line+marker series | Number of clusters `k` | Inertia | Two-colour variant pair | linear |
| 14 | `kmeans_silhouette_by_k` | 2 overlaid line+marker series, selected `k` ringed | Number of clusters `k` | Mean silhouette coefficient | Two-colour variant pair | linear |
| 15 | `kmeans_calinski_harabasz_by_k` | 2 overlaid line+marker series | Number of clusters `k` | Calinski-Harabasz score | Two-colour variant pair | linear |
| 16 | `dbscan_k_distance_plot` | 2 sorted curves + 2 horizontal `eps` reference lines | Points sorted by k-distance | Distance to the k-th nearest neighbour | Two-colour variant pair; `eps` lines dashed in the matching colour | linear |

**Variant pair colour.** One fixed two-colour mapping reused by every overlaid figure:
`VARIANT_COLORS = {"with_ttl": "#1f77b4", "without_ttl": "#d62728"}` (tab10 blue and red — CVD
tolerant, and consistently "blue = with the shortcut, red = without" across figures 8, 9, 10, 13, 14,
15, 16). Line style is also varied (`-` vs `--`) so the two series stay distinguishable in a greyscale
print.

**Infinite VIF bars.** An infinite VIF cannot be plotted on a log axis. The rule: replace `inf` with
`10 × max(finite VIF)`, hatch those bars with `///`, and add a legend entry reading
"VIF is infinite (perfect collinearity)". The clipped height is a drawing convention, never a number
the table reports — `variance_inflation_factors` still carries `inf`.

### English titles and axis labels

| # | `name` | Title | x label | y label |
|---|---|---|---|---|
| 1 | `attack_cat_distribution_bars` | Attack category distribution in the cleaned training partition | Rows (log scale) | Attack category |
| 2 | `key_feature_histograms_raw` | Key feature distributions in raw units | *(per panel)* Value | Rows (log scale) |
| 3 | `key_feature_histograms_log1p` | Key feature distributions after log1p | *(per panel)* log1p(value) | Rows |
| 4 | `key_feature_ecdf_raw_vs_log1p` | Empirical cumulative distributions: raw units versus log1p | left: Value (symlog scale, linear below 1); right: log1p(value) | Cumulative proportion of rows |
| 5 | `key_feature_boxplots_by_attack_cat` | Key feature spread by attack category | *(per panel)* Value (symlog scale, linear below 1) | Attack category |
| 6 | `pearson_correlation_heatmap` | Pearson correlation between modelled numeric features | Feature | Feature |
| 7 | `spearman_correlation_heatmap` | Spearman rank correlation between modelled numeric features | Feature | Feature |
| 8 | `variance_inflation_factors_bars` | Variance inflation factor per numeric feature, with and without the TTL shortcut pair | Feature | Variance inflation factor (log scale) |
| 9 | `pca_scree_plot` | PCA explained variance per component | Principal component | Explained variance ratio |
| 10 | `pca_cumulative_variance` | PCA cumulative explained variance | Number of components | Cumulative explained variance ratio |
| 11 | `pca_top_loadings_pc1_pc2_bars` | Largest PC1 and PC2 loadings by feature block | Loading | Feature |
| 12 | `pca_scatter_pc1_pc2_by_attack_cat` | Training rows projected onto PC1 and PC2, coloured by attack category | PC1 (`{share:.1f}`% of variance) | PC2 (`{share:.1f}`% of variance) |
| 13 | `kmeans_elbow_inertia` | K-Means inertia by number of clusters | Number of clusters (k) | Inertia (within-cluster sum of squares) |
| 14 | `kmeans_silhouette_by_k` | K-Means silhouette by number of clusters (stratified subsample) | Number of clusters (k) | Mean silhouette coefficient |
| 15 | `kmeans_calinski_harabasz_by_k` | K-Means Calinski-Harabasz score by number of clusters | Number of clusters (k) | Calinski-Harabasz score |
| 16 | `dbscan_k_distance_plot` | Sorted k-nearest-neighbour distance and the selected eps | Points sorted by k-distance | Distance to the k-th nearest neighbour |

Colorbar labels for figures 6 and 7 are "Pearson correlation coefficient" and "Spearman rank
correlation coefficient". Panel titles on multi-panel figures name the feature or the variant
(`"With TTL shortcut"` / `"Without TTL shortcut"`, `"PC1"` / `"PC2"`). The variance shares
interpolated into figure 12's axis labels are deterministic values from
`pca.explained_variance_ratio_`, so the labels are stable across runs.

**Figure geometry** (fixed so PNG byte size is predictable): single-panel figures `10 × 6` in;
heatmaps `11 × 9.5` in with 7-point tick labels, x-ticks rotated 90°; the 2 × 3 grids `12 × 7` in;
the 3 × 2 boxplot grid `12 × 14` in; the 2 × 2 loading grid `12 × 10` in; the 1 × 2 scatter
`13 × 6` in.

---

## ResultsWriter registration plan

Every `name` below satisfies `ARTIFACT_NAME_PATTERN` = `^[a-z][a-z0-9_]{0,63}$`: lowercase-letter
first character, lowercase alphanumerics and underscores only, longest name 48 characters
(`kmeans_silhouette_floor0_sensitivity_without_ttl`). **No name begins with a digit.**

> **Name uniqueness is treated as global across tables and figures.** The foundation design states
> *"Registering the same `name` twice raises `ValueError`"* without saying whether the registry is
> per-kind. Assuming a single registry is the safe read, so the one collision in the proposal's
> inventory — `pca_top_loadings_pc1_pc2` appearing as both a table and a figure — is resolved by
> naming the figure `pca_top_loadings_pc1_pc2_bars`. No other pair collides.

### Common registration rules

1. **Every `add_table` call supplies `sort_by` with an ascending total key.** Where a descending order
   would read better (skew, VIF, cluster size), the reader sorts the CSV; the committed byte order is
   deterministic by an unambiguous ascending key. This removes any dependence on pandas tie ordering.
2. **Every frame is `reset_index()`ed before registration** — `ResultsWriter` writes with
   `index=False` unconditionally.
3. **Table cells end with a bounded preview** (`frame.head(10)` as the last expression) so the
   notebook shows its work with a fixed-shape HTML repr. The `add_table` call is a non-terminal
   statement, so its returned `Path` is never echoed.
4. **Figure cells end with `plt.close(fig)`.**
5. `variant` column values are exactly `with_ttl` and `without_ttl`.

### Tables (17)

| § | `name` | Columns | `sort_by` | `title` | `description` |
|---|---|---|---|---|---|
| 1 | `partition_shape_and_dtypes` | `position, column, dtype, non_null, role` | `["position"]` | Column inventory of the cleaned training partition | Every kept column in canonical order with its dtype, non-null count and role (`feature_numeric`, `feature_skewed`, `feature_categorical`, `target`). |
| 1 | `numeric_summary_mean_vs_median` | `feature, count, mean, median, std, min, q25, q75, max, skew, mean_median_ratio` | `["feature"]` | Central tendency and spread of the numeric features | Raw-unit summary contrasting mean against median to expose right skew. `mean_median_ratio` is null where the median is zero. |
| 1 | `categorical_value_counts` | `column, category, rows, share` | `["column", "category"]` | Categorical feature frequencies | Value counts and population shares for `proto`, `service` and `state` on the cleaned training partition. |
| 1 | `attack_cat_distribution` | `attack_cat, rows, share` | `["attack_cat"]` | Attack category balance of the cleaned training partition | Row count and share per class, describing the label without using it as an input. |
| 2 | `pearson_correlation_matrix` | `feature` + 36 feature columns | `["feature"]` | Pearson correlation between modelled numeric features | Linear correlation over the 36 numeric and skewed features in modelled form (`log1p` then standardised), with-TTL variant. |
| 2 | `spearman_correlation_matrix` | `feature` + 36 feature columns | `["feature"]` | Spearman rank correlation between modelled numeric features | Rank correlation over the same 36 features. Identical in raw units, because `log1p` is strictly increasing. |
| 2 | `high_correlation_pairs` | `feature_a, feature_b, pearson, spearman, method_flagged` | `["feature_a", "feature_b"]` | Feature pairs above the 0.9 correlation threshold | Upper-triangle pairs exceeding \|r\| > 0.9 under Pearson, Spearman or both. |
| 2 | `variance_inflation_factors` | `variant, feature, r_squared, vif, status` | `["variant", "feature"]` | Variance inflation factor per numeric feature, both TTL variants | VIF computed as `1 / (1 - R²)` from an ordinary least-squares fit of each feature on all the others, in modelled form, one-hot columns excluded. `status` is `ok`, `zero_variance` or `perfect_collinearity`. |
| 3 | `pca_explained_variance` | `variant, component, explained, cumulative` | `["variant", "component"]` | PCA explained variance per component, both TTL variants | Scree data for the full component set of each variant's design matrix. |
| 3 | `pca_components_for_variance` | `variant, threshold, n_components` | `["variant", "threshold"]` | Components required to reach 90% and 95% of variance | The dimensionality answer per variant, and the source of `min_samples` for DBSCAN. |
| 3 | `pca_top_loadings_pc1_pc2` | `variant, component, rank, feature, loading, abs_loading` | `["variant", "component", "rank"]` | Largest PC1 and PC2 loadings, both TTL variants | Top 15 loadings by absolute value per component, under a fixed sign convention so signs do not flip between runs. |
| 3 | `pca_variance_by_feature_block` | `variant, block, n_columns, share_of_total_variance, share_of_pc1_loading_sq, share_of_pc2_loading_sq` | `["variant", "block"]` | Variance contribution of each feature block | Share of total design-matrix variance and of PC1/PC2 squared loadings held by the numeric, skewed and one-hot blocks. Each share column sums to 1.0 per variant. |
| 4 | `kmeans_k_sweep_metrics` | `variant, k, inertia, silhouette, calinski_harabasz` | `["variant", "k"]` | K-Means sweep over k = 2 to 12, both TTL variants | Inertia and Calinski-Harabasz on the full cleaned training partition; silhouette on the stratified subsample, using labels from that same full-partition fit. |
| 4 | `dbscan_cluster_summary` | `variant, cluster, rows, share` | `["variant", "cluster"]` | DBSCAN cluster sizes on the stratified subsample | Cluster sizes including the `-1` noise label. A property of the subsample, not of the full partition. |
| 4 | `clustering_sample_allocation` | `sampler, attack_cat, available, allocated, proportional, floor_applied, population_share, sample_share` | `["sampler", "attack_cat"]` | Stratified subsample allocation for both samplers | Per-class allocation for the `floor=50` sample and the `floor=0` proportional sample, evidencing how far the default floor over-represents minority classes. |
| 5 | `cluster_profile_original_units` | `variant, cluster, feature, cluster_mean, global_mean, ratio, std_deviations` | `["variant", "cluster", "feature"]` | Per-cluster feature means in original units | Cluster means of the unscaled numeric and skewed features against the partition-wide mean, with the gap expressed in global standard deviations. Null where the global mean or standard deviation is zero. |
| 5 | `cluster_attack_cat_composition` | `variant, cluster, attack_cat, rows, share_of_cluster, share_of_class` | `["variant", "cluster", "attack_cat"]` | Attack category composition of each cluster | Post-hoc cross-tabulation of cluster labels against true classes, including zero cells. Profiling only: no metric is optimised against these labels. |

### Figures (16)

Titles and axis labels are in [Figure specifications](#figure-specifications). Manifest
`description` values:

| § | `name` | `title` | `description` |
|---|---|---|---|
| 1 | `attack_cat_distribution_bars` | Attack category distribution in the cleaned training partition | Sorted horizontal bars on a logarithmic row axis, because the largest class outnumbers the smallest by roughly two orders of magnitude. |
| 1 | `key_feature_histograms_raw` | Key feature distributions in raw units | Histogram grid for six volumetric and rate features in raw units, with a logarithmic count axis so the right tail is visible. |
| 1 | `key_feature_histograms_log1p` | Key feature distributions after log1p | The same six features after the `log1p` transform the preprocessing pipeline applies to skewed columns. |
| 1 | `key_feature_ecdf_raw_vs_log1p` | Empirical cumulative distributions: raw units versus log1p | Two panels showing the same six empirical cumulative distributions under a raw symlog axis and a `log1p` linear axis. The curves are identical by construction: `log1p` is strictly increasing, so it re-spaces the axis without changing the distribution. |
| 1 | `key_feature_boxplots_by_attack_cat` | Key feature spread by attack category | Horizontal boxplots per attack category on a symlog value axis. Whiskers are the 1st and 99th percentiles; individual outliers are omitted. |
| 2 | `pearson_correlation_heatmap` | Pearson correlation between modelled numeric features | Diverging palette centred exactly at zero with symmetric limits of -1 and 1. Values are in the matching CSV; the grid is not annotated. |
| 2 | `spearman_correlation_heatmap` | Spearman rank correlation between modelled numeric features | Same palette, centring and limits as the Pearson heatmap, so the two are directly comparable. |
| 2 | `variance_inflation_factors_bars` | Variance inflation factor per numeric feature, with and without the TTL shortcut pair | Grouped bars on a logarithmic axis. Infinite values are drawn at a clipped height with hatching; the table reports them as `inf`. |
| 3 | `pca_scree_plot` | PCA explained variance per component | Explained variance ratio per component for both TTL variants overlaid. |
| 3 | `pca_cumulative_variance` | PCA cumulative explained variance | Cumulative variance for both variants with reference lines at 90% and 95%. |
| 3 | `pca_top_loadings_pc1_pc2_bars` | Largest PC1 and PC2 loadings by feature block | Four panels (variant by component) of signed loading bars coloured by feature block, making the one-hot variance asymmetry visible. |
| 3 | `pca_scatter_pc1_pc2_by_attack_cat` | Training rows projected onto PC1 and PC2, coloured by attack category | One panel per TTL variant, drawn on the stratified subsample. Colour is applied after the projection was computed; no label entered the fit. |
| 4 | `kmeans_elbow_inertia` | K-Means inertia by number of clusters | Inertia on the full cleaned training partition for both variants. |
| 4 | `kmeans_silhouette_by_k` | K-Means silhouette by number of clusters (stratified subsample) | Mean silhouette on the stratified subsample for both variants, with the selected k marked. The value is a subsample estimate, not a population figure. |
| 4 | `kmeans_calinski_harabasz_by_k` | K-Means Calinski-Harabasz score by number of clusters | Calinski-Harabasz on the full cleaned training partition for both variants. |
| 4 | `dbscan_k_distance_plot` | Sorted k-nearest-neighbour distance and the selected eps | Sorted k-distance curve per variant with the selected `eps` drawn as a reference line, so a reader can disagree with the automatic selection from the picture. |

### Metrics (29)

Variant-independent (7):

| `name` | Type | `description` |
|---|---|---|
| `train_rows` | int | Rows in the cleaned training partition analysed by this notebook. |
| `attack_cat_classes` | int | Distinct attack categories present in the cleaned training partition. |
| `feature_columns_with_ttl` | int | Size of the feature allow-list including the TTL shortcut pair. |
| `feature_columns_without_ttl` | int | Size of the feature allow-list excluding `sttl` and `ct_state_ttl`. |
| `correlated_pairs_above_0_9` | int | Feature pairs whose absolute correlation exceeds 0.9 under Pearson or Spearman. |
| `silhouette_sample_rows` | int | Rows in the `floor=50` stratified subsample used for silhouette and DBSCAN. |
| `silhouette_sensitivity_sample_rows` | int | Rows in the `floor=0` proportional subsample used for the sensitivity silhouette. |

Per variant, registered once as `<stem>_with_ttl` and once as `<stem>_without_ttl` (11 × 2 = 22):

| `name` stem | Type | `description` (variant named in the text) |
|---|---|---|
| `max_vif` | float or `"inf"` | Largest variance inflation factor. Registered as the string `inf` when at least one feature is perfectly collinear, so the JSON stays strictly valid. |
| `pca_components_90` | int | Principal components required to reach 90% of the design-matrix variance. |
| `pca_components_95` | int | Principal components required to reach 95% of the design-matrix variance. |
| `onehot_share_of_total_variance` | float | Share of total design-matrix variance held by the unscaled one-hot block. |
| `kmeans_selected_k` | int | Cluster count selected as the silhouette maximum over k = 2 to 12, ties broken toward the smaller k. |
| `kmeans_silhouette_at_selected_k` | float | Mean silhouette at the selected k on the `floor=50` stratified subsample. |
| `kmeans_silhouette_floor0_sensitivity` | float | Mean silhouette at the selected k on the `floor=0` proportional subsample, the population-representative comparison. |
| `dbscan_eps` | float | Neighbourhood radius selected by the normalised chord rule on the sorted k-distance curve. |
| `dbscan_min_samples` | int | Minimum neighbourhood size, fixed at twice the 90%-variance component count. |
| `dbscan_cluster_count` | int | Clusters found on the stratified subsample, excluding the noise label. |
| `dbscan_noise_fraction` | float | Share of subsample rows assigned to the noise label. A subsample statistic, not a partition statistic. |

### Notes (6)

| `id` | Content the text MUST carry |
|---|---|
| `silhouette_is_subsample_estimate` | Silhouette is computed on a stratified subsample of at most 10,000 rows, stratified by `attack_cat` with a per-class floor of 50, seed 42. K-Means itself is fitted on the full cleaned training partition; only the metric is sampled. The value is an estimate, and the floor makes the sample class-rebalanced rather than population-representative. |
| `dbscan_is_subsample_property` | The DBSCAN noise fraction, cluster count and cluster sizes are properties of that same subsample. They are not claims about the full cleaned training partition. |
| `dbscan_eps_not_transferable` | Local density scales with row count. An `eps` calibrated on a 10,000-row sample is too large for the full partition and must not be lifted into another notebook. Also records whether the chord rule degenerated to a fallback. |
| `attack_cat_is_post_hoc_only` | `attack_cat` is used for stratification, colouring and post-hoc profiling only. It never enters a feature matrix and is never a fitting target. No external cluster-validation metric (ARI, NMI) is computed here; those are reserved for notebook 2. |
| `onehot_variance_asymmetry` | The categorical branch of `build_preprocessor` carries no scaler, so one-hot columns hold at most 0.25 variance against 1.0 for standardised numeric columns. The measured one-hot share of total variance is stated. PCA is numeric-dominated by construction; a numeric-heavy PC1 is expected and is not evidence that the categorical features are uninformative. |
| `ttl_shortcut_effect` | States, with the measured numbers, whether removing `sttl` and `ct_state_ttl` changed the maximum VIF, the 90% component count, the selected k and the cluster composition — and therefore whether any conclusion in this notebook depends on the known testbed shortcut. |

### Sidecar

`counts.json`, written through `write_json(payload, name="counts")` from the same `METRICS` and
`NOTES` dicts that fed `add_metric` / `add_note` (Decision 11). Shape:
`{"metrics": {name: value, ...}, "notes": {id: text, ...}}`. Unregistered, mirroring the convention
`cleaning_report.py` established so both producer folders read the same way.

### Inventory totals

| Kind | Count | Matches proposal? |
|---|---|---|
| Tables | 17 | Yes |
| Figures | 16 | Yes (one renamed: `pca_top_loadings_pc1_pc2_bars`) |
| Metrics | 29 | Grew from "~20" for variant symmetry — Decision 12 |
| Notes | 6 | Yes |
| Sidecars | 1 (`counts.json`) | Yes |

---

## Where the `sttl` / `ct_state_ttl` with-and-without reporting applies

Required call-out per `openspec/config.yaml` `rules.design`. AGENTS.md: *"Report supervised results
with and without `sttl` and `ct_state_ttl` (known testbed shortcut)."* This notebook produces no
supervised result, but the same pair drives every unsupervised result here, so the rule is applied in
full.

| Location | Does the toggle apply? | What it does |
|---|---|---|
| `columns.TTL_SHORTCUT_COLUMNS` | — | The single definition of the pair. The notebook never writes `sttl` or `ct_state_ttl` as a literal. |
| `feature_columns(include_ttl)` | **Yes** | Filters the 39-name allow-list down to 37. A filter of an allow-list, not a set difference. |
| `build_preprocessor_pair()` | **Yes** | Both TTL columns are plain numeric, so exclusion removes exactly two columns from the `numeric` branch. No fitted value changes shape or meaning. |
| `VariantSpace` (`design`, `names`, `blocks`, `modelled`) | **Yes** | Built twice. Everything downstream reads from it, so no analysis cell can accidentally report one variant. |
| `variance_inflation_factors` → table + figure + `max_vif_{v}` | **Yes** | VIF is **multivariate** — each feature is regressed on all the others — so removing two columns changes *every other feature's* value. This is the clearest case for dual reporting. |
| `pearson_correlation_matrix`, `spearman_correlation_matrix`, their heatmaps | **No, deliberately** | Pearson and Spearman are **pairwise**. The without-TTL matrix is the with-TTL matrix with two rows and two columns deleted; every remaining cell is numerically identical. A second heatmap would carry zero new information. Section 2 states this asymmetry in prose: *run it where it can change, say why it cannot change where it cannot.* |
| `pca_explained_variance`, `pca_components_for_variance`, `pca_top_loadings_pc1_pc2`, `pca_variance_by_feature_block` and all four PCA figures | **Yes** | Two fewer columns changes every component, every loading and the 90%/95% component counts. |
| `kmeans_k_sweep_metrics`, the three sweep figures, `kmeans_selected_k_{v}`, both silhouette metrics | **Yes** | K-Means runs in the PCA space, which differs between variants, so every cluster assignment can differ. |
| `dbscan_cluster_summary`, `dbscan_k_distance_plot`, `dbscan_eps_{v}`, `dbscan_min_samples_{v}`, `dbscan_cluster_count_{v}`, `dbscan_noise_fraction_{v}` | **Yes** | `min_samples = 2 × n_components_90`, which is itself variant-dependent, so both the parameter and the result must be reported twice (Decision 12). |
| `cluster_profile_original_units`, `cluster_attack_cat_composition` | **Yes** (proposal **Q4**) | *"Does removing the TTL shortcut change what the clusters represent"* is precisely the question the with/without rule exists to answer. Cost: two extra values in a `variant` column in two CSVs. |
| Section 1 descriptive tables and figures | **Not applicable** | Descriptive statistics describe all 41 columns of the cleaned partition; the TTL pair appears as two of them. There is no model and therefore no shortcut to exclude. |
| `clustering_sample_allocation` | **Not applicable** | Sampling is stratified by `attack_cat`, independent of the feature allow-list. Both samplers draw the same rows for both variants. |
| Section 6 conclusions and the `ttl_shortcut_effect` note | **Yes, in prose** | Section 6 MUST state explicitly whether any conclusion changes when the pair is removed. Success criterion: both `with_ttl` and `without_ttl` appear in the `variant` column of `variance_inflation_factors`, `pca_explained_variance`, `pca_components_for_variance`, `kmeans_k_sweep_metrics`, `cluster_profile_original_units` and `cluster_attack_cat_composition`. |

---

## Determinism and diff churn

### What is pinned, and by what

| Source of drift | How it is pinned |
|---|---|
| Row order of the partition | `clean_partitions` ends with `reset_index(drop=True)` on both frames, so positional order equals `0…n-1`. Everything positional in this notebook rests on that. |
| Subsample selection | `stratified_subsample(..., seed=42)` on `index_frame`, whose row order is that same positional order. Largest-remainder allocation with a stated tie-break, one `default_rng(42)` consumed in sorted class order. |
| PCA subspace | `PCA(svd_solver="full")` — `"auto"` may select the randomized solver on larger inputs; `"full"` is exactly deterministic. |
| PCA component signs | The notebook's own `fix_component_signs` rule (Decision 6), independent of LAPACK and of scikit-learn's private `svd_flip`. Scores are indexed, never re-transformed. |
| K-Means | `KMeans(n_clusters=k, n_init=10, random_state=42)`, default `k-means++` init and `lloyd` algorithm. |
| Selected `k` | `argmax` silhouette with an explicit tie-break toward the smaller `k`. No eyeballing. |
| Section 5 labels | Reuse of the sweep's fitted estimator; no refit. |
| DBSCAN `eps` | The normalised chord rule (Decision 9). `np.argmax` returns the first maximum, so ties are broken deterministically. Three named fallbacks, each recorded in a note. |
| DBSCAN itself | Deterministic given the input, `eps` and `min_samples`; border-point assignment depends on row order, which is fixed. |
| VIF | Ordinary least squares on a fixed column set in `FEATURE_COLUMNS` order. |
| Table byte order | Every `add_table` supplies an ascending total `sort_by`; the writer applies it with `kind="stable"`. |
| CSV / JSON / PNG bytes | `ResultsWriter` contract: `lineterminator="\n"`, `float_format="%.6f"`, JSON at fixed indent with a trailing newline, PNG with `metadata={"Software": None, "Creation Time": None}`. Without that last override matplotlib stamps its version and the wall-clock time into PNG text chunks and every rerun is a binary diff. |
| Inline figure size | `plt.rcParams["figure.dpi"] = 100` and explicit `figsize` per figure. |
| Text metrics | `plt.rcParams["font.family"] = "DejaVu Sans"`, which ships with matplotlib, so a missing system font cannot silently change glyph widths. |
| DataFrame HTML reprs | `pd.set_option` display limits set once in `s0_02_style`; every table cell previews `frame.head(10)`. |
| Stray object reprs | Every figure cell ends with `plt.close(fig)` (returns `None`). No bare `fig` as a last expression, and no `add_table` / `add_figure` return value left as a terminal expression. |
| Execution counts | One `nbconvert --execute --inplace` pass always renumbers code cells `1…N` in order. Interactive partial re-runs are never committed. |
| Cell ids | nbformat v4.5 `id` fields are hand-assigned and preserved on re-execution, never regenerated (Decision 1). |
| Wall-clock values | `generated_at` lives only in `manifest.json`. No wall time is ever registered as a metric. |

### What is honestly not fixed

| Not fixed | Why, and what is done about it |
|---|---|
| **Inline PNG base64 blobs** | Produced by the IPython inline backend, which does **not** apply `ResultsWriter`'s metadata suppression. Those blobs can differ across matplotlib and freetype versions even when the plotted data is identical, and **any re-execution rewrites every blob regardless**. The mitigation is procedural, not technical: execute once, deliberately, as the last step before commit. |
| `manifest.json`'s `generated_at` | Changes on every run by definition. Confined to one line of one file; `SOURCE_DATE_EPOCH` removes even that. |
| `metadata.language_info` | The first execution settles it with the concrete interpreter version and CodeMirror mode. One-time, then stable. |
| Notebook and results weight | Estimated at ~10 MB total (proposal F4). `sdd-apply` reports the measured `du -sh` of both the `.ipynb` and `results/eda_reduction_clustering/`; if either exceeds twice the estimate that is a finding, not a rounding error. |
| Repository-wide notebook diff ergonomics | A `.gitattributes` notebook diff driver would help, but it is a repository-wide policy change riding on a notebook change. Recorded as a follow-up (proposal F4), not smuggled in here. |

### The determinism check

Re-run the notebook and compare:

- `results/eda_reduction_clustering/tables/*.csv` — **byte-identical**.
- `results/eda_reduction_clustering/figures/*.png` — **byte-identical**.
- `results/eda_reduction_clustering/counts.json` — **byte-identical**.
- `results/eda_reduction_clustering/manifest.json` — differs in exactly one line, `generated_at`.
- `notebooks/01_eda_reduction_clustering.ipynb` — **expected to differ** in its base64 output blobs.
  That is the documented limit above, not a failure.

---

## Build and execution procedure

```bash
# 1. Structural gate — catches malformed JSON before a kernel is ever started.
uv run python -c "import sys, nbformat; nbformat.validate(nbformat.read(sys.argv[1], as_version=4)); print('nbformat OK')" \
  notebooks/01_eda_reduction_clustering.ipynb

# 2. One execution pass. Renumbers code cells 1..N and preserves cell ids.
uv run jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 \
  notebooks/01_eda_reduction_clustering.ipynb
```

**`--ExecutePreprocessor.timeout=-1` is load-bearing, not decoration.** The K-Means sweep cell runs
for minutes (Decision 8). Several nbconvert releases default `ExecutePreprocessor.timeout` to 30
seconds, which would abort that cell and take the whole run down with a `TimeoutError` that looks like
a code bug. `-1` disables the per-cell timeout explicitly, so the behaviour does not depend on which
nbconvert release is installed.

**`--allow-errors` is never passed.** Stopping at the first erroring cell is what makes the
"no manifest ⇒ incomplete folder" guarantee hold (Decision 3).

`sdd-apply` runs step 2 twice, with section 6's narrative markdown authored between the two runs
(Decision 13). The second run's output is the committed state.

---

## File Changes

| File | Action | Description |
|---|---|---|
| `notebooks/01_eda_reduction_clustering.ipynb` | Create | The deliverable. ~73 cells of nbformat v4.5 JSON, committed executed with outputs. |
| `results/eda_reduction_clustering/manifest.json` | Create (generated, committed) | 17 tables, 16 figures, 29 metrics, 6 notes. Written last, by `close()`. |
| `results/eda_reduction_clustering/counts.json` | Create (generated, committed) | Human-readable mirror of the metric and note registry. |
| `results/eda_reduction_clustering/tables/*.csv` | Create (generated, committed) | 17 files. |
| `results/eda_reduction_clustering/figures/*.png` | Create (generated, committed) | 16 files at dpi 150. |
| `openspec/changes/notebook-01-eda-reduction-clustering/design.md` | Create | This document. |
| `src/nids/**` | **Untouched** | Frozen API, concurrently owned by `project-foundation`. Read-only reference. |
| `pyproject.toml` | **Untouched** | No dependency added (proposal F1/Q3). `statsmodels` and `jupytext` are deliberately absent. |
| `tests/**` | **Untouched** | `openspec/config.yaml` `rules.tasks`: tests are required for `src/` and skill scripts; notebooks are exploratory. |
| `data/raw/**` | **Untouched** | Read-only, reached only through `nids.validation` and `nids.data`. |
| `results/data_cleaning/**` | **Untouched** | The output contract forbids one producer writing into another's folder. |
| `openspec/changes/project-foundation/**` | **Untouched** | Concurrently owned. |
| `.gitattributes` | **Untouched** | A notebook diff driver is a repository-wide policy change; recorded as a follow-up. |

---

## Interfaces / Contracts

Notebook-local definitions, all in `s0_09_helpers`. These are the only non-trivial constructs
`sdd-apply` invents; everything else is a call into `nids` or arithmetic on an array.

```python
VARIANTS: dict[str, bool] = {"with_ttl": True, "without_ttl": False}
KEY_FEATURES: tuple[str, ...] = ("dur", "sbytes", "dbytes", "rate", "sload", "dload")
K_RANGE = range(2, 13)
VARIANCE_THRESHOLDS: tuple[float, ...] = (0.90, 0.95)
NOTEBOOK_ID: str = "eda_reduction_clustering"      # provisional, proposal Q1 option A
SEED: int = 42

@dataclass(frozen=True)
class VariantSpace:
    key: str
    include_ttl: bool
    columns: list[str]
    pipeline: Pipeline
    names: list[str]
    design: np.ndarray
    modelled: pd.DataFrame
    blocks: dict[str, list[str]]

def build_variant_space(key: str, include_ttl: bool) -> VariantSpace: ...
def variance_inflation_factors(frame: pd.DataFrame) -> pd.DataFrame: ...
def fix_component_signs(components: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
def select_eps(sample: np.ndarray, min_samples: int) -> tuple[float, np.ndarray, str]: ...
def record_metric(name: str, value: int | float | str, description: str) -> None: ...
def record_note(note_id: str, text: str) -> None: ...
def new_figure(nrows: int = 1, ncols: int = 1, *, figsize: tuple[float, float]) -> tuple[Figure, Any]: ...
```

`select_eps` returns `(eps, kdist, rule)` where `rule` is one of `chord`, `flat_curve_median` or
`smallest_positive`, so the fallback taken is a value the notebook can put into a note rather than a
branch a reader has to infer.

**Dependency surface — no new dependency.** The notebook imports only from the declared set:
`numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn` (`PCA`, `KMeans`, `DBSCAN`,
`NearestNeighbors`, `LinearRegression`, `silhouette_score`, `calinski_harabasz_score`), the standard
library (`dataclasses`), and `nids`. `scipy` is declared but not needed — the ECDF is
`np.sort` plus `np.arange`, and skew comes from `pandas.DataFrame.skew()`. `umap-learn` stays
declared and unused (proposal Q2). `statsmodels` is not added (proposal Q3 / F1). `nbformat` and
`nbconvert` are tooling, invoked from the shell, never imported by the notebook.

---

## Testing Strategy

`openspec/config.yaml` `rules.tasks`: *"Tests are required only for code in `src/` and for skill
scripts; notebooks are exploratory and do not follow RED-GREEN-REFACTOR."* `strict_tdd: false`. This
change writes **no pytest tests** and modifies none.

Verification is instead performed by the executed notebook itself plus mechanical checks a reviewer
can re-run.

| Layer | What is verified | How |
|---|---|---|
| Runtime, in-notebook | Label boundary, allow-list sizes, test-partition unreachability | Assertion cell `s0_08_assert_labels`; failure aborts the run before any artifact is written |
| Runtime, in-notebook | No target column in either fitted design matrix; both matrices cover every row | Assertion cell `s0_11_assert_matrix` |
| Runtime, in-notebook | The variance decomposition is a true decomposition | `np.isclose(column_variance.sum(), pca.explained_variance_.sum())` in `s3_09_blocks` |
| Runtime, in-notebook | Manifest completeness | `s6_05_close` re-reads `manifest.json` and asserts every declared `tables[].path` and `figures[].path` resolves inside the folder |
| Runtime, in-notebook | Inventory matches the design | `s6_05_close` asserts 17 tables, 16 figures, 29 metrics, 6 notes |
| Build | The document is a valid notebook | `nbformat.validate` gate before execution |
| Build | Clean execution | `nbconvert --execute` exits 0 and no cell carries an `error` output |
| Static | No test-partition access | `rg -n 'load_raw_test\|\.test\b\|test_df\|X_test' notebooks/01_eda_reduction_clustering.ipynb` returns nothing |
| Static | No re-implementation of `nids` | `rg -n 'read_csv\|duplicated(\|StandardScaler(\|OneHotEncoder(\|drop(columns=\|to_csv(\|savefig(' notebooks/01_eda_reduction_clustering.ipynb` returns nothing — see the VIF rephrasing note in [The `nids` API surface consumed](#the-nids-api-surface-consumed) |
| Static | Seed discipline | The only seed literal in the notebook is `42` |
| Static | English everywhere | Manual review of every markdown cell, figure title, axis label, legend entry and table column name |
| Reproducibility | Byte stability | Re-run and diff per [The determinism check](#the-determinism-check) |
| Reporting | Dual TTL coverage | Both `with_ttl` and `without_ttl` appear in the `variant` column of the six dual-variant tables, and section 6 states in prose whether the conclusions differ |

---

## Threat Matrix

**N/A — no routing, shell, subprocess, VCS/PR automation, executable-file classification, or
process-integration boundary.**

The notebook spawns no subprocess, parses no untrusted input, performs no routing and touches no
VCS automation. The one security-adjacent surface — writing files under `results/` from a
caller-supplied identifier — is owned by `project-foundation` and already closed there:
`results_dir()` validates `notebook_id` against `NOTEBOOK_ID_PATTERN` **before** any filesystem call
and re-checks `resolved.is_relative_to(results_root().resolve())` afterwards; `ResultsWriter`
validates every artifact name against the same pattern and appends the extension itself, so a caller
cannot supply a separator, an extension or a `..`. This notebook passes one hard-coded constant
(`NOTEBOOK_ID = "eda_reduction_clustering"`) and 33 hard-coded artifact names. No task is manufactured
for this row.

`nbconvert` is invoked from the shell by a human or by `sdd-apply` as a build command, on a path
authored in this same change. It is not process integration and takes no external input.

---

## Migration / Rollout

**No migration required.** The change adds files and modifies none. Rollback per the proposal:

1. **Full revert** — `git revert <sha>` removes the notebook and the whole
   `results/eda_reduction_clustering/` folder. No source module, dependency, test or other producer's
   folder is affected.
2. **Outputs bad, notebook fine** — delete `results/eda_reduction_clustering/` and re-run the single
   `nbconvert` command. `prune_undeclared=True` removes orphans from a previous schema on `close()`,
   so a partial rewrite cannot leave stale files the manifest does not declare.
3. **Run aborted mid-way** — every artifact is written `<path>.tmp` then `os.replace`d, and the
   manifest is written last. A crashed run leaves no manifest, which is the unambiguous signal that
   the folder is incomplete.
4. **Blast radius if the analysis is simply wrong** — zero outside its own two paths. No downstream
   artifact consumes this folder yet.

---

## Refinements to the proposal

Every place this design goes beyond `proposal.md`'s literal wording, with its reason. None contradicts
the proposal's intent.

| # | Refinement | Reason |
|---|---|---|
| R1 | `ResultsWriter` is opened in setup and closed in the final cell, not held in a `with` block | A `with` block cannot span cells. The guarantee is identical because the manifest is written only by `close()`/`__exit__` and `nbconvert` stops at the first error (Decision 3). |
| R2 | `stratified_subsample` is called on a two-column `index_frame`, not on the partition | `SubsampleResult` drops the original index and exposes no positions accessor, so there is otherwise no way to evaluate a full-partition fit on the sampled rows (Decision 4). Recorded as a finding for `project-foundation`. |
| R3 | `variance_inflation_factors` gains a `status` column | The proposal's F1 requires "a reason" for `NaN` rows; `status` is that reason, with three fixed values. |
| R4 | The VIF loop uses positional column selection instead of `drop(columns=[...])` | Keeps the proposal's `rg` success criterion literal, with no footnote (see the exception note in the API section). |
| R5 | `max_vif_{v}` is registered as the string `"inf"` when any VIF is infinite | `json.dumps(float("inf"))` emits the invalid token `Infinity`, which would make `manifest.json` unparseable by a strict reader. |
| R6 | An explicit notebook-level PCA sign convention, on top of `svd_solver="full"` | The proposal relies on scikit-learn's private `svd_flip`, whose `u_based_decision` default has changed across releases. The explicit rule is version-independent (Decision 6). |
| R7 | Correlation matrices are computed on the modelled form | Consistency with VIF, which F1 explicitly places in modelled form. A reader comparing correlation against VIF must be looking at one space (Decision 7). |
| R8 | `key_feature_ecdf_raw_vs_log1p` is two panels of the same curves, not two different curves | The ECDF is invariant under a monotone transform, so a literal "raw versus log1p" overlay would draw one curve twice. Two panels show the same curves under two axis parameterisations, which is the comparison the figure was meant to make — and the markdown states the invariance as a finding. |
| R9 | Figure `pca_top_loadings_pc1_pc2` renamed `pca_top_loadings_pc1_pc2_bars` | The table of the same name would collide if `ResultsWriter`'s name registry is global. The foundation design does not say it is per-kind, so the safe read is assumed. |
| R10 | Metric inventory grew from "~20" to 29 | The proposal's illustrative list registers the DBSCAN and silhouette families for the with-TTL variant only, and gives `dbscan_min_samples` no suffix although its value is variant-dependent. Symmetry restores the dual-reporting rule (Decision 12). |
| R11 | `pca_variance_by_feature_block` gains `n_columns`, `share_of_pc1_loading_sq`, `share_of_pc2_loading_sq` | Separates the structural cause (trace share) from the visible effect (leading-component composition), and gives the table two arithmetic self-checks a reader can run on the CSV (see [F6 measurement](#f6-measurement-the-one-hot-variance-asymmetry)). |
| R12 | Section 6 is finalised by a second `nbconvert` pass | Authored conclusions must quote observed numbers. Markdown carries no `execution_count`, so the committed file still shows a clean `1…N` (Decision 13). |
| R13 | `KEY_FEATURES` is a frozen six-name constant | The proposal says "key volumetric features" without naming them. Freezing the list keeps the figure inventory from becoming data-derived. |
| R14 | `--ExecutePreprocessor.timeout=-1` is mandatory in the execute command | Several nbconvert releases default to a 30-second per-cell timeout, which the K-Means sweep would exceed. |
| R15 | `del cleaned` after binding `train` | Upgrades "the test partition is never used" from a grep result to a runtime guarantee inside the executed notebook. |

---

## Open Questions

- [ ] **Q1 (BLOCKING for apply, provisional here)** — `results_dir("01_eda_reduction_clustering")`
      raises `ValueError`, verified against committed `src/nids/paths.py:19`. This design assumes
      **option A**: notebook file unchanged, `NOTEBOOK_ID = "eda_reduction_clustering"`,
      results at `results/eda_reduction_clustering/`. **Awaiting the user's ruling.** If the answer
      changes, exactly one notebook constant and one folder path change; nothing else in this
      document depends on it.
- [ ] **Blocked on `project-foundation` slices 3–5.** `load_clean_partitions`, `build_preprocessor`,
      `build_preprocessor_pair`, `stratified_subsample` and `ResultsWriter` are not committed yet.
      Planning is complete against the frozen foundation design; `sdd-apply` MUST re-verify all
      consumed signatures against committed source before writing its first cell.
- [ ] **Finding to route to `project-foundation` (not a blocker).** `SubsampleResult` exposes no
      `selected_positions`. Decision 4 documents the notebook's workaround; a one-field addition
      would remove it. Out of scope here — `src/nids/` is concurrently owned.
- [ ] **Assumption to confirm at apply time.** `ResultsWriter`'s artifact-name registry is assumed
      **global** across tables and figures. If it turns out to be per-kind, R9's rename is harmless
      but unnecessary. No action needed either way.
- [ ] **Proposal Q2, Q3, Q4 are treated as settled** by the proposal (t-SNE/UMAP stay out;
      `statsmodels` is not added; section 5 profiles both variants). This design implements those
      answers. Overrule them at the orchestrator level if the user disagrees; each is a local change
      here.





