# Final Report — UNSW-NB15 Machine Learning Project

Unattended build, 2026-09-18 23:40 to 2026-09-19 04:15 (UTC-5). Every code commit
passed the project's external reviewer (GGA) before landing. `--no-verify` was
never used. `data/raw/` is byte-identical to its starting state.

---

## 1. What was built

| Component | Location | What it is |
|---|---|---|
| `nids` package | `src/nids/` | Installable shared library: paths, column allow-lists, loading, validation, cleaning, fitted transformers, pipeline factory, stratified subsampling, results contract, cleaning report, tabular helpers |
| Notebooks | `notebooks/` | Three executed notebooks with committed outputs |
| Skills | `skills/` | Three parametrized, dataset-agnostic Claude Code skills |
| Dashboard | `dashboard/` | Read-only Streamlit app discovering `results/*/manifest.json` |
| Results | `results/` | Four committed output folders, each with a machine-readable manifest |
| Tests | `tests/` | 242 passing, 5 deselected (slow real-data tests, all passing when run) |
| SDD artifacts | `openspec/changes/` | Six changes, each with proposal, spec, design and tasks |

---

## 2. Commits per milestone

| Milestone | Commit | Subject |
|---|---|---|
| Planning | `3481bb9` | `docs(openspec): add project-foundation proposal, specs, design and tasks` |
| 1 — Foundation slice 1 | `d71e306` | `feat(packaging): add build system and package skeleton` |
| 1 — Foundation slice 2 | `675c5c6` | `feat(data): add raw partition loading and validation` |
| 1 — Foundation slice 3 | `cb1c201` | `feat(cleaning): add ordered dataset cleaning and diagnostics` |
| (reviewer switch) | `fe9db4f` | `chore(gga): switch reviewer to claude` (made by the user) |
| 1 — Foundation slice 4 | `7cac494` | `feat(preprocessing): add fitted pipeline factory and subsampling` |
| 1 — Foundation slice 5 | `b0ae92f` | `feat(results): add output contract and cleaning report` |
| 6 — Dashboard deps | `15e29c2` | `chore(deps): add streamlit and enable repo-root imports for pytest` |
| 6 — Dashboard | `c3ffae6` | `feat(dashboard): add read-only streamlit results dashboard` |
| 4 — Notebook 3 | `4c9a8cb` | `feat(notebooks): add classification notebook with executed outputs` |
| 3 — Notebook 2 | `f476990` | `feat(notebooks): add clustering-reduction notebook with executed outputs` |
| 2 — Notebook 1 | `525b934` | `feat(notebooks): add EDA reduction and clustering notebook with executed outputs` |
| 5 — Skills | `8e17774` | `feat(skills): add parametrized notebook skills with tests` |
| 7 — Documentation | `daa6a2a` | `docs(readme): document setup, data, notebooks, skills, dashboard and findings` |

---

## 3. Key findings

### 3.1 Cleaning (`results/data_cleaning/`)

| Metric | Value |
|---|---|
| Training rows retained | 107,740 (from 175,341) |
| Testing rows retained | 78,084 (from 82,332) |
| Duplicate training rows dropped | 67,601 |
| Leaking testing rows removed | 4,248 |
| Duplicate rows **kept** inside testing | 26,387 (32.05%) |
| Label-noise feature combinations | 1,772 |
| **Contradictory testing rows retained** | **4,294 (5.499%)** |
| Raw null cells | 0 |

The last row is the project's error floor: 4,294 testing rows are feature-identical
to a training row but carry a different label. No function of the features can
classify both correctly. Roughly 5.5% of the retained testing set is unwinnable
by construction.

A premise worth correcting: the reordering that drops `stcpb`/`dtcpb` before
deduplication was adopted because it is the principled order, **not** because it
reveals hidden duplicates. Measured directly: 67,601 duplicates with those columns
present, 67,601 without. 54.89% of rows carry `stcpb == dtcpb == 0`, so rows that
duplicate each other already agreed on them.

### 3.2 Notebook 1 — reduce then cluster (`results/eda_reduction_clustering/`)

| Metric | With TTL | Without TTL |
|---|---|---|
| PCA components for 90% variance | 12 | 12 |
| PCA components for 95% variance | 17 | 16 |
| K-Means selected k | 10 | 10 |
| Silhouette at selected k | 0.472 | 0.442 |
| DBSCAN noise fraction | 0.27% | 0.27% |
| One-hot share of total PCA variance | 4.05% | 4.27% |

`ackdat`, `synack` and `tcprtt` are perfectly collinear (VIF infinite) in both
variants. Sixteen feature pairs exceed |r| > 0.9, including `swin`↔`dwin`
(r = 0.987) and `trans_depth`↔`ct_flw_http_mthd` (Spearman 0.999).

Cluster structure: several near-pure `Normal` clusters, one attack-dominated
cluster (74.4% `Generic`, driven by the connection-count feature family — a
scanning-like signature), and several mixed `Fuzzers`/`Exploits` clusters with no
dominant class.

### 3.3 Notebook 2 — cluster then reduce (`results/clustering_reduction/`)

| Metric | With TTL | Without TTL |
|---|---|---|
| Best method | K-Means | Spectral |
| Best ARI vs `attack_cat` | 0.163 | **−0.005** |
| Best NMI vs `attack_cat` | 0.289 | **0.005** |
| Best ARI vs `label` | 0.183 | 0.003 |
| K-Means silhouette at k* | 0.405 | 0.376 |
| Bootstrap stability (mean ARI) | 0.811 | 0.998 |
| DBSCAN noise fraction | 1.16% | 1.02% |

The 0.998 stability without TTL is not a quality result: that run selected k=2,
and a two-way split is trivially reproducible across bootstraps. It is simpler,
not better.

Pipeline comparison: reduce-then-cluster reaches a cleaner silhouette than
cluster-then-reduce in both variants (0.472/0.442 against 0.405/0.376), which is
expected since PCA denoises before clustering. The comparison is directional, not
controlled — different populations and different spaces.

### 3.4 Notebook 3 — classification (`results/classification/`)

Evaluated exactly once on the cleaned testing partition (`test_evaluations_count = 1`).

| Model | macro F1 | balanced acc | accuracy | training rows |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.512** | 0.577 | 0.743 | 107,740 |
| RandomForest | 0.503 | **0.638** | 0.692 | 107,740 |
| KNN | 0.420 | 0.457 | 0.706 | 20,000 |
| SVC (RBF) | 0.377 | 0.580 | 0.619 | 5,000 |
| GaussianProcess | 0.373 | 0.462 | 0.676 | 2,000 |
| LogisticRegression | 0.356 | 0.577 | 0.615 | 107,740 |
| Dummy | 0.089 | 0.098 | 0.269 | 107,740 |

Binary attack-vs-normal: ROC-AUC 0.983, PR-AUC 0.987. Cost-optimal threshold at
FN:FP = 20:1 is 0.075, reducing expected cost 82% versus the default 0.5. The
threshold was selected on out-of-fold **training** probabilities, never on test.

Hardest classes are **Backdoor (F1 0.072)** and **Analysis (F1 0.107)** — not the
rarest. Worms reaches F1 0.591 on 44 testing rows. Rarity was solvable by the
subsample floor; semantic overlap with Exploits was not.

Total test error for the best model is about 25.7% against a 5.5% label-noise
floor, so roughly one fifth of the error is unbeatable and four fifths is genuine
model limitation.

### 3.5 The TTL shortcut result

This is the most useful finding, and it needs both halves to be read correctly.

- **Supervised**: dropping `sttl`/`ct_state_ttl` moves the best model's macro F1
  by **+0.0087**. Essentially nothing.
- **Unsupervised**: the same drop collapses cluster alignment with `attack_cat`
  from ARI 0.163 to **−0.005**, and changes which algorithm wins.

The clusters still exist without the shortcut — notebook 1 still selects k=10 with
a 0.442 silhouette. They simply stop corresponding to attack categories. The TTL
pair is not what makes this data classifiable; it is what makes the attack classes
look like natural clusters.

Reporting both variants is an AGENTS.md rule. Without it, this finding would have
been invisible.

---

## 4. Decisions made autonomously

Each is recorded in its change folder's `DECISIONS.md` or design document.

| # | Decision | Why |
|---|---|---|
| 1 | Results folder `results/eda_reduction_clustering/`, notebook keeps `01_` prefix | `NOTEBOOK_ID_PATTERN = ^[a-z][a-z0-9_]{0,63}$` rejects a leading digit; verified by an actual `ValueError`. The alternatives were editing a committed module owned by another change, or losing the ordering signal in `notebooks/`. |
| 2 | VIF as `1/(1-R²)` via scikit-learn; `statsmodels` not added | Identical quantity, no new dependency, and `LinearRegression` fits the intercept that `statsmodels` callers must add manually and routinely forget. |
| 3 | Non-finite metrics registered as the string `"inf"` | `json.dumps(float("inf"))` emits the bare token `Infinity`, which is invalid JSON and would make a committed manifest unparseable. |
| 4 | `--ExecutePreprocessor.timeout=-1` on every notebook execution | Several nbconvert releases default to a 30-second per-cell timeout. |
| 5 | kNN capped at 20,000 reference rows (a third cap beyond the two requested) | Brute-force kNN is O(n_train × n_test); 107,740 × 78,073 alone exceeds the whole compute budget. Disclosed in every results table. |
| 6 | Cost threshold chosen on out-of-fold training probabilities | Choosing it on test is selection on test under another name, and would void the single-evaluation guarantee. |
| 7 | GaussianProcess at 2,000 rows, SVC at 5,000, searches on a 30,000-row train subsample with `n_iter ≤ 15` and 3-fold CV | The only way to fit the ~15-minute budget. Every table carries its training row count so a 2,000-row GP is never silently compared against a 107,740-row forest. |
| 8 | `_build_generic_preprocessor` first made private to dodge an AST guard — **then reversed** | The reviewer was right that hiding from a guard is not compliance. The function is public and the guard was made precise: it now rejects DataFrame/Series/ndarray annotations rather than permitting only `bool`. |
| 9 | Unsupervised skills first had their train/test split removed — **then reversed** | I authorized the removal with a docstring exemption. The reviewer's rejection was correct: AGENTS.md states "Split first" unconditionally, the rule concerns the operation not the model, and `--dataset <testing csv>` would have fitted a scaler on test data. A docstring is not an amendment to the standard. |
| 10 | Skills route UNSW input through the project's own cleaning before any fit | `RareCategoryGrouper` documents that it must be fitted on the cleaned partition, because deduplication changes the row-count denominator and therefore which categories cross the 1% threshold. Reading raw and fitting the frozen pipeline satisfied the rule in name only. |
| 11 | Leakage guards raise `ValueError`, never bare `assert` | Asserts vanish under `python -O`. A guard for a blocking rule must not be removable by an interpreter flag. |
| 12 | RDD (`gentle-ai review`) disabled for this clone at the user's request | Explicitly requested so no consent prompt could block the unattended run. Global setting untouched. GGA remained active on every commit. |

---

## 5. What failed, and what was fixed

### 5.1 Failures during the run

| Failure | Resolution |
|---|---|
| GGA provider failed with no output | Codex was pinned to `gpt-5.6-sol`, rejected for a ChatGPT account. The user repinned it. |
| GGA failed again mid-run | Codex usage limit reached. The user switched `.gga` to `PROVIDER="claude"` (`fe9db4f`). |
| `sdd-explore` could not write its OpenSpec mirror | That agent has no Write tool. The orchestrator materialized the file. |
| A proposal-update agent stopped before writing its Engram mirror | Detected by comparing stores; mirror repaired and verified. |
| A spec agent wrote a condensed Engram mirror instead of the full document | Detected, replaced with the full 25,497-byte text. |
| A tasks document claimed 55 tasks while its own breakdown summed to 52 | Corrected in the file and the mirror resynced. |

### 5.2 Real defects caught by review, not by tests

These are the ones worth remembering.

1. **Target leak in the skills** — `--exclude id`, the invocation documented in
   every `SKILL.md`, failed a subset check, fell to a generic path, and let
   `label` (`attack_cat != "Normal"`, a perfect predictor) into the feature
   matrix. Passing nothing was safe; passing the careful flag leaked.
2. **The `id` column was not dropped** on the same path, and the slow tests
   encoded the violating call.
3. **A guard that could not fire** — the leak check compared `".train" in source`,
   so `fit(pd.concat([result.train, result.test]))` passed while fitting on test.
4. **`--exclude id` crashed** after a round-2 fix, because cleaning removes `id`
   before the validator checks it. Every slow test used `--exclude label`, so the
   suite walked around the bug instead of through it.
5. **Deduplication ran before the train/test split**, reshaping the held-out set
   with a function whose own docstring says "training partition only".
6. **Results could escape `results/`** — `results_dir()` enforces containment and
   simply was not called; a test asserted the bypassing behaviour.
7. **Silent empty subsample** — `stratified_subsample` stringified labels then
   compared them against an int64 column, so every count was zero, the allocator
   returned nothing, and the caller got an empty frame with exit code 0. This was
   in committed foundation code. It never bit the notebooks only because they
   stratify by `attack_cat`, a string.

Item 7 is the one to take seriously: it produced wrong output with a clean exit.
Everything else failed loudly or failed a rule.

### 5.3 Skipped or not done

- **SDD archive** was not run for any change. The `openspec/changes/*` folders
  remain active rather than archived. All artifacts are committed and complete;
  only the archive step is outstanding.
- **`sdd-verify`** was not run as a separate phase. Verification was performed
  inline by each apply phase and independently spot-checked by the orchestrator.
- **`tests/test_repo_hygiene.py` does not scan `skills/`** — its AST literal scan
  covers `src/nids` and `tests` only. The skills carry no dataset literals today,
  but the guard does not enforce that.
- The **zero-variance VIF branch** is present and correct but never exercised on
  this dataset: no feature has exactly zero variance.

---

## 6. Suggested next steps

1. **Archive the six SDD changes** (`openspec/changes/*` → archive) to close the
   cycle and merge delta specs into `openspec/specs/`.
2. **Extend the hygiene AST scan to `skills/`** so the no-hardcoded-counts rule
   covers every Python file in the repository, not two directories.
3. **Investigate Backdoor and Analysis directly.** They are the real failures
   (F1 0.072 and 0.107), and they are failures of class separability, not of
   class balance. Per-class error analysis against the confusion matrices would
   say more than another model.
4. **Re-run the classification notebook without compute caps** on a machine with
   time to spare, and compare. The current numbers are honest but budgeted.
5. **Reconsider the leakage comparison key.** The current `"full_row"` default
   retains 4,294 contradictory rows. Switching to `"features_only"` removes 8,542
   rows instead of 4,248 and eliminates the error floor — a different, also
   defensible, experimental design. The parameter already exists; it is one
   argument.
6. **Report a benchmark caveat prominently in any write-up.** These metrics are
   computed on 78,084 testing rows, not the published 82,332, and are therefore
   not comparable with published UNSW-NB15 results.
