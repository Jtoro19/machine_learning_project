# Orchestrator Decisions — notebook-01-eda-reduction-clustering

Decisions taken autonomously during the unattended run, with rationale.

| # | Decision | Rationale |
|---|---|---|
| 1 | `notebook_id` is `eda_reduction_clustering`; results land in `results/eda_reduction_clustering/`. The notebook file keeps `notebooks/01_eda_reduction_clustering.ipynb`. **FINAL, no longer provisional.** | The committed `NOTEBOOK_ID_PATTERN = ^[a-z][a-z0-9_]{0,63}$` in `src/nids/paths.py` rejects a leading digit. Verified empirically: `results_dir("01_eda_reduction_clustering")` raises `ValueError`. The alternatives were editing `paths.py` (owned by another change, wider blast radius) or renaming the notebook (loses the execution-order signal). Option A changes one constant and preserves both the ordering signal and the package contract. |
| 2 | VIF computed as `1 / (1 - R^2)` from `sklearn.linear_model.LinearRegression`, no new dependency. | `statsmodels` is not declared. The quantity is identical, not an approximation, and `LinearRegression` fits an intercept by default — the constant column `statsmodels` callers must add manually and routinely forget, which silently inflates VIF. |
| 3 | Infinite metrics register as the string `"inf"`, never a float infinity. | `json.dumps(float("inf"))` emits the bare token `Infinity`, which is invalid JSON and would make the committed `manifest.json` unparseable by a strict reader. |
| 4 | The execute command carries `--ExecutePreprocessor.timeout=-1`. | Several nbconvert releases default to a 30-second per-cell timeout; the K-Means sweep exceeds it. |
| 5 | The "raw vs log1p" ECDF figure becomes two panels of the same curves under two axis parameterisations. | An ECDF is invariant under any strictly increasing transform, so a literal overlay would draw the identical curve twice. Name and intent preserved. |
