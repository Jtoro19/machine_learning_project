# Orchestrator Decisions — dashboard

Decisions taken autonomously during the unattended run, with rationale.

| # | Decision | Rationale |
|---|---|---|
| 1 | The planning artifacts' claim that "Strict TDD is enabled for this project" was **overruled** before apply. Strict TDD is DISABLED; no observed-RED step is required. | `openspec/config.yaml` records `strict_tdd: false`, set by explicit user correction at SDD init. Tests are still mandatory for the dashboard helpers per AGENTS.md, but test-first ordering is not. Acting on the false premise would have imposed a ceremony this project explicitly opted out of. |
| 2 | Notebook 3's `notebook_id` is `classification` (settled), but the plan's path-stem matching for the model comparison table is kept. | Matching by table stem is more robust than a hardcoded folder name and costs nothing now that the id is fixed. |
| 3 | The headless Streamlit smoke check runs under `timeout 45`. | An unattended run cannot afford a server process that never exits. A timeout kill after a clean start is a pass; a traceback is a failure. |
