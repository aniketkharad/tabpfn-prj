# PART 0: `AGENTS.md` (paste verbatim)

## Mission

RelPilot is an MCP server (plus a thin Gemini CLI agent) that makes zero-shot predictions over a multi-table relational database using TabPFN-Rel (the hosted TabPFN-3.5 API; no feature engineering, no hand-written SQL joins). It lets an agent act on predictions ONLY where a temporal backtest has shown the model to be trustworthy.

Pitch: "Ask your database about the future. The agent acts only where the model has proven it deserves trust."

## Working rules (non-negotiable)

1. **Human in the loop.**
   - **Installs:** before ANY dependency install or change (`uv add`, `uv pip`, `uv sync` that changes deps) and before ANY dataset download: STOP. State the package or URL, version, purpose and size, then wait for my "proceed".
   - **Keys:** needed are `GEMINI_API_KEY` and `TABPFN_TOKEN`. When needed, tell me to add them to `.env` (copied from `.env.example`) and wait for "done". Never print, log, echo, commit or request secrets in chat. Check presence with boolean checks only. `.env` goes in `.gitignore` in the first commit.
   - **Checkpoints:** at every CHECKPOINT, stop and summarize what is done, what is next, and what you need from me.
   - **Narration:** before any command that touches the network or files outside `src/`, write one line: "About to do X because Y".
2. **Environment: uv only.**
   - Use `uv init`, `uv add`, `uv run`, `uv sync`. Pin Python in `.python-version` (3+ stable versoin unless a dependency needs otherwise; ask).
   - The venv is project-local. Prefer `uv run --env-file .env` over adding python-dotenv (verify support in the installed uv).
   - Never use `pip`, global installs or `sudo`. Never touch files outside the repo. Flag anything needing elevated privileges.
   - Assume this runs inside my VM or container.
   - Network is allowed only to: the Gemini API, the Prior Labs API, PyPI via uv (after approval), and dataset URLs I approved.
3. **Minimum code.** Hand-written Python in `src/` must stay at or under about 400 lines (tests excluded). Prefer library functionality (TabPFN-Rel/RPI, the RelArena model registry, the MCP SDK, google-genai) over custom code. No frontend framework. If a feature needs more than about 60 lines, justify it and ask first.
4. **Quota is limited.** Config defaults: `MAX_TRAIN_ENTITIES=2000`, `MAX_PREDICT_ENTITIES=500`, one backtest cutoff, at most 3 TabPFN fits per run, `MAX_GEMINI_CALLS_PER_SESSION=30`. Print API usage after each fit. On HTTP 429, stop and tell me.
5. **Verify, don't recall.** Never hard-code API or model facts from memory. Record verified facts with source URLs in `docs/NOTES.md`. The Gemini model id comes from env `GEMINI_MODEL` (pick the latest Flash from the models list).
6. **Honesty.** Numbers in README or RESULTS come only from runs actually executed. Report null or negative results as they are. Keep run artifacts.

## Architecture (MVP)

```
workspaces/<name>/
  data/*.csv|parquet      relational tables (any DB export)
  schema.yaml             PKs, FKs, time columns (or introspected)
  tasks/*.yaml            declarative prediction tasks   (community unit 1)
  skills/*.md             playbooks the agent can read   (community unit 2)
src/relpilot/
  workspace.py            load tables, schema, tasks, skills
  engine.py               thin wrapper over TabPFN-Rel RPI + free baselines; backtest; trust report
  gate.py                 backtest-derived act-threshold; act vs review decision
  server.py               MCP server (stdio)
  agent.py                Gemini CLI that connects to the MCP server
runs/<run_id>/            artifacts (gitignored)
outbox.jsonl             proposed actions (audit trail)
```

## MCP tools (5)

1. `describe_workspace()`: tables, columns, FKs, time columns, available tasks and skills.
2. `read_skill(name)`.
3. `run_task(task, model="tabpfn-rel-client", cutoff=None)`: fit on labels strictly before the validation cutoff, backtest on the latest fully-labelled period, predict current entities, write artifacts, and return a compact trust report plus the top-K.
4. `trust_report(run_id)`: metric vs baselines (constant, plus LightGBM if the registry exposes it at no API cost), calibration bins and ECE, precision and lift at k, and the recommended act-threshold with its supporting sample size.
5. `propose_actions(run_id, action, min_precision=0.8)`: the gate. Entities above the backtest-derived threshold go to `outbox.jsonl` with status `proposed`. The rest go to a `needs_review` list with reasons. If no threshold reaches the precision target, propose nothing and say so.

Rules: tool outputs are compact JSON (<= 3 KB); large outputs go to `runs/`; the data source is read-only; there is no free-form SQL tool.

## Task YAML (the community unit)

`name`, `description`, entity table and id column, label definition (declarative, e.g. "no rows in table X for the entity in the next N days"), time column, window length, action hints. The exact schema is decided in Prompt 1 from what RPI supports.

## Deliberately NOT building

Web UI (optional Prompt 4), auth, Postgres/BigQuery connectors (document as extension points), a custom subgraph sampler or kNN exemplar retrieval, real external action integrations, multi-user support.

---
