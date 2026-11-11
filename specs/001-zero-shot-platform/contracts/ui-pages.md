# Contract: UI Pages

Two pages registered with `st.navigation` (FR-01). Pages call only the services in
[services.md](services.md). Each element lists the requirement it satisfies.

## Administration page

| Element | Behavior | Req |
|---------|----------|-----|
| Banner | "No authentication — local/trusted use only." always visible | NFR-04 |
| Storage paths panel | Shows `MODEL_DIR` and `HF_HOME` (and cache targets) separately | US-03 |
| Model table | Per model: name, `repo_id`, revision (short + full on hover), licence, status (`Not available` / `Downloading…` / `Ready` / `Failed: <msg>` / `Incomplete installation`), local path when ready, "in use" badge | US-01, US-03 |
| Empty catalogue | "No models configured — add one to `configs/models.yaml`." | Edge case |
| Refresh button | Re-reads installation state | FR-03 |
| Download | Multiselect of models not yet ready + **Download selected** button; `st.status` per model; summary lists each outcome independently | US-02, AC-05/06 |
| Delete | Per ready or incomplete model; **Delete…** opens an inline two-step confirmation box stating the exact directory removed and that the HF cache is untouched; refused with explanation if in use | US-04, FR-05 |
| Clear HF cache | Button opens an inline two-step confirmation listing target dirs; reports `cleared (X MB)` / `nothing to clear` / error; refused while any model in use or download running | US-05 |
| Label sets | List of sets; create/edit form (key — immutable after create, name, labels one-per-line); inline validation errors naming the label; delete with inline two-step confirmation | US-06, AC-10 |
| Load warnings | Invalid label-set entries skipped at load are listed | FR-06 |

## Classification page

| Element | Behavior | Req |
|---------|----------|-----|
| Text area | Character counter `n / MAX_INPUT_CHARS` | US-07 |
| Model selector | Only `Ready` models; if none: warning "No model is ready — an administrator must download one on the Administration page." and Classify disabled | US-07, AC-17 |
| Label-set selector | Only valid sets; shows the labels of the selection | US-07 |
| Classify button | Calls `PredictionService.predict` only | FR-08, FR-12 |
| Spinner | During model load / inference | — |

Result rendering by `PredictionOutcome.status`:

| Status | Display |
|--------|---------|
| `SUCCEEDED` | ✅ "Saved" + predicted class + confidence (as %) ; table of all labels with scores, descending (bar column); model name, `repo_id`, revision; timestamp (UTC); request ID; object key |
| `COMPUTED_NOT_PERSISTED` | ⚠️ "Prediction produced but **not persisted** — storage error: <msg>." Result shown below the warning, explicitly labelled *not saved*; request ID shown. Never shows "Saved". |
| `FAILED` | ❌ error `user_message` (e.g. "Text exceeds the 5000-character limit."); no result shown |

The page never renders a success state before `SUCCEEDED` is returned (FR-08).

**Confirmation pattern**: destructive actions use an inline two-step confirmation (a highlighted box with *Confirm* / *Cancel*, state kept in `st.session_state`) rather than `st.dialog`. Streamlit's headless `AppTest` cannot drive dialog fragments, so dialogs could not be verified automatically (found during implementation, 2026-11-11).
