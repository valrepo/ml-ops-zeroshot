# Data Model: Zero-Shot Text Classification Platform

**Feature**: `001-zero-shot-platform` | **Date**: 2026-11-11 | **Spec**: [spec.md](spec.md)

Every entity is defined once in `src/zeroshot/domain/` as a pydantic model (or a frozen
dataclass for runtime-only values). Services and the UI import these definitions and never
redefine them (Principles VI and XIII).

## Overview

```text
ModelDefinition (configs/models.yaml) ──1:0..1──> ModelInstallation (MODEL_DIR/<slug>/<rev>/)
        │                                                  │ .complete.json = CompletionMarker
        │                                                  │
LabelSet (data/label_sets.yaml)                             │ ready
        │                                                   ▼
        └──────────────> ClassificationRequest ──> ClassificationResult ──> PredictionRecord
                                  │                                        (MinIO object)
                                  └──────────────> PredictionOutcome (failed | computed_not_persisted | succeeded)
```

## 1. ModelDefinition

Source: `configs/models.yaml`, loaded at startup and read-only at runtime.

| Field | Type | Required | Validation |
|-------|------|----------|------------|
| `key` | str | yes | Stable app identifier, `^[a-z0-9][a-z0-9-]{1,62}$`, unique |
| `name` | str | yes | Display name, 1–80 chars |
| `repo_id` | str | yes | `^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$`; no `..` segments |
| `task` | literal | yes | Must be `zero-shot-classification` |
| `revision` | str | yes | 40-char lowercase hex commit SHA. Branch or tag names (e.g. `main`) are **rejected** in the final configuration |
| `license` | str \| null | no | SPDX-style identifier when known |
| `inference_defaults.multi_label` | bool | yes | Explicit, never inherited implicitly |
| `inference_defaults.hypothesis_template` | str | yes | Must contain exactly one `{}` |
| `download.allow_patterns` | list[str] | no | Defaults from research R5 |
| `download.ignore_patterns` | list[str] | no | Defaults from research R5 |

Derived values: `slug = repo_id.replace("/", "--")`. The install path is
`MODEL_DIR/<slug>/<revision>/`; it is resolved and must stay inside `MODEL_DIR`
(`Path.resolve().is_relative_to(MODEL_DIR.resolve())`).

Catalogue rules: at least one model is required (spec edge case "No configured models"
produces a clear message instead of a crash); `key` and `repo_id` must each be unique.
Initial values are in [research R4](research.md#r4-model-revisions-licences-and-default-settings-verified).

## 2. CompletionMarker

File: `MODEL_DIR/<slug>/<revision>/.complete.json`, written last, before the atomic rename.

| Field | Type | Notes |
|-------|------|-------|
| `marker_version` | str | `"1"` |
| `model_id` | str | Equals `ModelDefinition.repo_id` |
| `revision` | str | Resolved immutable SHA; must equal the configured revision |
| `downloaded_at` | str | ISO-8601 UTC (`...Z`) |
| `files` | list[str] | Relative paths of downloaded files |
| `total_bytes` | int | ≥ 0 |

A marker is **valid** only if it parses, `model_id` and `revision` match the definition, and
every listed file exists.

## 3. ModelInstallation (runtime view)

Computed by `ModelManager.list_models()`; it is never stored.

| Field | Type | Notes |
|-------|------|-------|
| `definition` | ModelDefinition | |
| `status` | enum `ModelStatus` | See the state machine below |
| `local_path` | Path \| None | Set when the model is ready |
| `marker` | CompletionMarker \| None | |
| `error` | str \| None | User-safe message from the most recent failed attempt |
| `in_use` | bool | Lease count > 0 (from `ModelUsageRegistry`) |

`ModelStatus` values: `NOT_AVAILABLE`, `DOWNLOADING`, `READY`, `FAILED`, `INCOMPLETE`.

### State machine

```text
            download requested                validation OK + rename
NOT_AVAILABLE ───────────────────> DOWNLOADING ──────────────────────> READY
     ▲   ▲                              │                                │
     │   │    failure (staging removed) │                                │ delete (no active lease)
     │   │                              ▼                                │
     │   └────────── retry ───────── FAILED                              │
     │                                                                   │
     │   download requested (final path replaced)                        │
     │ ┌──────────────────────────── INCOMPLETE <── found at startup/refresh:
     │ │                                 │        final path exists, marker missing/invalid
     │ └──> DOWNLOADING                  │ delete
     └───────────────────────────────────┴───────────────────────────────┘
```

- `READY` holds **only** when a valid marker exists at the final path.
- `FAILED` lasts only in memory for the current process: after a restart, a failed model
  appears as `NOT_AVAILABLE`, because staging directories are cleaned up and never treated
  as models.
- `INCOMPLETE`: the final path exists **without** a valid marker (for example, a directory
  created by hand, or a marker whose `model_id`/`revision` does not match, or a listed file
  that is missing). It is shown as "Incomplete installation", is never `READY`, and can be
  deleted or downloaded again. A new download replaces the final path atomically.
- `delete` while a lease is active raises `ModelInUseError`, and the state is unchanged.

## 4. LabelSet

Source: runtime file `LABEL_SETS_FILE` (default `data/label_sets.yaml`), seeded from
`configs/labels.yaml`.

| Field | Type | Validation |
|-------|------|------------|
| `key` | str | `^[a-z0-9][a-z0-9_-]{0,62}$`, unique, immutable after creation |
| `name` | str | 1–80 chars after trimming |
| `labels` | list[str] | ≥ 2 entries; each is trimmed, non-empty and ≤ 100 chars; **no duplicates under `casefold()`**; order is preserved; the user's text is stored as entered after trimming |

The seed (`configs/labels.yaml`) contains two sets:
- `default`: `sports`, `politics`, `technology`, `finance` (the brief §13 demo);
- `extended`: the same four plus `health` and `education` (the brief §3.3 example).
Validation errors identify the offending label and its position (FR-06, spec edge case
"Empty or duplicate labels").

## 5. InferenceParameters

| Field | Type | Notes |
|-------|------|-------|
| `multi_label` | bool | From the model's `inference_defaults` |
| `hypothesis_template` | str | From the model's `inference_defaults`; exactly one `{}` |
| `device` | literal `"cpu"` \| `"mps"` | From `INFERENCE_DEVICE` (default `cpu`). Scores can differ slightly between devices, so the device is a result-affecting parameter (Principle III) |

In v1 these come from the model definition and settings only; the UI does not override
them. They are stored verbatim in every record (FR-07, Principle XII).

## 6. ClassificationRequest (runtime)

| Field | Type | Validation (all before inference, FR-07) |
|-------|------|------------------------------------------|
| `request_id` | UUID4 | Generated by the prediction service |
| `text` | str | `text.strip()` non-empty; `len(text) <= MAX_INPUT_CHARS` (default 5000). The text is kept as submitted, not trimmed |
| `model_key` | str | Must exist in the catalogue and be `READY` |
| `label_set_key` | str | Must exist and be valid |

## 7. ClassificationResult (runtime, returned by the classifier)

| Field | Type | Notes |
|-------|------|-------|
| `ranked` | list[`LabelScore`(label, score)] | One entry per candidate label, sorted by `(-score, candidate_index)` |
| `predicted_class` | str | `ranked[0].label` |
| `confidence` | float | `ranked[0].score` |
| `model_id`, `model_revision` | str | From the marker, so the revision is the one actually loaded |
| `inference_parameters` | InferenceParameters | Effective values |
| `latency_ms` | float | Measured around the pipeline call only |

Invariants: every score is finite and in [0, 1]; in single-label mode,
`abs(sum(scores) - 1) < 1e-6`; the set of labels in `ranked` equals the candidate labels.

## 8. PredictionRecord (persisted)

The authoritative JSON Schema is
[contracts/prediction-record.schema.json](contracts/prediction-record.schema.json); the
pydantic model `PredictionRecord` mirrors it, and a unit test checks the two stay in sync.

- **Object key**: `predictions/{timestamp:%Y}/{timestamp:%m}/{timestamp:%d}/{request_id}.json`,
  where the date parts come from the record's own UTC `timestamp`.
- **Content type**: `application/json; charset=utf-8`. Encoding: UTF-8 with
  `ensure_ascii=False` and no NaN or Infinity values (`allow_nan=False`).
- **`schema_version`**: `"1.0"`. Changes that add fields increase the minor number; renaming
  or removing a field increases the major number and needs a contract note (Principle XIV).
- **Write semantics**: `put_object` is called once per request ID. A request ID is never
  reused, and the service does not retry under a new ID.
- **Cross-field rules**:
  - `confidence == ranked_predictions[0].score`;
  - `predicted_class == ranked_predictions[0].label`;
  - `{p.label for p in ranked_predictions} == set(candidate_labels)`, with equal lengths;
  - `timestamp` ends with `Z`.

## 9. PredictionOutcome (runtime, returned to the UI)

| Field | Type | Notes |
|-------|------|-------|
| `status` | enum `OutcomeStatus` | `FAILED`, `COMPUTED_NOT_PERSISTED`, or `SUCCEEDED` |
| `request_id` | UUID | Always set, so the UI can quote it for log correlation |
| `result` | ClassificationResult \| None | Set unless `FAILED` |
| `record` | PredictionRecord \| None | Set unless `FAILED` |
| `object_key` | str \| None | Set only when `SUCCEEDED` |
| `error` | ZeroShotError \| None | Set unless `SUCCEEDED` |

Exactly one status per request (FR-12). `FAILED` means nothing was written to MinIO.

## 10. Settings (environment)

Defined in [contracts/configuration.md](contracts/configuration.md). Loaded once at startup
by `Settings` (pydantic-settings). Invalid values stop the app with a message that names the
variable but never prints its value when it is a secret.
