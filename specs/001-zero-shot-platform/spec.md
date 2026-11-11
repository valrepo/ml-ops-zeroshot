# Feature Specification: Zero-Shot Text Classification Platform

**Feature Branch**: `001-zero-shot-platform`

**Created**: 2026-11-11

**Status**: Draft

**Input**: the assignment brief
`homework02-MLOps_Project_ ZeroShot_TextClassification_SpecKit (1).md`, which is kept
locally and deliberately **not committed**. It was refined through a draft specification
(v1.0.0, 2026-11-10) that has since been removed. References to "the brief §n" below cite
its sections.
**This document is now self-contained and authoritative.** Governed by constitution
v1.1.1.

> **ID convention**: user stories (US-xx), functional requirements (FR-xx), non-functional
> requirements (NFR-xx) and acceptance criteria (AC-xx) are this specification's own stable
> identifiers. The brief's 22 numbered acceptance criteria are mapped to them in
> "Assignment acceptance-criteria mapping" under Success Criteria.
>
> **File name**: the brief's sample layout names this file `specification.md`. Spec Kit's
> commands (`/speckit-plan`, `/speckit-tasks`, `/speckit-analyze`) require `spec.md`, so the
> Spec Kit name is kept (AC-23).
>
> **Technology names**: the constitution (Principle I) and the assignment fix the technology
> stack: Hugging Face zero-shot models, a web UI with separate pages, and MinIO as the
> prediction store. Product names, the required bucket name and environment-variable names
> therefore appear in this spec as **mandated constraints**, not as design choices. Every
> other "how" decision belongs in `plan.md`.

## Overview

A small, reproducible MLOps learning platform for zero-shot text classification. An
**administrator** manages locally available zero-shot models and reusable candidate-label
sets. A **user** submits text, chooses a ready model and a saved label set, and sees the
predicted class and all labels ranked by score. Every successful classification is stored
as a structured, traceable record in object storage, so evaluators can inspect it
independently of the UI session.

The platform shows practical MLOps foundations: reproducible dependencies and model
revisions, model lifecycle management, inference traceability, object-storage persistence,
configuration management, logging, automated testing, containerized local infrastructure
and CI validation.

**Problem**: a notebook-based zero-shot classifier demonstrates inference but provides no
repeatable workflow, and does not make model availability, model identity, inputs, outputs
and persistence observable or reproducible.

### Actors

- **Administrator**: operates the local or trusted environment. Inspects, downloads and
  deletes models, clears the model cache, and manages candidate-label sets. There is no
  authentication in v1; the administration page is for local or trusted use only.
- **User**: classifies text on the classification page. No accounts or persistent user
  identity are required in v1.
- **Evaluator / developer**: reproduces the project from a clean checkout and inspects
  stored predictions.

## User Scenarios & Testing *(mandatory)*

**MVP slice**: US-01 + US-02 + US-07 + US-08. These let an administrator make one model
ready, and let a user classify text and get a stored, traceable record.

### US-07 - Classify text (Priority: P1)

As a user, I want to submit text with a selected model and label set, so that I receive a
zero-shot classification result.

**Why this priority**: classification is the core value of the platform. Every other story
exists to support or record it.

**Independent Test**: with one model already in the ready state and a saved label set
available, enter text, choose the model and label set, and confirm that a predicted class,
its score, and a ranked score for every label are displayed with the model and timestamp.

**Acceptance Scenarios**:

1. **Given** the classification page is open, **When** I enter or paste text, **Then** the
   text is accepted into the input area.
2. **Given** some models are ready and some are not, **When** I open the model selector,
   **Then** only ready models are offered.
3. **Given** saved label sets exist, **When** I open the label-set selector, **Then** only
   valid saved label sets are offered.
4. **Given** the text is empty or whitespace only, **When** I submit, **Then** the request is
   rejected before classification with a clear message.
5. **Given** the text is longer than the configured maximum length, **When** I submit,
   **Then** the request is rejected before classification with a message stating the limit.
6. **Given** no model is ready or the selected label set is invalid, **When** I try to
   submit, **Then** submission is prevented and a clear message explains why.
7. **Given** valid text, a ready model and a valid label set, **When** I submit, **Then** I
   receive exactly one predicted class and a score for every candidate label.
8. **Given** a successful result, **When** it is displayed, **Then** labels are sorted by
   descending score, the predicted class is the highest-scoring label, and the model used
   and the prediction timestamp are shown.

---

### US-08 - Persist a prediction (Priority: P1)

As a project evaluator, I want every successful classification saved in object storage as a
structured record, so that I can inspect the output independently of the UI session.

**Why this priority**: the assignment requires every successful classification to be stored
in MinIO; a classification is not fully successful without it.

**Independent Test**: submit one valid classification, then inspect the storage bucket and
confirm that exactly one record exists at the expected date-based key, containing every
required field.

**Acceptance Scenarios**:

1. **Given** the configured bucket does not exist, **When** the application starts or first
   writes, **Then** the bucket is created automatically.
2. **Given** a successful classification, **When** it completes, **Then** one record is
   written as valid JSON to the bucket.
3. **Given** a stored record, **When** I inspect its location, **Then** the key follows
   `predictions/YYYY/MM/DD/{request_id}.json` using the UTC date of the prediction.
4. **Given** a stored record, **When** I inspect its contents, **Then** it contains every
   required field of the Prediction Record (see Key Entities).
5. **Given** classification succeeded but the storage write failed, **When** the result is
   shown, **Then** the UI states explicitly that a prediction was produced but **not
   persisted**, and does not report full success.

---

### US-01 - Inspect configured models (Priority: P1)

As an administrator, I want to see the models supported by the application and whether each
is available locally, so that I know which models can be used for classification.

**Why this priority**: users can only classify with ready models, so model visibility is a
precondition for the MVP.

**Independent Test**: open the administration page with the initial configuration and no
downloads, and confirm that both required models are listed as not available.

**Acceptance Scenarios**:

1. **Given** the initial model configuration, **When** I open the administration page,
   **Then** at least the two required models are listed.
2. **Given** a model is not downloaded, **When** I view the list, **Then** its status shows
   it is not locally available.
3. **Given** a model is fully downloaded and validated, **When** I view the list, **Then**
   its status shows it is ready.
4. **Given** a model has a failed or incomplete download, **When** I view the list, **Then**
   it is not presented as ready.

---

### US-02 - Download a model (Priority: P1)

As an administrator, I want to download one or more configured models, so that they can be
used locally for classification.

**Why this priority**: without at least one downloaded model no classification can happen.

**Independent Test**: from a state with no models downloaded, select both models, download
them in one action, and confirm that each reaches the ready state and that its outcome is
reported separately.

**Acceptance Scenarios**:

1. **Given** a configured model is not available locally, **When** I request its download,
   **Then** its files are downloaded into the managed model storage.
2. **Given** I select several models, **When** I request their download, **Then** each is
   downloaded and reported independently, and one failure does not stop the others.
3. **Given** a model is already available at the configured revision, **When** I request its
   download again, **Then** the existing files are reused and no unnecessary download occurs.
4. **Given** a download fails, **When** the operation ends, **Then** a useful error is
   reported and no partial download is presented as ready.
5. **Given** a download completes and passes validation, **When** the operation ends,
   **Then** a completion marker is written and the model status changes to ready.
6. **Given** a download is in progress, **When** I watch the page, **Then** I see progress
   or a clear operation status where practical.

---

### US-06 - Manage candidate-label sets (Priority: P2)

As an administrator, I want to create and maintain reusable candidate-label sets, so that
users classify text against consistent categories.

**Why this priority**: classification needs a label set, but a shipped default label set is
enough for the MVP; editing label sets adds flexibility.

**Independent Test**: create a named label set with three labels, save it, restart the
application, and confirm that it is still available on the classification page.

**Acceptance Scenarios**:

1. **Given** the administration page, **When** I create a named label set with labels,
   **Then** it can be saved.
2. **Given** an existing label set, **When** I add or remove labels and save, **Then** the
   changes are kept.
3. **Given** saved label sets, **When** I choose one, **Then** it is available for
   classification.
4. **Given** a label set containing an empty label or a duplicate (after trimming and
   case-insensitive comparison), **When** I save it, **Then** it is rejected and the problem
   is identified.
5. **Given** a label set with fewer than two distinct labels, **When** I save or use it,
   **Then** it is rejected.
6. **Given** valid saved label sets, **When** the application restarts, **Then** they are
   still available.

---

### US-03 - Inspect model details (Priority: P2)

As an administrator, I want to inspect a model's identity and local storage information, so
that I understand which artifact will be used for classification.

**Why this priority**: supports traceability and troubleshooting but is not needed to
classify.

**Independent Test**: with one model ready, open its details and confirm that the name,
repository identifier, revision, availability and storage locations are shown.

**Acceptance Scenarios**:

1. **Given** a listed model, **When** I view it, **Then** I see its human-readable name and
   its Hugging Face identifier.
2. **Given** a listed model, **When** I view it, **Then** its configured revision is visible.
3. **Given** a listed model, **When** I view it, **Then** its local availability and the
   managed model directory are shown.
4. **Given** the model cache location differs from the managed model directory, **When** I
   view the administration page, **Then** the cache location is shown separately.

---

### US-09 - Reproduce the application locally (Priority: P2)

As a developer or evaluator, I want documented setup and test instructions, so that I can
run and verify the project from a clean checkout.

**Why this priority**: reproducibility is a graded MLOps outcome and a constitution
principle, but it delivers no end-user feature by itself.

**Independent Test**: on a clean machine, follow only the README to install, start the
storage service and the application, and run the unit and integration tests.

**Acceptance Scenarios**:

1. **Given** a clean checkout, **When** I follow the README, **Then** the environment is
   installed from the committed dependency lock file using documented commands.
2. **Given** the environment is installed, **When** I follow the README, **Then** the local
   storage service starts through Docker Compose.
3. **Given** the storage service is running, **When** I follow the README, **Then** the
   application starts using documented commands.
4. **Given** no network access and no downloaded models, **When** I run the unit tests,
   **Then** they run and pass.
5. **Given** a real local storage service, **When** I run the integration tests, **Then**
   they run against it.
6. **Given** a change is pushed, **When** CI runs, **Then** it provisions the storage service
   and executes the integration tests.

---

### US-04 - Delete a downloaded model (Priority: P3)

As an administrator, I want to delete a model the application downloaded, so that I can
reclaim disk space.

**Why this priority**: housekeeping; not required to classify.

**Independent Test**: with one ready model that is not in use, delete it after confirming,
and confirm that it is no longer listed as ready.

**Acceptance Scenarios**:

1. **Given** a delete request, **When** its target is outside the managed model root,
   **Then** the operation is refused.
2. **Given** a ready model, **When** I request deletion, **Then** an explicit confirmation is
   required before anything is removed, and the scope of the deletion is shown.
3. **Given** a model is being used by a classification in progress, **When** I request its
   deletion, **Then** the deletion is refused and the reason is explained.
4. **Given** a successful deletion, **When** I view the model list, **Then** the model is no
   longer listed as ready.
5. **Given** a deletion fails, **When** the operation ends, **Then** the failure is reported
   and the displayed state is refreshed.
6. **Given** a managed model is deleted, **When** the operation ends, **Then** unrelated
   model-cache content is left untouched.

---

### US-05 - Clear the model cache (Priority: P3)

As an administrator, I want to clear the Hugging Face cache, so that I can reclaim cache
storage when needed.

**Why this priority**: housekeeping; not required to classify.

**Independent Test**: with cache content present and no classification running, request
cleanup and confirm that the targeted location is shown beforehand and that the outcome is
reported.

**Acceptance Scenarios**:

1. **Given** the administration page, **When** I want to clear the cache, **Then** it is an
   explicit action that shows the targeted cache location before it runs.
2. **Given** the cache directory does not exist, **When** I request cleanup, **Then** it is
   treated as a safe no-op and the outcome is reported.
3. **Given** a classification is in progress, **When** I request cleanup, **Then** files it
   needs are not removed.
4. **Given** a cleanup runs, **When** it ends, **Then** its result (success, no-op or
   failure) is reported clearly.
5. **Given** the README, **When** I read it, **Then** it explains the difference between the
   managed model directory and the model cache, and the consequences of clearing each.

---

### Edge Cases

| Situation | Expected behavior |
|-----------|-------------------|
| Empty or whitespace-only input | Rejected before classification; the error is explained. |
| Text exceeds the maximum length | Rejected before classification; the configured limit is stated. |
| Fewer than two candidate labels | Classification is prevented and the requirement is explained. |
| Empty or duplicate labels | The label set is rejected and the problem is identified. |
| No configured models | A useful message is shown on both pages; no classification is attempted. |
| Selected model not downloaded | Classification is prevented; the message directs the administrator to download it. |
| Model download fails or is interrupted | Failure is reported; the model is not marked ready and no usable partial installation is left. |
| Model cannot be loaded | A model-loading error is reported; diagnostic logs are kept without exposing secrets. |
| Classification fails | Reported as not completed; no prediction record is written. |
| Storage service unavailable | Reported as "prediction produced but not persisted"; never shown as fully successful. |
| Bucket creation fails | Reported as a storage initialization failure; the application does not silently continue without persistence. |
| Deletion requested while the model is in use | Refused, with the reason explained. |
| Cache cleanup requested while a classification runs | Files in use are not removed. |
| Cache directory does not exist | Cleanup is a safe no-op and the outcome is reported. |
| Cache cleanup fails | Failure is reported; the application stays stable. |
| Model identifier or label crafted to escape a directory (e.g. `../`) | Rejected; no file outside the managed directories is read, written or deleted. |
| Two labels receive identical scores | Ties are broken deterministically by the label's position in the candidate list. |

## Requirements *(mandatory)*

### Functional Requirements

**FR-01 - Application navigation**
- The application MUST provide two distinct pages: **Administration** (model and label-set
  management) and **Classification** (text input, model and label-set selection,
  classification and result display).
- Page code MUST be presentation-only; all business logic MUST live in separately testable
  services.

**FR-02 - Configured model catalogue**
- Model definitions MUST be loaded from a documented configuration file.
- The initial catalogue MUST include:

  | Hugging Face model identifier | Display name |
  |-------------------------------|--------------|
  | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | DeBERTa Zero-Shot v2 |
  | `facebook/bart-large-mnli` | BART MNLI |

- Each definition MUST contain: a stable identifier, display name, repository identifier,
  task (`zero-shot-classification`), an explicit revision, and a licence when available. A
  definition MAY override the default classification settings (multi-label behavior and
  hypothesis template); each model's defaults MUST be verified against its model card during
  planning.
- The revision used for reproducible classification MUST be immutable. A mutable reference
  (e.g. `main`) MAY be used only for initial discovery and MUST NOT be treated as pinned.
- Adding a compatible model SHOULD require only a configuration change.

**FR-03 - Model download and local availability**
- The system MUST determine whether a configured model at its revision is already available
  locally, and MUST NOT download it again unnecessarily.
- The system MUST download missing models from Hugging Face into the managed model root.
- The administrator MUST be able to download one or more selected models in a single action;
  each model's outcome MUST be tracked and reported independently.
- A download MUST be validated before it is marked ready. Validation MUST NOT require running
  a classification; the exact checks are defined in the plan.
- A completion marker (model identifier, resolved immutable revision, download timestamp)
  MUST be written only after validation succeeds. A model without a valid marker is never
  ready.
- Network, repository, revision, disk and loading failures MUST NOT leave a partial download
  exposed as ready. Downloads SHOULD go to a temporary location and become ready only after
  validation.
- Errors returned to the UI MUST be useful and MUST NOT expose secrets.
- The administrator MUST be able to refresh the availability list.

**FR-04 - Storage and cache configuration**
- The following MUST be configurable through the environment: managed model directory
  (`MODEL_DIR`), Hugging Face cache root (`HF_HOME`), Hugging Face access token for gated or
  private models (`HF_TOKEN`), storage-service endpoint, credentials, secure-connection
  setting and bucket (`MINIO_BUCKET`), application version, compute device for
  classification (`INFERENCE_DEVICE`, default CPU, never chosen automatically), and maximum
  input length
  (`MAX_INPUT_CHARS`, default 5000 characters).
- The managed model directory and the cache root MUST have distinct, documented
  responsibilities; the system MUST NOT assume they are the same directory.
- The system MUST use only supported cache configuration. It relies on `HF_HOME` and not on
  the deprecated `TRANSFORMERS_CACHE` listed in the assignment's example; this deviation MUST
  be recorded in an ADR and in the README.
- Downloaded models, cache directories, `.env` files and credentials MUST be excluded from
  version control.

**FR-05 - Model deletion and cache cleanup**
- Deleting a managed model and clearing the model cache MUST be separate operations.
- Both MUST validate their target paths and MUST NEVER recursively delete an arbitrary path
  supplied by the UI or user.
- Both MUST be coordinated with model loading and active classification so that files in use
  are not removed.
- Deleting a model MUST release its cached loaded instance.
- The UI MUST make the scope of each destructive operation clear before it runs.

**FR-06 - Candidate-label management**
- Named label sets MUST be loaded and saved through a documented, simple configuration
  mechanism and MUST survive application restarts.
- Each label set MUST have a stable key, a display name, and at least two distinct, non-empty
  labels.
- Whitespace MUST be handled consistently; empty labels and duplicates (after trimming and
  case-insensitive comparison) MUST be rejected. The user-visible label text MUST be
  preserved as entered for classification.
- The classification page MUST use saved label sets only; ad hoc label editing on that page
  is out of scope for v1.

**FR-07 - Zero-shot classification**
- Classification MUST use a Hugging Face zero-shot model with the PyTorch backend
  (constitution Principle I).
- Inputs: text, a locally ready model, at least two valid candidate labels, and explicit
  classification settings (multi-label behavior, hypothesis template, and compute device).
  The compute device can change scores slightly, so it is a result-affecting setting and
  is recorded (constitution Principle III).
- Validation (non-empty text within `MAX_INPUT_CHARS`; at least two distinct, non-empty
  labels) MUST happen before classification.
- Outputs: predicted class, its confidence score, every candidate label with its score, the
  model identifier and immutable revision used, the effective settings, and the latency when
  measured.
- Scores MUST stay associated with their labels and be sorted in descending order for display
  and storage. The predicted class MUST be the highest-scoring label; ties MUST be broken by
  the label's position in the candidate list.
- Score meaning MUST be documented: in single-label mode, scores are normalized across labels
  and sum to 1; in multi-label mode, each label is scored independently.
- Default settings MUST be documented and deterministic, and the effective settings MUST be
  recorded with each prediction.
- Loaded models SHOULD be reused across requests, and more large models than necessary MUST
  NOT be held in memory. Reuse MUST be invalidated when a model is deleted or its
  configuration changes.

**FR-08 - Classification user interface**
- The classification page MUST provide: a text input area, a selector listing only ready
  models, a selector listing only valid saved label sets, and an explicit classify action.
- It MUST display: the predicted class and its score, all labels with scores in descending
  order, the model identifier and revision, and the prediction timestamp.
- It MUST display useful validation, model-loading, classification and storage errors.
- It MUST NOT show a fully successful state until storage has succeeded.
- It MUST classify only through the prediction workflow (FR-12).

**FR-09 - Prediction persistence**
- MinIO MUST be the system of record for classification outputs.
- The required default bucket is `zeroshot_learning_output`, and the bucket MUST remain
  configurable. **Planning verified that the storage server rejects this name**
  (`InvalidBucketName`; S3 naming rules forbid underscores). The default is therefore
  `zeroshot-learning-output` under ADR-0002, **accepted by the project owner** on
  2026-11-11, as allowed by constitution Principle VI (v1.1.0).
- The bucket MUST be created at startup or before the first write if it does not exist.
- Each successful classification MUST be stored as one JSON record at
  `predictions/YYYY/MM/DD/{request_id}.json`, with the date in UTC and a unique request ID
  (UUIDv4).
- Records MUST conform to the Prediction Record contract (Key Entities) and use a documented
  timestamp format.
- A storage failure MUST be logged and reported to the user; classification completion and
  storage completion MUST be distinguishable, and a failed write MUST NOT be silently
  discarded.

**FR-10 - Logging and operational feedback**
- Structured logs MUST cover: startup and configuration validation; model download start,
  success and failure; model deletion and cache cleanup; model loading; classification start
  and completion; bucket checks and record writes; and errors from external services.
- Logs MUST include the request ID where relevant and MUST NOT contain credentials, tokens or
  the full submitted text.

**FR-11 - Local infrastructure and developer commands**
- The repository MUST provide: Docker Compose configuration for a local storage service with
  persistent data and console access; a dependency manifest with a committed lock file; an
  `.env.example` containing placeholders only; ignore rules for credentials, models and
  caches; a CPU-only application container image containing no secrets, `.env` files or
  model files (models and cache supplied through mounted volumes); documented developer
  commands (plain `uv run …` / `docker compose …` commands documented in the README); and CI
  configuration.
- The README MUST document: starting, stopping and restarting the storage service and the
  application; opening the storage console; downloading models; classifying text; inspecting
  stored records; model caching, deletion and cache cleanup; removing stored predictions;
  running tests; and troubleshooting.

**FR-12 - Prediction workflow orchestration**
- A single prediction workflow MUST handle each classification request, keeping the UI,
  classifier and storage decoupled. For each request it MUST:
  1. Validate the input text and the selected label set.
  2. Verify that the selected model is ready and mark it as in use for the duration of the
     classification, releasing it afterwards, including on failure.
  3. Run the classification.
  4. Build the prediction record (request ID, UTC timestamp, application version, label-set
     key, model name, identifier and revision, effective settings, latency).
  5. Validate the record against the Prediction Record contract.
  6. Store the record.
- It MUST return exactly one of three outcomes:
  - **Failed**: validation, model or classification failure; nothing is stored.
  - **Computed but not persisted**: classification succeeded, storage failed; the UI says so
    explicitly.
  - **Fully successful**: classification and storage both succeeded.
- The request ID MUST appear in the logs and in the stored record.

### Non-Functional Requirements

- **NFR-01 Reproducibility**: the runtime and dependency versions MUST be pinned or locked
  and the installation procedure documented; model identifiers, immutable revisions and
  effective settings MUST be recorded; the application MUST be runnable from a clean checkout
  using the README.
- **NFR-02 Separation of concerns**: presentation, configuration, model catalogue and
  lifecycle, model loading and classification, workflow orchestration, persistence and
  logging MUST be separate; core services MUST be testable without launching the UI.
- **NFR-03 Reliability and data integrity**: failed downloads MUST NOT be ready; deletion
  MUST be restricted to managed paths; active classification MUST be protected from
  concurrent deletion or cache cleanup; request IDs MUST be unique; stored records MUST be
  contract-valid and complete; storage failures MUST be visible and distinct from
  classification failures.
- **NFR-04 Security and privacy**:
  - Credentials MUST come only from the environment and MUST NOT be committed or appear in
    logs, user-facing errors or container images.
  - Input MUST be validated before classification.
  - Paths derived from model identifiers or labels MUST resolve inside the managed
    directories.
  - Logs MUST NOT contain the full input text; they SHOULD record the request ID and the
    input length or a non-reversible hash.
  - User-facing errors MUST NOT expose stack traces or secrets.
  - Development storage credentials MUST be labeled local-only.
  - The README MUST warn that the administration page is unauthenticated and for local or
    trusted use only, and MUST explain that stored records contain submitted text, where
    they are stored, and how to remove them (console, CLI client, or deleting the local data
    volume).
- **NFR-05 Resource management**: no more large models than necessary SHOULD be held in
  memory; loaded models SHOULD be reused when safe; cleanup MUST account for active
  classification; tests MUST NOT download large models.
- **NFR-06 Maintainability and extensibility**: compatible models SHOULD be addable, and
  label sets maintainable, without code changes; serialization MUST be isolated from UI
  code; changes to the record contract MUST be documented and tested.
- **NFR-07 Observability**: lifecycle and classification events MUST be logged with useful
  metadata and request correlation; latency SHOULD be measured and stored. A metrics
  endpoint, experiment tracker or external monitoring stack is not required in v1.

### Testing Requirements

- **Unit tests** (no network, no large downloads; mocks or lightweight fixtures):
  - Configuration: valid and invalid settings, including the maximum input length.
  - Catalogue: revision, licence and optional default settings.
  - Label sets: parsing and validation.
  - Model management: missing-model detection; reuse at the configured revision; safe path
    resolution, including malicious identifiers; marker written only after validation;
    failed downloads never ready; independent multi-model outcomes; deletion restricted to
    managed paths, blocked while in use, and releasing the loaded model; cache cleanup with
    a missing directory and with errors.
  - Classifier: all validation rejections; a score for every label; descending order with
    deterministic ties; predicted class and confidence matching the top entry; revision and
    settings included; reuse of loaded models.
  - Prediction workflow: the three outcomes; no record on classification failure; a storage
    failure never reported as full success; the in-use mark released on failure; request ID
    and label-set key present in the record, and request ID present in the logs.
  - Storage: bucket created when missing and reused when present; valid JSON; required
    fields enforced before writing; correct UTC keys; unique request IDs; failures
    propagated.
- **Integration tests** (real storage service): bucket creation, including acceptance of
  the default bucket name; record persistence; key format; required fields and types;
  retrieval; and clear failure when the service is unavailable.
- **End-to-end verification** (automated or a documented manual check): start the storage
  service and the application, make a model available, submit a valid request, then verify
  the displayed prediction, the matching stored record, and its model and settings metadata.
  Routine CI SHOULD use a deterministic lightweight classifier stand-in instead of large
  downloads; real model loading MAY be verified manually.
- **CI**: install from the lock file; run format and lint checks; run unit tests; provision
  the storage service; run integration tests; check repository hygiene (no credentials,
  `.env` files, model artifacts or caches committed).

### Key Entities *(include if feature involves data)*

- **Model Definition**: a configured, supported model. Attributes: stable identifier,
  display name, Hugging Face repository identifier, task, immutable revision, licence
  (optional), and optional default settings (multi-label behavior, hypothesis template).
- **Model Installation**: the local state of a Model Definition. Attributes: status
  (not available / downloading / ready / failed / incomplete), local location, and a
  completion marker (model identifier, resolved revision, download timestamp). It is ready
  only when a valid marker exists. "Incomplete" means files exist at the install location
  without a valid marker; it is never usable.
- **Label Set**: a reusable set of candidate categories. Attributes: stable key, display
  name, and an ordered list of at least two distinct, non-empty labels.
- **Classification Request**: one user submission. Attributes: request ID, input text,
  selected model, selected label set, effective settings. Produces exactly one outcome:
  failed, computed but not persisted, or fully successful.
- **Prediction Record**: the stored result of a classification, located at
  `predictions/YYYY/MM/DD/{request_id}.json` (UTC date of `timestamp`).

  | Field | Type | Required | Description |
  |-------|------|----------|-------------|
  | `schema_version` | string | Yes | Version of the record contract. |
  | `request_id` | string | Yes | Unique identifier of the request (UUIDv4). |
  | `timestamp` | string | Yes | UTC time in ISO 8601 format. |
  | `input_text` | string | Yes | Submitted text. |
  | `predicted_class` | string | Yes | Highest-scoring candidate label. |
  | `confidence` | number | Yes | Score of `predicted_class`. |
  | `candidate_labels` | array of strings | Yes | Labels supplied to the classifier. |
  | `ranked_predictions` | array of `{label, score}` | Yes | All labels with scores, descending. |
  | `model_name` | string | Yes | Human-readable configured model name. |
  | `model` | string | Yes | Hugging Face repository identifier. |
  | `model_revision` | string | Yes | Immutable revision used. |
  | `inference_parameters` | object | Yes | Effective settings, incl. multi-label, hypothesis template and compute device. |
  | `label_set` | string | Yes | Key of the saved label set used. |
  | `application_version` | string | Yes | Version of the application. |
  | `inference_latency_ms` | number or null | No | Measured classification duration. |
  | `session_id` | string or null | No | Non-identifying session correlation value. |

  Invariants: `confidence` equals the score of `predicted_class`; every score is a finite
  number; `ranked_predictions` has exactly one entry per candidate label. The brief's
  example JSON (brief §3.5) shows the intended shape. This contract adds `schema_version`,
  `model_name`, `inference_parameters` and `label_set`. It also requires `model_revision`
  to be an immutable commit SHA rather than the brief's illustrative `"main"` (constitution
  Principle III). The normative definition is `contracts/prediction-record.schema.json`.

## Success Criteria *(mandatory)*

### Measurable Outcomes

**Primary acceptance criteria** (all mandatory; a superset of the brief's §11 list, see the
mapping below):

| ID | Criterion | Verification |
|----|-----------|--------------|
| AC-01 | The application starts using documented commands from a clean checkout. | Manual / CI smoke check |
| AC-02 | The storage service starts locally through Docker Compose and keeps data across container restarts. | Integration / manual |
| AC-03 | The application creates the configured default bucket automatically (`zeroshot-learning-output` per ADR-0002, accepted by the project owner; the required `zeroshot_learning_output` is rejected by the server). | Integration test |
| AC-04 | The administration page shows at least the two required models. | UI verification |
| AC-05 | The administrator can download the DeBERTa zero-shot model. | Manual |
| AC-06 | The administrator can download the second model, alone or together with another in one multi-select action. | Manual |
| AC-07 | A model already downloaded at the configured revision is not downloaded again. | Unit test |
| AC-08 | The administrator can safely delete a managed downloaded model. | Unit test / manual |
| AC-09 | The administrator can safely request model-cache cleanup. | Unit test / manual |
| AC-10 | The administrator can create and maintain saved label sets. | Unit test / UI verification |
| AC-11 | The classification page accepts text and allows selection of a ready model and a valid label set. | UI verification |
| AC-12 | The application returns a predicted class and confidence score. | Classifier test / manual |
| AC-13 | The application displays all candidate labels ranked by score. | Classifier test / UI verification |
| AC-14 | Every successful classification is stored as JSON in the storage service. | Integration test |
| AC-15 | Stored records contain all required fields and traceability metadata. | Integration test |
| AC-16 | Invalid input and invalid label sets produce clear errors. | Unit test / UI verification |
| AC-17 | Missing or unavailable models are handled clearly. | Unit test / UI verification |
| AC-18 | Storage connection and write errors are handled clearly and never reported as full success. | Integration test / UI verification |
| AC-19 | All unit tests pass. | CI |
| AC-20 | Integration tests pass against a real storage service. | CI |
| AC-21 | README setup, usage, model lifecycle, record inspection, testing and troubleshooting instructions are complete. | Documentation review |
| AC-22 | No secrets, real `.env` files, model artifacts or caches are tracked in version control. | CI / repository review |
| AC-23 | The repository contains the constitution, specification, plan, data model and task breakdown. | Repository review |
| AC-24 | The plan contains a Constitution Check and every mandatory requirement maps to tasks and verification. | Spec review |

**Assignment acceptance-criteria mapping** (brief §11, criteria 1–22, mapped to this spec):

| Brief # | Brief criterion (short) | This spec | Note |
|---------|-------------------------|-----------|------|
| 1 | App starts with documented commands | AC-01 | |
| 2 | MinIO starts via Docker Compose | AC-02 | This spec also requires data to survive restarts |
| 3 | Bucket `zeroshot_learning_output` auto-created | AC-03 | **Deviation**: the server rejects the name, so `zeroshot-learning-output` is used (ADR-0002, accepted by the project owner) |
| 4 | At least two configured models visible | AC-04 | |
| 5 | Download DeBERTa | AC-05 | |
| 6 | Download second model | AC-06 | |
| 7 | No unnecessary re-download | AC-07 | |
| 8 | Delete a local model | AC-08 | |
| 9 | Clear the HF cache | AC-09 | |
| 10 | User enters text and selects candidate labels | AC-11 (+ AC-10) | Labels are chosen as a saved label set (brief §3.4 "Select a candidate-label set") |
| 11 | Predicted class and confidence | AC-12 | |
| 12 | Ranked predictions shown | AC-13 | |
| 13 | Every success saved as JSON in MinIO | AC-14 | |
| 14 | Required metadata stored | AC-15 | |
| 15 | Invalid input handled clearly | AC-16 | |
| 16 | Missing models handled clearly | AC-17 | |
| 17 | MinIO connection errors handled clearly | AC-18 | |
| 18 | Unit tests pass | AC-19 | |
| 19 | Integration tests pass with MinIO running | AC-20 | |
| 20 | README with setup, usage, testing, troubleshooting | AC-21 | README contents follow the brief's "README FILE" list (task T069) |
| 21 | No secrets or model files committed | AC-22 | |
| 22 | Constitution, spec, plan, data model, tasks in repo | AC-23 | The spec file is `spec.md` (Spec Kit name) |
| — | (no brief equivalent) | AC-10, AC-24 | Added: label-set maintenance; Constitution Check and traceability |

**Quantitative outcomes**:

- **SC-001**: 100% of classifications reported as fully successful have exactly one matching
  stored record containing every required field.
- **SC-002**: 0 models are shown as ready after a failed or interrupted download, across all
  failure cases in the Edge Cases table.
- **SC-003**: 100% of invalid submissions listed in the Edge Cases table are rejected before
  classification, each with a message that names the problem.
- **SC-004**: an evaluator following only the README can install the project, start the
  services and run the full test suite in under 30 minutes, excluding model download time.
- **SC-005**: the default unit-test suite completes with network access disabled and
  downloads 0 bytes of model files.
- **SC-006**: once a model is loaded, a classification of text up to the maximum length with
  up to 10 labels displays its result within 10 seconds on the reference setup, using the
  CPU compute device.
- **SC-007**: every stored record can be traced back to one model revision, one label set,
  and one set of settings with no information outside the record itself.
- **SC-008**: 0 files outside the managed model and cache directories are removed by any
  deletion or cleanup operation, including when given malicious identifiers.

## Assumptions

**Mandated constraints** (constitution Principle I and the assignment):

- The platform uses Hugging Face zero-shot models with the PyTorch backend, a two-page web
  UI, and MinIO as the object store and system of record.
- It is designed for local or trusted use only, not for an internet-exposed, multi-user
  deployment.

**Scope (v1)**:

- Out of scope: model training, fine-tuning or evaluation pipelines; non-zero-shot tasks;
  model providers other than Hugging Face; storage providers other than MinIO; cloud
  deployment, high availability or GPU optimization; authentication, authorization or
  tenancy; mandatory experiment tracking or monitoring services; prediction-history
  browsing, CSV export, batch classification, model comparison, and storage lifecycle
  policies.
- Optional assignment enhancements are considered only after all mandatory acceptance
  criteria are met, and must then be added consistently to the spec, plan, data model,
  tasks, tests and acceptance criteria.
- Two items on the brief's optional list (§8) are **in v1 scope by decision of the
  constitution**, not as late extras: the Docker image for the Streamlit application
  (constitution "Scope Boundaries → In Scope (v1)") and prediction-latency measurement
  (the `inference_latency_ms` record field; constitution Principles VI and IX). All other
  optional items are out of scope.

**Defaults chosen in this spec**:

- A deletion requested while the model is in use is **refused**, not deferred, because it is
  simpler and the brief only requires deletion to be an explicit, safe administrator action.
- At least one default label set (key `default`) ships with the application, so a fresh
  install can classify immediately after a model download.
- Label duplicates are detected after trimming and case-insensitive comparison, consistent
  with constitution Principle VIII.
- SC-006's reference setup is the development machine recorded in the plan, with
  classification forced onto the CPU. Optional hardware acceleration may give slightly
  different scores; this platform-dependent behavior is documented (constitution Principle
  III), and the device used is recorded with every prediction.

**To verify during planning** (all resolved in `research.md`):

- Both initial model repositories are accessible and compatible with the chosen library
  versions.
- Their revisions can be resolved to immutable commit identifiers.
- The development environment has enough disk and memory for at least one model.
- The chosen storage-server image and client versions are compatible and available.
- ~~The default bucket name `zeroshot_learning_output` is accepted by the chosen storage
  server and client.~~ **Verified false on 2026-11-11**: the pinned MinIO server rejects it
  with `InvalidBucketName`. The deviation is recorded in
  `docs/adr/0002-bucket-name-hyphens.md` (status: Accepted by the project owner). The
  name stays configurable.
- Each model's default hypothesis template and multi-label setting match its model card.

**Risks**:

- Model downloads are large and sensitive to network interruptions.
- Models differ in memory needs and loading time.
- The managed model directory and the model cache can be confused unless documented.
- Removing model or cache files during classification can cause failures unless lifecycle
  operations are coordinated.
- A classification is only fully successful when storage is available.
- Stored records contain submitted text, which has privacy implications; the
  local/trusted-use limitation must be explicit.

**Definition of done**: all mandatory acceptance criteria are met; the application is
reproducible from a clean checkout; responsibilities are separated; successful
classifications are reliably stored as traceable records; unit and integration tests pass
locally and in CI; the constitution, spec, plan, data model and tasks are in the repository;
`/speckit-analyze` has run and its findings are resolved or documented; and no secrets,
model artifacts or caches are committed.
