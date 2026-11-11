<!--
Sync Impact Report
==================
Version change: 1.1.0 → 1.1.1 (2026-11-11)
Bump rationale: PATCH. In "Scope Boundaries → In Scope (v1)", "Makefile" becomes
"documented developer commands". The assignment brief does not require a Makefile (it
appears only in the brief's *recommended* structure), and spec FR-11 already accepted
equivalent documented commands. No principle or requirement changes.

Modified sections:
  - Scope Boundaries → In Scope (v1): Makefile → documented developer commands
    (`uv run …` / `docker compose …` in the README).

Modified principles: none in 1.1.1. In 1.1.0:
  - VI. MinIO Is the System of Record. A different default bucket, used when the required
    name is rejected, now needs an ADR with the verification evidence, approved by the
    instructor OR the project owner (previously: the instructor only). The replacement must
    stay as close as possible to the required name, and the README must state the deviation.

Added sections: none
Removed sections: none

Templates and artifacts:
  - ✅ .specify/templates/*: reviewed, no change needed (no template references this rule)
  - ✅ docs/adr/0002-bucket-name-hyphens.md: approval by the project owner now satisfies VI
  - ✅ specs/001-zero-shot-platform/plan.md: Constitution Check result updated to PASS
  - ✅ specs/001-zero-shot-platform/spec.md, research.md, tasks.md: references updated
  - ✅ README.md: exists; Deviations section states the bucket change (required by VI)
  - ✅ docs/adr/0001-hf-home-instead-of-transformers-cache.md: unaffected

History:
  - 1.0.0 (2026-11-06): first ratified version, written by the project owner.
  - 1.1.0 (2026-11-11): Principle VI owner-approved bucket exception (MINOR).

Follow-up TODOs: none.
-->
# Zero-Shot Text Classification MLOps Platform Constitution

## Project Purpose

Build a reproducible, Python-based zero-shot text-classification
platform using PyTorch, Hugging Face Transformers, Streamlit, and MinIO,
for learning and demonstrating MLOps practices. An administrator manages
models and candidate-label sets; a user classifies text; every
successful classification is persisted as structured JSON in MinIO.

This constitution defines the non-negotiable engineering principles and
quality standards for the project. The specification, implementation
plan, data model, tasks, and code MUST comply with it. Where they
conflict, this document takes precedence.

The constitution defines **what** must remain true. Implementation
details and technical choices belong in `plan.md`, unless explicitly
constrained here.

## Conventions

The keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY are to be
interpreted as normative requirements, consistent with RFC 2119.

## Core Principles

### I. Fixed Technology Stack

-   Application code MUST be written in Python. Other languages are
    permitted only for supporting configuration, infrastructure,
    deployment, or automation files, such as YAML, JSON, TOML,
    Dockerfiles, Docker Compose files, Makefiles, shell scripts, and CI
    workflows.
-   PyTorch MUST be the deep-learning framework used for model loading
    and inference.
-   Models MUST be loaded through Hugging Face Transformers, using its
    zero-shot classification pipeline or an equivalent Transformers
    implementation backed by PyTorch. The application MUST NOT silently
    fall back to TensorFlow, JAX, ONNX Runtime, or another inference
    backend.
-   Streamlit MUST be used for the user interface.
-   MinIO MUST be used as the object-storage system and system of record
    for classification outputs.
-   The project MUST use the Technology Baseline defined in this
    constitution. Adding, replacing, or materially changing a stack
    component requires a documented Architecture Decision Record (ADR).

*Verified by*: dependency and configuration review; CI checks;
Constitution Check in `plan.md`.

### II. Separation of Concerns

The UI, model management, inference, storage, configuration, and logging
responsibilities MUST remain separate.

-   **UI (Streamlit pages)**: rendering, navigation, and user input
    only. UI code MUST NOT call Hugging Face, Transformers, filesystem
    model-management, or MinIO APIs directly.
-   **Model manager**: model configuration, download, local
    availability, revision tracking, deletion, and Hugging Face cache
    cleanup. It MUST NOT run inference or write predictions.
-   **Classifier**: input validation, pipeline initialization and
    caching, inference, and prediction ranking. It MUST NOT download
    models or access MinIO.
-   **Storage**: bucket creation, prediction schema validation,
    serialization, and persistence.
-   **Configuration**: loading and validating application settings and
    model/label configuration.
-   **Logging**: shared structured logging configuration and helpers; it
    MUST NOT contain business logic.
-   **Prediction service (orchestration)**: the single component that
    composes input validation, the classifier, and the storage service
    into one classification workflow and reports distinct outcomes
    (failed, computed but not persisted, fully successful). The UI MUST
    call this service rather than composing classifier and storage
    itself, so that the classifier and storage remain decoupled.
-   **Active-use coordination**: protection of a model against deletion
    or cache cleanup during inference MUST be implemented through a
    small shared mechanism (for example a usage registry or lease)
    defined in `plan.md`, not through a direct dependency between the
    model manager and the classifier.

Core services MUST be importable and testable without starting
Streamlit.

*Verified by*: unit tests that import services without Streamlit; code
review against the module boundaries defined in `plan.md`.

### III. Reproducibility and Traceability

-   The Python version, dependencies, model identifiers, resolved model
    revisions, inference parameters, and MinIO configuration MUST be
    pinned or documented sufficiently for another developer to reproduce
    the application from the repository and README.
-   Exact dependency pins MUST be recorded in `uv.lock`;
    `pyproject.toml` MUST declare the supported dependency constraints.
    A `requirements.txt` MAY be exported from `uv.lock` (for example
    with `uv export`) for compatibility with the assignment's
    recommended structure; `uv.lock` remains authoritative.
-   The resolved model revision (preferably an immutable commit hash)
    MUST be recorded when a model is downloaded and MUST be included in
    every prediction record.
-   A mutable model reference such as `main` MAY be used to discover a
    revision, but it MUST be resolved to an immutable revision before
    the downloaded model is considered ready.
-   Inference parameters that can affect results MUST be explicit,
    configured, and documented. This includes, where applicable,
    `multi_label` and `hypothesis_template`.
-   Every prediction record MUST contain enough information to identify
    the model revision, candidate labels, and inference configuration
    used for that prediction.
-   Changes to model definitions or inference configuration MUST be
    traceable through version control and/or documented configuration.

Reproducibility does not imply that every hardware environment will
produce bit-for-bit identical results. Any known platform-dependent
behavior or limitation MUST be documented.

*Verified by*: `uv sync --locked` succeeds on a clean checkout; model
revision and inference configuration appear in stored prediction
records; README reproduction walkthrough succeeds in a clean
environment.

### IV. Configuration and Secrets

-   Environment-specific settings MUST come from environment variables,
    optionally loaded from a local `.env` file for development.
-   Model and candidate-label definitions MUST be configuration-driven.
    Adding a supported zero-shot model SHOULD require only a
    configuration change, provided it is compatible with the established
    model interface.
-   Hugging Face cache configuration MUST use `HF_HOME`. The deprecated
    `TRANSFORMERS_CACHE` variable MUST NOT be relied upon. The
    assignment's example environment lists `TRANSFORMERS_CACHE`;
    omitting it is a deliberate deviation, recorded in
    `docs/adr/0001-hf-home-instead-of-transformers-cache.md` and
    explained in the README. `.env.example` MAY mention it only as a
    commented-out deprecated variable.
-   `.env.example` MUST contain placeholders or safe local-development
    defaults only.
-   Secrets, `.env` files containing real credentials, downloaded
    models, and cache directories MUST be git-ignored and MUST NOT be
    committed or baked into application images.
-   Configuration MUST be validated at application startup or when the
    relevant feature is invoked. Missing or invalid configuration MUST
    produce actionable error messages without exposing secrets.

*Verified by*: repository hygiene checks in CI; review of `.gitignore`,
`.env.example`, and configuration validation.

### V. Model Lifecycle Management

-   Configured models MUST be downloaded from Hugging Face into a
    controlled application directory under `MODEL_DIR`, or managed
    through a documented Hugging Face cache strategy.
-   The administrator MUST be able to select one or more configured
    models and download them in a single action. Each model's outcome
    MUST be tracked and reported independently, and the failure of one
    download MUST NOT affect the others.
-   The relationship between `MODEL_DIR` and `HF_HOME` MUST be explicit
    in `plan.md` and README. The implementation SHOULD avoid maintaining
    unnecessary duplicate copies of the same model.
-   Filesystem paths derived from model identifiers MUST be normalized
    and validated. They MUST NOT resolve outside the application-managed
    model/cache directories.
-   A model that is already complete locally at the required revision
    MUST NOT be downloaded again unnecessarily.
-   Downloads MUST be atomic from the application's perspective: a model
    MUST NOT be marked ready until all required files have been
    downloaded and validated and a completion marker has been written.
    Validation MUST NOT require running inference; the exact checks (for
    example required files present and model configuration loadable)
    are defined in `plan.md`.
-   The completion marker MUST identify at least the model id, resolved
    revision, and download timestamp.
-   Failed or interrupted downloads MUST NOT be treated as ready models.
    Temporary or partial download state MUST be cleaned up safely or
    clearly identified for recovery.
-   Each configured model MUST declare its licence when licence
    information is available. The application MUST NOT imply that
    downloading a model grants permission to use it for every purpose.
-   Model deletion and Hugging Face cache cleanup MUST be explicit
    administrator actions.
-   Deletion MUST be restricted to application-managed paths. Resolved
    paths MUST be checked before deletion to prevent deleting files
    outside those paths.
-   A model MUST NOT be deleted while an inference operation is actively
    using it. The implementation MUST define and document how active use
    is detected or prevented.
-   Deleting a model MUST invalidate or evict its cached inference
    pipeline.
-   Initialized pipelines SHOULD be cached to avoid repeated model
    initialization. Large models MUST NOT be held in memory
    unnecessarily.
-   Cache cleanup MUST handle the case where the cache directory does
    not exist.

*Verified by*: model-manager tests covering missing and existing models,
revision handling, failed downloads, path safety, deletion, active-use
protection, pipeline eviction, and cache cleanup.

### VI. MinIO Is the System of Record

-   Every successful classification MUST be persisted as one structured
    JSON object in the bucket configured by `MINIO_BUCKET`. The default
    bucket MUST be `zeroshot_learning_output`, subject to the exception
    below.
-   The bucket MUST be created automatically when it does not exist.
-   The bucket name MUST remain configurable through `MINIO_BUCKET`.
    Because some S3-compatible servers restrict bucket names (for
    example, underscores), acceptance of the default name by the
    selected MinIO server image and SDK MUST be verified before
    implementation. If it is rejected, a different default MAY be used
    only when all of the following hold:
    -   it is recorded in an ADR that includes the verification
        evidence, such as the server or SDK error;
    -   the ADR is approved by the instructor **or** by the project
        owner, and the approver and date are recorded in the ADR;
    -   the replacement stays as close as possible to the required name
        (for example, each `_` replaced by `-`);
    -   the deviation is stated prominently in the README.
-   Object keys MUST follow `predictions/YYYY/MM/DD/{request_id}.json`,
    using the UTC date.
-   Each prediction record MUST contain:
    -   `schema_version` (version of the persisted prediction schema)
    -   `request_id` (UUID4)
    -   `timestamp` (ISO-8601 UTC)
    -   `input_text`
    -   `predicted_class`
    -   `confidence`
    -   `candidate_labels`
    -   `ranked_predictions` (label and score, sorted in descending
        score order)
    -   `model` (Hugging Face model identifier)
    -   `model_name`
    -   `model_revision` (resolved immutable revision when available)
    -   `application_version`
    -   `inference_parameters` (the effective parameters that can affect
        the prediction)
    -   `label_set` (key of the saved candidate-label set used; the
        classification page uses saved label sets only in v1)
-   The record SHOULD also contain the following fields when available:
    -   `session_id`
    -   `inference_latency_ms`
-   The prediction schema MUST be defined in one place and validated
    before writing.
-   Timestamps MUST be stored in UTC using ISO-8601 format. Request
    identifiers MUST be unique.
-   Prediction scores MUST be finite numeric values within the expected
    probability-score range returned by the selected classification
    implementation. The implementation MUST document the score semantics
    provided by that implementation.
-   Local files MAY be used for temporary processing, caching, or tests,
    but MUST NOT replace MinIO as the system of record. No alternative
    object-storage backend may be introduced without an ADR.
-   A prediction MUST be considered fully successful only when inference
    and MinIO persistence have both succeeded.
-   If inference succeeds but the MinIO write fails, the UI MUST clearly
    state that a prediction was produced but was **not persisted**. It
    MUST NOT report the operation as a fully successful classification.
-   The application MUST NOT silently discard a MinIO persistence
    failure or claim that the result was saved when it was not.

*Verified by*: storage unit tests and an end-to-end integration test
that verifies the prediction object, schema, metadata, and required
object key in MinIO.

### VII. Testing Discipline and Requirement Traceability

-   Core services MUST have `pytest` unit tests that run without network
    access or large model downloads. Tests SHOULD use mocks, fakes, or
    lightweight fixtures where appropriate.
-   MinIO behavior MUST be covered by integration tests against a real
    MinIO instance. Integration tests MAY be skipped locally when MinIO
    is unavailable, but the CI pipeline MUST start MinIO and execute
    them.
-   At least one end-to-end test MUST verify that a prediction is
    written to MinIO with all required fields and the required
    date-based object key.
-   Every mandatory functional requirement and acceptance criterion MUST
    be traceable to one or more tests or to an explicitly documented
    manual verification step when automation is not reasonably possible.
-   Tests that require large model downloads MUST use mocks or
    lightweight fixtures in the normal test suite. Any optional
    real-model smoke test MUST be clearly separated from the default
    test suite.
-   Tests MUST cover relevant failure paths, not only successful
    execution.

*Verified by*: Quality Gates; requirement-to-test traceability in
`tasks.md` or `plan.md`.

### VIII. Explicit Error Handling

-   Invalid input MUST be rejected before inference. At minimum, this
    includes:
    -   Empty or whitespace-only text.
    -   Fewer than two candidate labels.
    -   Empty candidate labels.
    -   Duplicate labels after trimming and case-insensitive
        normalization.
    -   Text exceeding the maximum length configured by
        `MAX_INPUT_CHARS` and documented in the specification and
        README.
-   Not-ready models, failed downloads, missing or invalid label sets,
    invalid configuration, and MinIO failures MUST each produce
    distinct, user-readable messages.
-   User-facing errors MUST NOT expose credentials, tokens, raw stack
    traces, or unnecessary internal details.
-   Core services MUST raise typed domain exceptions for expected
    operational failures. The UI MUST translate these exceptions into
    appropriate user-facing messages.
-   An inference failure MUST NOT create a successful prediction record.
-   A persistence failure MUST be distinguishable from an inference
    failure.

*Verified by*: unit tests for each documented error case and
demonstration of the relevant acceptance criteria.

### IX. Observability

-   Application logs MUST be structured and MUST use the shared logging
    utility.
-   Important operations MUST be logged, including model download,
    deletion, cache cleanup, pipeline loading, inference, bucket
    creation, and MinIO writes.
-   Logs SHOULD include operation name, model id, resolved revision,
    request id, duration, and outcome where applicable.
-   The prediction `request_id` MUST appear in both application logs and
    the corresponding MinIO record.
-   Logs MUST make it possible to distinguish inference success from
    persistence success.
-   The application SHOULD measure inference latency and model-loading
    duration, even if these measurements are initially displayed only in
    logs or the prediction record.
-   The architecture SHOULD allow future integration with external
    metrics and monitoring systems without requiring changes to the core
    classification contract.

*Verified by*: tests or focused manual checks for key log events;
inspection of logs during the final demonstration.

### X. Security and Privacy

-   Credentials, tokens, and secrets MUST NOT appear in logs, source
    control, application images, or user-facing error messages.
-   Logs MUST NOT contain full input text. They SHOULD log the request
    id and, where useful, input length or a non-reversible hash instead.
-   User input MUST be validated before inference and persistence.
-   Model identifiers, label names, and other user-controlled values
    MUST NOT be used as filesystem paths without validation and safe
    path resolution.
-   The admin page has no authentication in v1. The README MUST clearly
    state that it is intended for local development or trusted
    environments only and MUST NOT be exposed to untrusted users.
-   MinIO credentials MUST be configurable through environment
    variables. Development credentials MUST be clearly identified as
    local-only and MUST NOT be presented as production-safe credentials.
-   The application MUST document that prediction records contain
    user-submitted text and explain where those records are stored and
    how they can be removed from the local MinIO environment.

*Verified by*: path-safety and input-validation tests; repository
hygiene checks; README review.

### XI. Simplicity (YAGNI)

-   The implementation MUST prioritize the mandatory requirements and
    acceptance criteria.
-   Optional features MUST NOT be started before all mandatory
    acceptance criteria are met.
-   Optional features MUST be explicitly selected and scoped before they
    are added to the implementation tasks.
-   Any additional complexity, including an extra service, abstraction
    layer, framework, or infrastructure component, MUST be justified in
    the implementation plan's Complexity Tracking table.
-   The project MUST NOT introduce production-grade infrastructure or
    operational mechanisms unless they are required by the assignment or
    justified by a documented decision.

*Verified by*: Constitution Check and Complexity Tracking in `plan.md`.

### XII. ML Experiment and Inference Traceability

-   Each prediction MUST be traceable to the exact configured model
    revision and effective inference parameters used.
-   Candidate labels and the selected label-set identity, when
    applicable, MUST be recorded with the prediction.
-   Model configuration and inference-parameter changes MUST be tracked
    through version control or documented configuration history.
-   The prediction record MUST provide a stable basis for future offline
    evaluation or experiment tracking.
-   The architecture SHOULD allow future integration with an
    experiment-tracking platform, such as MLflow, without making that
    platform a mandatory dependency in v1.

*Verified by*: inspection of stored prediction records and configuration
history.

### XIII. Data Integrity

-   Data written to MinIO MUST conform to the documented prediction
    schema.
-   Required fields MUST be validated before persistence.
-   Serialization MUST produce valid JSON and MUST fail clearly when
    required data is invalid.
-   The object key date MUST be derived from the record's UTC timestamp.
-   A request id MUST identify one prediction record and MUST NOT be
    reused for a different prediction.
-   The application MUST report persistence errors accurately and MUST
    NOT silently treat an incomplete write as successful.

*Verified by*: schema-validation, serialization, object-key, and MinIO
integration tests.

### XIV. Extensibility

-   New compatible zero-shot models SHOULD be addable through
    configuration without changing the core classification workflow.
-   Model management and inference MUST NOT depend on the identifiers of
    the two initial models.
-   New observability or evaluation capabilities SHOULD be integrable
    without changing the meaning or required fields of existing
    prediction records.
-   Any change to the prediction schema MUST be documented and tested.
    Backward compatibility SHOULD be preserved where practical;
    incompatible changes MUST be explicitly documented.
-   Extensibility MUST NOT justify speculative abstractions or
    infrastructure that are not needed by the current requirements.

*Verified by*: configuration-driven model tests and review of component
boundaries.

### XV. Documentation

-   The README MUST document: prerequisites and supported platforms;
    reproducible setup from the lock file; starting, stopping, and
    restarting MinIO and opening its console; starting Streamlit;
    downloading models; classifying text; verifying the stored JSON in
    the bucket; running unit and integration tests; and
    troubleshooting.
-   Model caching behavior MUST be documented: the relationship between
    `MODEL_DIR` and `HF_HOME`, reuse of existing models, atomic
    downloads and failed-download behavior, and the consequences of
    model deletion and of Hugging Face cache cleanup.
-   Prediction storage MUST be documented: object key layout, the JSON
    schema (or a link to the data model), the fact that records contain
    user-submitted text, and how to remove them from the local MinIO
    environment (console, `mc`, or removing the local data volume).
-   The README MUST state that the admin page has no authentication and
    is intended for local development or trusted environments only.
-   Documentation MUST be updated in the same change as the behavior it
    describes.

*Verified by*: README review against Quality Gate 8; clean-checkout
walkthrough.

## Technology Baseline

The following baseline is the project's declared compatibility target.
Exact resolved versions MUST be recorded in `uv.lock`. The
implementation plan MUST record the versions actually resolved and
tested for the development and CI environments.

  -----------------------------------------------------------------------
  Component               Constraint              Reference version
  ----------------------- ----------------------- -----------------------
  Python                  `>=3.14,<3.15`          3.14.10
                          (`.python-version`)     

  torch                   `>=2.15,<3`             2.15.0

  transformers            `>=5.22,<6`             5.22.2

  huggingface_hub         As required by          2.4.0
                          Transformers; exact     
                          version locked          

  streamlit               `>=1.67,<2`             1.67.0

  minio (Python SDK)      `>=7.2,<8`              7.2.20

  uv                      Pinned in CI and        Project-selected and
                          documented in README    verified version

  pytest                  `>=9.2,<10`             9.2.0

  ruff                    `>=0.17,<0.18`          0.17.4

  MinIO server image      Explicitly pinned image Project-selected and
                          reference; never        verified image
                          `latest`                
  -----------------------------------------------------------------------

The reference versions above MUST be checked for compatibility and
availability before implementation. If a reference version cannot be
installed or tested on a supported platform, the implementation plan
MUST document the discrepancy and propose a compatible version before
development proceeds. The lockfile and CI results are authoritative for
the actual resolved environment.

### Upgrade Policy

-   Within an existing supported major line, dependencies MAY be updated
    by refreshing `uv.lock`, provided all quality gates pass.
-   A new major dependency line, a new Python minor version, or a
    material stack change requires an ADR and review of this
    constitution.
-   A dependency MUST NOT be upgraded solely because a newer version
    exists; compatibility with the project and supported platforms must
    be checked.
-   The project MUST document known dependency or platform limitations
    that affect reproducibility.

## Supported Platforms

-   Development: macOS 14 or later on Apple Silicon (arm64), and Linux
    x86_64, subject to compatibility with the locked dependencies.
-   Docker and CI: Linux x86_64.
-   CPU inference MUST work in the supported Docker/CI environment.
-   GPU acceleration (CUDA or MPS) MAY be used when available, but MUST
    NOT be required for the mandatory application workflow or tests.

## Scope Boundaries

### In Scope (v1)

-   Administrator page for configured models: status, identifier, name,
    revision, download of one or more selected models, refresh, and
    deletion.
-   Display of model directory and Hugging Face cache location.
-   Explicit Hugging Face cache cleanup.
-   Candidate-label management: add, remove, save, and select label sets
    using the documented configuration/storage mechanism.
-   Classification page: text input, ready-model selection, label-set
    selection, predicted class, confidence, ranked scores, model used,
    and timestamp.
-   MinIO persistence of every successful classification, with automatic
    bucket creation.
-   Initial models:
    -   `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`
    -   `facebook/bart-large-mnli`
-   Docker Compose support for MinIO, a Dockerfile for the application
    (CPU-only image containing no secrets, `.env` files, or model
    files), documented developer commands, CI, tests, and README
    documentation.

### Out of Scope (v1)

-   Authentication and multi-tenant access control.
-   Model training, fine-tuning, or dedicated model-evaluation
    pipelines.
-   Non-zero-shot classification tasks or non-Hugging Face model
    sources.
-   GPU-specific optimization.
-   Cloud or production deployment and high availability.
-   Any object-storage backend other than MinIO.
-   Mandatory Prometheus, MLflow, or other external
    monitoring/experiment-tracking services.
-   Optional features listed in the assignment unless explicitly
    selected after the mandatory acceptance criteria are satisfied.

## Quality Gates and Compliance

A change or final submission is complete only when all applicable gates
pass:

1.  **Lint and formatting**: configured formatting and lint checks pass.
2.  **Unit tests**: all unit tests pass without network access or large
    model downloads.
3.  **Integration tests**: MinIO integration tests pass against a real
    MinIO instance.
4.  **End-to-end persistence test**: at least one test verifies that a
    prediction is written to MinIO with all required fields and the
    required object key.
5.  **CI**: CI runs lint/format checks, unit tests, repository-hygiene
    checks, and MinIO integration tests. CI MUST provision MinIO for
    integration tests, for example through Docker Compose or an
    equivalent supported mechanism.
6.  **Repository hygiene**: no downloaded models, caches, real `.env`
    files, or credentials are tracked.
7.  **Reproducibility**: `uv sync --locked` succeeds in the supported CI
    environment; configuration comes from environment variables or
    documented configuration files.
8.  **Documentation** (Principle XV): the README covers setup, starting MinIO and its
    console, starting Streamlit, downloading models, classifying text,
    verifying JSON in the bucket, stopping and restarting services,
    model caching and cleanup, testing, and troubleshooting.
9.  **Acceptance**: the mandatory acceptance criteria from the
    assignment are met and demonstrated.
10. **Traceability**: the specification, plan, data model, and tasks are
    present in `specs/001-zero-shot-platform/`; mandatory requirements
    are mapped to implementation tasks and tests or documented manual
    verification steps.
11. **Constitution Check**: the implementation plan contains a
    Constitution Check mapping each principle to the planned design and
    verification approach. Any justified exception is documented through
    an ADR.

## Governance

-   The specification, plan, data model, and tasks MUST be reviewed
    against this constitution before implementation.
-   Every implementation plan MUST include a Constitution Check mapping
    the principles and quality gates to design decisions and
    verification activities.
-   **Decision records**: an exception to a principle or a material
    change to the technology stack MUST be documented in an ADR under
    `docs/adr/NNNN-<slug>.md`. Each ADR MUST include status, owner,
    date, review date, affected principle, context, decision, rationale,
    benefits, risks, and impact on reproducibility.
-   An exception without an ADR is a constitution violation. ADRs past
    their review date MUST be revisited.
-   **Amendments** are made through the project's Spec Kit constitution
    workflow. Amendments MUST update the Sync Impact Report and trigger
    a review of dependent templates and artifacts.
-   **Versioning** follows semantic versioning:
    -   MAJOR: a principle is removed or redefined.
    -   MINOR: a principle or section is added, or guidance is
        materially expanded.
    -   PATCH: clarification or wording changes that do not alter
        requirements.
-   Before final submission, the project MUST run `/speckit-analyze` and
    resolve or document any identified inconsistencies.
-   Any optional feature selected for implementation MUST be reflected
    in the specification, plan, data model (if applicable), tasks,
    tests, and acceptance criteria.

**Version**: 1.1.1\
**Ratified**: 2026-11-06\
**Last Amended**: 2026-11-11
