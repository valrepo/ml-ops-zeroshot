# Implementation Plan: Zero-Shot Text Classification Platform

**Branch**: `001-zero-shot-platform` | **Date**: 2026-11-11 | **Spec**: [spec.md](spec.md)

**Input**: Feature specification from `/specs/001-zero-shot-platform/spec.md`

## Summary

A local, two-page Streamlit app for zero-shot text classification with two pinned Hugging
Face models (DeBERTa v3 zero-shot v2.0 and BART-large-MNLI) on PyTorch. Every successful
prediction is stored as a schema-validated JSON record in MinIO.

The core is a plain Python package, `zeroshot`, made of decoupled services:
- `ModelManager`: atomic, revision-pinned downloads into `MODEL_DIR`, safe deletion, and
  `HF_HOME` cache cleanup;
- `LabelSetService`: YAML-backed label sets;
- `Classifier`: transformers pipeline with an LRU of loaded models and deterministic ranking;
- `StorageService`: MinIO;
- `PredictionService`: orchestration with three explicit outcomes.

A small `ModelUsageRegistry` coordinates active use without coupling the model manager to the
classifier. Packaging uses a universal `uv.lock` with CPU-only torch on Linux, Docker Compose
for MinIO and the app, and GitHub Actions running lint, unit and real-MinIO integration
tests.

Research confirmed every baseline version. It also found that the server **rejects** the
required bucket name `zeroshot_learning_output`, so the plan uses `zeroshot-learning-output`
under [ADR-0002](../../docs/adr/0002-bucket-name-hyphens.md), accepted by the project owner.

**Schedule**: the brief's submission deadline is **2026-11-14**. Delivery is MVP-first
(tasks.md "Implementation Strategy").

## Architecture

This section answers each item on the brief's §6.3 list. It refines the brief's suggested
architecture (§4) by adding a prediction service between the classification page and
storage, so that the UI never composes the classifier and MinIO itself (constitution
Principle II).

```text
                         ┌──────────────────────────────┐
                         │  Streamlit app (ui/app.py)   │  st.navigation
                         └──────┬────────────────┬──────┘
                 ┌──────────────▼───┐        ┌───▼───────────────────┐
                 │ Administration   │        │ Classification page   │
                 │ page             │        │ text · model · labels │
                 └──┬────────────┬──┘        └───────────┬───────────┘
                    │            │                       │
          ┌─────────▼──────┐ ┌───▼────────────┐ ┌────────▼─────────────┐
          │ ModelManager   │ │ LabelSetService│ │ PredictionService    │
          │ download/cache │ │ YAML label sets│ │ validate→lease→      │
          │ delete/cleanup │ └────────────────┘ │ classify→record→save │
          └───┬────────┬───┘                    └───┬──────────────┬───┘
              │        │   ModelUsageRegistry       │              │
              │        └──── (leases, eviction) ────┤              │
   ┌──────────▼───────┐                  ┌──────────▼───────┐ ┌────▼────────────────┐
   │ Hugging Face Hub │                  │ Classifier       │ │ StorageService      │
   │ → MODEL_DIR      │                  │ transformers     │ │ MinIO bucket,       │
   │   (pinned SHA)   │                  │ pipeline, PyTorch│ │ predictions/Y/M/D/  │
   └──────────────────┘                  └──────────────────┘ └─────────────────────┘
      config.py (Settings, YAML loaders) and logging.py (JSON logs) are used by all
```

| Brief §6.3 item | Where it is defined |
|-----------------|---------------------|
| Application architecture and main components | The diagram above; component contracts in [contracts/services.md](contracts/services.md); module layout under "Source Code" |
| Technology choices | Technical Context; [research R2–R3](research.md#r2-runtime-and-dependency-versions-verified) |
| Streamlit page design | [contracts/ui-pages.md](contracts/ui-pages.md) |
| Model download and caching approach | [research R5–R7](research.md#r5-which-files-to-download): file filtering, `MODEL_DIR` vs `HF_HOME`, atomic staging and completion marker |
| Hugging Face integration | `huggingface_hub.snapshot_download` at a pinned commit SHA ([R4](research.md#r4-model-revisions-licences-and-default-settings-verified)); `transformers.pipeline("zero-shot-classification")` on PyTorch with an explicit device ([R14](research.md#r14-explicit-compute-device)) |
| MinIO integration | `StorageService` ([services.md](contracts/services.md)); bucket rules ([R1](research.md#r1-default-bucket-name-zeroshot_learning_output--rejected-verified), ADR-0002); image and ports (ADR-0003) |
| Configuration strategy | [contracts/configuration.md](contracts/configuration.md): environment variables (`.env`), `configs/models.yaml`, `configs/labels.yaml` |
| Testing strategy | [research R11](research.md#r11-testing-strategy-and-ci); Constitution Check row VII |
| Docker and local-development approach | [research R12](research.md#r12-containers-and-ports): `uv run streamlit run …` (host) or `docker compose --profile app up` (container); MinIO through Compose |
| Deployment approach | **Local or trusted environments only** (spec Assumptions; no authentication). The supported "deployment" is `docker compose --profile app up` on one host: app and MinIO containers, a persistent MinIO volume, and bind-mounted model, cache and data directories. Cloud deployment and high availability are out of scope (constitution Scope Boundaries) |

**Differences from the brief's recommended layout (§5)**. The brief marks its layout as a
recommendation and its whole document as a draft guideline.

| Brief | This plan | Reason |
|-------|-----------|--------|
| `app/` with `pages/1_Admin.py`, `pages/2_Classify_Text.py` | `src/zeroshot/ui/` with `admin_page.py`, `classify_page.py`, registered through `st.navigation` | `src/` layout makes the core importable and testable without Streamlit (Principle II); `st.navigation` replaces the legacy `pages/` auto-discovery |
| `app/services/minio_storage.py` | `src/zeroshot/services/storage.py` (+ `prediction.py`, `label_sets.py`) | Adds the orchestration (FR-12) and label-set services |
| `app/utils/config.py`, `logging_config.py` | `src/zeroshot/config.py`, `logging.py` | Same responsibilities |
| `configs/labels.yaml` | `configs/labels.yaml` (seed) + `data/label_sets.yaml` (runtime, git-ignored) | Admin edits survive restarts without dirtying git ([R9](research.md#r9-label-set-persistence)) |
| `models/{model_id}` | `models/<org>--<name>/<revision>/` | One directory per pinned revision; `/` in repo IDs cannot be used as a path segment safely |
| `tests/test_*.py` (3 files) | `tests/unit`, `tests/integration`, `tests/e2e`, `tests/model` | The three test tiers of Principle VII |
| `specs/.../specification.md` | `specs/.../spec.md` | Spec Kit commands require `spec.md` |
| `requirements.txt` | Exported from `uv.lock` (T004) | `uv.lock` is authoritative (Principle III) |
| `HF_HOME=./.cache/huggingface`, `TRANSFORMERS_CACHE=…` | `HF_HOME=./.cache/huggingface` only | ADR-0001 |
| `image: minio/minio`, ports 9000/9001 | `pgsty/minio:<pinned>`, ports 9010/9011 (configurable) | ADR-0003 |

## Technical Context

**Language/Version**: Python 3.14.10 (`.python-version`; `requires-python = ">=3.14,<3.15"`).

**Primary Dependencies** (all exist on PyPI, checked 2026-11-11; see [research R2](research.md#r2-runtime-and-dependency-versions-verified)):

| Package | Constraint | Resolved in `uv.lock` (2026-11-11) |
|---------|------------|------------------------------------|
| torch | `>=2.15,<3` | 2.15.0 (`+cpu` on Linux) |
| transformers | `>=5.22,<6` | 5.22.2 |
| huggingface_hub | locked | 2.4.0 |
| streamlit | `>=1.67,<2` | 1.67.0 |
| minio | `>=7.2,<8` | 7.2.20 |
| pydantic / pydantic-settings / PyYAML / filelock | `>=2.14,<3` / `>=2.16,<3` / `>=6.0,<7` / `>=4.2,<5` | 2.14.3 / 2.16.2 / 6.0.3 / 4.2.10 |
| pytest / ruff (dev) | `>=9.2,<10` / `>=0.17,<0.18` | 9.2.0 / 0.17.4 |
| jsonschema (dev, schema-sync test) | `>=4,<5` | 4.26.0 |
| uv | `required-version = ">=0.13.7,<0.14"` | 0.13.7 |
| MinIO server | pinned tag | `pgsty/minio:RELEASE.2026-08-04T00-00-00Z` |

Resolved versions in `uv.lock` are authoritative: 79 packages, with Python 3.14.10 for
development (macOS arm64) and CI (Linux x86_64, `torch 2.15.0+cpu`). Every resolved version
equals the Technology Baseline reference version, so there is no discrepancy to document
(Quality Gate 7).

**Storage**: MinIO bucket `zeroshot-learning-output` for prediction records (system of
record); local filesystem for `MODEL_DIR` (models), `HF_HOME` (library cache) and
`data/label_sets.yaml`.

**Testing**: pytest with the markers `integration` (real MinIO, required in CI) and `model`
(real-model smoke test, excluded by default); ruff; JSON Schema conformance tests.

**Target Platform**: development on macOS 14+ arm64 (reference: Apple M1 Pro, 16 GB RAM)
and Linux x86_64; Docker and CI on Linux x86_64, CPU-only. The compute device is explicit:
`INFERENCE_DEVICE` defaults to `cpu`, `mps` is opt-in on Apple Silicon, and it is never
auto-detected. The device is recorded in every prediction ([research R14](research.md#r14-explicit-compute-device)).

**Project Type**: single Python project, a Streamlit web app over a service package.

**Performance Goals**: SC-006: a result within 10 s for text up to 5000 characters with up
to 10 labels, once the model is loaded, on the reference machine with
`INFERENCE_DEVICE=cpu`. Model load time is logged but not counted against this target.
SC-004: clean clone to a passing full test suite in under 30 minutes, excluding model
downloads (timed in T079).

**Constraints**:
- memory: one model loaded at a time by default (`MAX_LOADED_MODELS=1`);
- disk: about 2 GB for both models, after file filtering (R5);
- unit tests run offline;
- host ports configurable, with defaults 9010, 9011 and 8502 (9000, 9001 and 8501 are busy
  on the reference machine);
- the app container contains no secrets or models.

**Scale/Scope**: one local operator, a few concurrent browser sessions, and a single app
process. Records: thousands, with no retention policy in v1.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Pre-research result**: PASS. No principle conflicted with the spec; the bucket-name risk
was carried into research as a verification item.

**Post-design result**: **PASS** (constitution v1.1.1). The default bucket deviation is
handled through Principle VI's exception path: ADR-0002 contains the verification evidence
and was approved by the project owner. Every principle is satisfied by the design below.

| # | Principle | Design decision | Verification |
|---|-----------|-----------------|--------------|
| I | Fixed Technology Stack | Python only. The transformers 5.x `zero-shot-classification` pipeline is PyTorch-only, so no fallback backend is possible. ONNX, Flax and Rust weights are excluded from downloads (R5). Streamlit UI; MinIO SDK. No stack additions; supporting libraries are listed in R2 | `pyproject.toml` review; a unit test asserts the pipeline is built with a torch model; CI |
| II | Separation of Concerns | `ui/` → services only. `ModelManager`, `Classifier`, `StorageService`, `LabelSetService`, `PredictionService`, `config`, `logging` are separate modules. Active use is coordinated through `core/usage.py::ModelUsageRegistry`, not through direct imports between the model manager and the classifier ([services.md](contracts/services.md)) | An import-boundary unit test (AST scan: `ui` may not import `transformers`/`huggingface_hub`/`minio`; `classifier` may not import `model_manager` or `storage`; core may not import `streamlit`); services are importable without Streamlit |
| III | Reproducibility and Traceability | `uv.lock` (universal, CPU torch on Linux via `[tool.uv.sources]`); `.python-version` 3.14.10; 40-character SHA revisions in `models.yaml`, with branch names rejected by the validator; the marker records the resolved revision; records carry the revision and settings, including the explicit compute device (`INFERENCE_DEVICE`, default `cpu`; R14). The README documents platform-dependent behavior (CPU vs MPS score differences; single-process active-use protection) | `uv sync --locked` in CI; catalogue validator unit tests; E2E record check (`inference_parameters.device == "cpu"`); README review |
| IV | Configuration and Secrets | `Settings` (pydantic-settings) from env and `.env`; `HF_HOME` only ([ADR-0001](../../docs/adr/0001-hf-home-instead-of-transformers-cache.md)); `.env.example` with placeholders; `.gitignore` and `.dockerignore` exclude `.env`, `models/`, `.cache/huggingface/`, `data/`; validation at startup with secret-free messages | Config unit tests; CI hygiene script; image inspection (`docker run --rm app ls -a /app`) |
| V | Model Lifecycle Management | Staging directory, validation without inference, `.complete.json`, then atomic `os.replace` (R7); sequential multi-model download with per-model outcomes; path resolution checked with `is_relative_to(MODEL_DIR)`; deletion refused while leased; pipeline eviction through `on_removed`; cleanup only of `HF_HOME/{hub,xet}`, and a missing directory is a no-op | Model-manager unit tests listed in spec "Testing Requirements" (including malicious `repo_id` such as `../../etc`) |
| VI | MinIO Is the System of Record | One JSON object per success at `predictions/YYYY/MM/DD/{uuid4}.json` (UTC from the record timestamp); one schema ([prediction-record.schema.json](contracts/prediction-record.schema.json)) mirrored by pydantic; bucket ensured at startup and before writing; three outcomes, with "not persisted" shown explicitly. **Exception**: default bucket `zeroshot-learning-output` (the server rejects `zeroshot_learning_output`; R1) | Storage unit tests; integration tests; E2E test; **ADR-0002 Accepted by the project owner** (allowed by Principle VI since v1.1.0) |
| VII | Testing Discipline and Requirement Traceability | `tests/unit` offline with fakes; `tests/integration` and `tests/e2e` against real MinIO (skipped locally if MinIO is unreachable, failing in CI with `REQUIRE_MINIO=1`); `model`-marked real-model smoke test excluded by default; failure paths covered; an FR/AC-to-test map is kept in `tasks.md` | CI; traceability table in `tasks.md` |
| VIII | Explicit Error Handling | A typed `ZeroShotError` hierarchy with stable `code` values and `user_message` ([services.md](contracts/services.md)); all validation happens before inference; `PredictionService` never raises; the UI shows only `user_message` | One unit test per error code; UI walkthrough ([quickstart](quickstart.md) §5–6) |
| IX | Observability | JSON logs (stdlib) with the named events in [services.md](contracts/services.md); `request_id` in logs and in the record; separate `prediction.inferred` and `prediction.persisted`/`persist_failed` events; latency and load duration recorded | `caplog` unit tests for key events; quickstart §7 grep |
| X | Security and Privacy | Logs contain `input_chars` and `input_sha256_12`, never the text; a redaction filter; no secrets in errors or the image; non-root container; path safety for identifiers; README warnings (no auth; records contain text; how to delete them) | Log-content unit test; hygiene script; README review |
| XI | Simplicity (YAGNI) | One process and one package; no API server, database, queue, metrics endpoint or MLflow; stdlib logging; an in-process registry instead of distributed locks; optional assignment features deferred | This table; Complexity Tracking is empty |
| XII | ML Experiment and Inference Traceability | Each record carries `model`, `model_revision`, `inference_parameters`, `candidate_labels`, `label_set`, `application_version` and `schema_version`; model and settings changes go through the git-tracked `configs/models.yaml` | E2E record assertions; git history |
| XIII | Data Integrity | Pydantic validation, including cross-field invariants, before `put_object`; `allow_nan=False`; the key derived from the record timestamp; one write per UUID4; storage errors propagated | Storage and record unit tests; schema-sync test (pydantic model compared with the JSON Schema) |
| XIV | Extensibility | A model is added by appending to `models.yaml`; code never branches on the identifiers of the two initial models; `schema_version` with a documented change policy (data-model §8); no speculative plug-in layers | A unit test with a third fake model entry; code review (grep for hard-coded repo IDs outside `configs/`) |
| XV | Documentation | The README is built from [quickstart.md](quickstart.md) and covers every item in Quality Gate 8, the `MODEL_DIR`/`HF_HOME` explanation (R6), both ADRs, and record deletion | README review against Quality Gate 8 |

**Quality Gates 1–11**:

| Gate | How it is met |
|------|---------------|
| 1 Lint | `ruff format --check`, `ruff check` (CI and `uv run ruff …` locally) |
| 2 Unit | `pytest tests/unit` offline |
| 3 Integration | `pytest -m integration tests/integration` against Compose MinIO |
| 4 E2E persistence | `tests/e2e/test_prediction_persisted.py` |
| 5 CI | `.github/workflows/ci.yml` (R11) |
| 6 Hygiene | `scripts/check_hygiene.py` in CI |
| 7 Reproducibility | `uv sync --locked`; resolved versions recorded in this plan after the lock file is created |
| 8 Documentation | README sections listed in quickstart |
| 9 Acceptance | Quickstart §2–8 walkthrough covers AC-01 to AC-18 |
| 10 Traceability | `tasks.md` maps every FR, NFR and AC to tasks and tests |
| 11 Constitution Check | This section; exceptions only through an ADR (ADR-0002) |

Technology Baseline: all reference versions are installable on the supported platforms (R2),
so no discrepancy needs to be documented. The local developer prerequisite is a uv upgrade
(0.11.29 → 0.13.x).

## Project Structure

### Documentation (this feature)

```text
specs/001-zero-shot-platform/
├── plan.md              # This file
├── research.md          # Phase 0 (R1–R13)
├── data-model.md        # Phase 1
├── quickstart.md        # Phase 1, validation guide and README outline
├── contracts/
│   ├── prediction-record.schema.json
│   ├── configuration.md
│   ├── services.md
│   └── ui-pages.md
├── checklists/requirements.md
└── tasks.md             # Phase 2 (/speckit-tasks; not created here)

docs/adr/
├── 0001-hf-home-instead-of-transformers-cache.md   # Accepted
├── 0002-bucket-name-hyphens.md                     # Accepted (project owner)
└── 0003-minio-image-and-ports.md                   # Accepted
```

### Source Code (repository root)

```text
src/zeroshot/
├── __init__.py              # __version__
├── config.py                # Settings (env), catalogue and label-set file loaders
├── errors.py                # ZeroShotError hierarchy
├── logging.py               # JSON formatter, redaction, setup_logging()
├── bootstrap.py             # build_services(settings) to wire the services
├── domain/
│   ├── models.py            # ModelDefinition, CompletionMarker, ModelInstallation, ModelStatus
│   ├── labels.py            # LabelSet (+ normalization and validation)
│   └── predictions.py       # InferenceParameters, LabelScore, ClassificationResult,
│                            #   PredictionRecord, PredictionOutcome, OutcomeStatus
├── core/
│   ├── usage.py             # ModelUsageRegistry (leases, exclusives, on_removed)
│   └── paths.py             # safe_join / slug / is_relative_to checks
├── services/
│   ├── model_manager.py
│   ├── label_sets.py
│   ├── classifier.py
│   ├── storage.py
│   └── prediction.py
├── tools/
│   └── show_latest.py       # prints the newest stored record (quickstart §7)
└── ui/
    ├── app.py               # st.navigation entry point; st.cache_resource(build_services)
    ├── admin_page.py
    └── classify_page.py

configs/
├── models.yaml
└── labels.yaml              # seed label sets (default, extended)

tests/
├── conftest.py              # shared fixtures: settings, tmp MODEL_DIR/HF_HOME, ready_model,
│                            #   registry, and the MinIO fixtures (minio_client, test_bucket)
├── fakes.py                 # FakePipeline, FakeStorage, FakeHub (snapshot_download stand-in)
├── unit/
│   ├── conftest.py          # autouse offline guard (unit tests only)
│   ├── test_config.py
│   ├── test_paths.py
│   ├── test_usage_registry.py
│   ├── test_model_manager.py
│   ├── test_label_sets.py
│   ├── test_classifier.py
│   ├── test_storage.py
│   ├── test_prediction_service.py
│   ├── test_record_schema_sync.py
│   ├── test_logging.py
│   └── test_import_boundaries.py
├── integration/             # each module sets pytestmark = pytest.mark.integration
│   ├── test_bucket.py       # including the ADR-0002 evidence test
│   └── test_storage_minio.py
├── e2e/
│   └── test_prediction_persisted.py
└── model/
    └── test_real_model_smoke.py   # -m model

scripts/check_hygiene.py
docker-compose.yml           # minio (default) and app (profile "app")
Dockerfile                   # python:3.14.10-slim-bookworm + uv 0.13.7, CPU torch, non-root
.dockerignore  .gitignore  .env.example  .python-version
pyproject.toml  uv.lock  README.md
.github/workflows/ci.yml
```

**Structure Decision**: a single project with a `src/` layout. The UI is a thin layer
(`src/zeroshot/ui`) over an importable service package, which is what Principle II and
NFR-02 require. A separate backend and frontend would add an API layer that the spec does
not need (Principle XI). Test folders mirror the three test tiers of Principle VII.

## Complexity Tracking

No constitution violation needs a complexity justification. The only deviation is the
default bucket name (Principle VI), which is a constraint imposed by S3 naming rules, not
added complexity. It is handled through [ADR-0002](../../docs/adr/0002-bucket-name-hyphens.md)
as the governance section requires.
