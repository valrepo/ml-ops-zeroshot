# Phase 0 Research: Zero-Shot Text Classification Platform

**Feature**: `001-zero-shot-platform` | **Date**: 2026-11-11 | **Plan**: [plan.md](plan.md)

Every item in the spec's "To verify during planning" list is resolved here. Results marked
**verified** were checked on 2026-11-11 against PyPI, the Hugging Face Hub API, Docker
Hub/GHCR, the pinned library sources, or a live MinIO container. Nothing in this file is
assumed from memory.

## R1. Default bucket name `zeroshot_learning_output` — REJECTED (verified)

**Test**: started `pgsty/minio:RELEASE.2026-08-04T00-00-00Z` and called `bucket_exists` /
`make_bucket` with the `minio` 7.2.20 SDK on Python 3.14.

| Bucket name | Result |
|-------------|--------|
| `zeroshot_learning_output` | `S3Error: InvalidBucketName — The specified bucket is not valid.` (rejected by the **server**) |
| `zeroshot-learning-output` | Created successfully |

- **Decision**: default `MINIO_BUCKET=zeroshot-learning-output` (hyphens instead of
  underscores). The bucket stays configurable. The storage service validates the configured
  name against S3 naming rules at startup and fails with an actionable message if it is
  invalid, so the underscore name is caught early rather than on the first write.
- **Governance**: this deviates from constitution Principle VI and spec FR-09/AC-03.
  It is recorded in [ADR-0002](../../docs/adr/0002-bucket-name-hyphens.md), now
  **Accepted** by the project owner, which Principle VI allows since constitution v1.1.0.
- **Rationale**: S3 naming rules (which MinIO enforces) allow only lowercase letters,
  digits, hyphens and dots. No server setting was found that would make the required name
  creatable, and an alternative object store is out of scope (Principle VI).
- **Alternatives considered**: (a) a different server image, rejected because all
  S3-compatible servers apply the same rules and Principle I fixes MinIO; (b) a single
  bucket with prefix `zeroshot_learning_output/`, rejected because it changes the required
  key layout and hides the deviation instead of documenting it.

## R2. Runtime and dependency versions (verified)

All Technology Baseline reference versions exist and are installable on Python 3.14:

| Component | Baseline | Verified | Notes |
|-----------|----------|----------|-------|
| Python | 3.14.10 | python.org release exists; image `python:3.14.10-slim-bookworm` exists | uv **0.11.29** (installed locally) only knows 3.14.6; uv ≥0.13 is needed to fetch 3.14.10 |
| torch | 2.15.0 | cp314 wheels: macOS arm64 (PyPI); Linux x86_64 `+cpu` (PyTorch CPU index) | see R3 |
| transformers | 5.22.2 | PyPI, `requires_python>=3.11` | PyTorch-only in v5; no TF/JAX fallback possible |
| huggingface_hub | 2.4.0 | PyPI | `snapshot_download(revision, local_dir, allow_patterns, ignore_patterns, token)` present |
| streamlit | 1.67.0 | PyPI | `st.navigation`, `st.Page`, `st.cache_resource`, `st.dialog` present |
| minio (SDK) | 7.2.20 | PyPI; live test against server | |
| pytest | 9.2.0 | PyPI | |
| ruff | 0.17.4 | PyPI | |
| uv | project-selected | **0.13.7** (latest on PyPI; `ghcr.io/astral-sh/uv:0.13.7` exists) | pinned via `[tool.uv] required-version` and in CI |
| MinIO server | pinned, never `latest` | **`pgsty/minio:RELEASE.2026-08-04T00-00-00Z`** (amd64 + arm64) | Official `minio/minio` images have been unavailable since Oct 2025 |

**Supporting libraries** (not stack components; no ADR needed because they replace nothing
in the baseline):

| Library | Version | Why |
|---------|---------|-----|
| pydantic | `>=2.14,<3` (2.14.3) | One definition and validation point for the prediction record, model catalogue and label sets (Principles VI, XIII) |
| pydantic-settings | `>=2.16,<3` (2.16.2) | Typed environment configuration with actionable errors (Principle IV) |
| PyYAML | `>=6.0,<7` (6.0.3) | Reading and writing `models.yaml` and `label_sets.yaml` |
| filelock | `>=4.2,<5` (4.2.10) | Cross-process lock for label-set writes; already a transitive dependency of transformers/hub |

- **Decision**: pin the baseline exactly as listed. The developer prerequisite is uv 0.13.x
  (`uv self update`). The resolved versions are recorded in the plan's Technical Context.
- **Alternatives considered**: staying on Python 3.14.6 so the installed uv works, rejected
  because it contradicts the baseline when a one-line uv upgrade fixes it.

## R3. CPU-only torch on Linux, native torch on macOS

- **Decision**: in `pyproject.toml`, `[tool.uv.sources]` maps `torch` to an explicit
  `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`) with marker
  `sys_platform == 'linux'`. On macOS, torch comes from PyPI (the arm64 wheel is already
  CPU+MPS). `uv.lock` stays universal for both platforms.
- **Rationale**: the PyPI Linux x86_64 wheel pulls in CUDA libraries (several GB), which
  breaks the CPU-only image requirement and wastes CI time. The `+cpu` wheel exists for
  cp314 manylinux x86_64 (verified).
- **Alternatives considered**: separate requirements files per platform, rejected because it
  breaks "`uv.lock` is authoritative".

## R4. Model revisions, licences and default settings (verified)

| Field | DeBERTa Zero-Shot v2 | BART MNLI |
|-------|----------------------|-----------|
| Repo | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `facebook/bart-large-mnli` |
| Immutable revision | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` |
| Licence | `mit` | `mit` |
| Gated | no | no |
| Pipeline tag | `zero-shot-classification` | `zero-shot-classification` |
| Architecture | `DebertaV2ForSequenceClassification` | `BartForSequenceClassification` |
| `id2label` | `{0: entailment, 1: not_entailment}` | `{0: contradiction, 1: neutral, 2: entailment}` |
| `hypothesis_template` | `"This text is about {}"` (model card) | `"This example is {}."` (pipeline default; matches card) |
| `multi_label` | `false` (model card) | `false` (pipeline default) |
| Weights needed | `model.safetensors` 369 MB | `model.safetensors` 1.63 GB |

- **Decision**: these values go into `configs/models.yaml`. Settings are explicit per model
  (never "whatever the pipeline defaults to") so that they are recorded and deterministic
  (FR-07).
- **Rationale**: both revisions are full 40-character commit SHAs and therefore immutable
  (Principle III). Both models declare an `entailment` label, which the zero-shot pipeline
  needs.

## R5. Which files to download

The repositories contain redundant weight formats: BART is about **6.9 GB** in total
(`pytorch_model.bin`, `flax_model.msgpack` and `rust_model.ot` duplicate the safetensors
weights), and DeBERTa ships an `onnx/` copy of about **750 MB**.

- **Decision**: `snapshot_download` with
  `allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt"]` and
  `ignore_patterns=["onnx/*", "*.bin", "*.msgpack", "*.ot", "*.h5"]`. A model entry MAY
  override these patterns in `models.yaml`. The result is about 380 MB for DeBERTa and
  about 1.64 GB for BART.
- **Rationale**: cuts BART's download by about 75%, and never downloads ONNX weights, so
  there is nothing for a non-PyTorch backend to fall back to (Principle I).
- **Alternatives considered**: downloading the full snapshot, rejected because of disk use
  (91% full on the reference machine) and download time.

## R6. `MODEL_DIR` vs `HF_HOME` (one copy per model)

- **Decision**:
  - `MODEL_DIR` (default `./models`) is the **authoritative, application-managed** store.
    Models are downloaded with `snapshot_download(local_dir=...)` straight into a staging
    directory under `MODEL_DIR/.staging/`, so there is **no second copy** in the Hub cache.
  - `HF_HOME` (default `./.cache/huggingface`) is the Hugging Face library's own home: the Hub cache
    (`hub/`), the Xet chunk cache (`xet/`) and similar. The app never reads model weights
    from it.
  - **Model deletion** removes `MODEL_DIR/<slug>/<revision>/` only.
  - **Cache cleanup** removes the contents of `HF_HOME/hub` and `HF_HOME/xet` only. It
    never touches `MODEL_DIR` or the token file (`HF_HOME/token`).
  - `TRANSFORMERS_CACHE` is not used ([ADR-0001](../../docs/adr/0001-hf-home-instead-of-transformers-cache.md)).
- **Rationale**: one copy per model (Principle V), and a clear answer to "what does each
  button delete?" (spec US-05, FR-05).
- **Alternatives considered**: loading models from the Hub cache, rejected because the cache
  layout is library-internal and cleanup would then break models that appear ready.

## R7. Atomic download, validation and completion marker

- **Decision**:
  1. Check: `MODEL_DIR/<slug>/<revision>/.complete.json` exists and matches the model id and
     revision, so the model is ready and no download happens (AC-07).
  2. Download into `MODEL_DIR/.staging/<slug>--<revision>--<uuid4>/`.
  3. Validate **without running inference**:
     - the required files exist (`config.json`, ≥1 `*.safetensors`, and tokenizer files);
     - `AutoConfig.from_pretrained(staging)` loads;
     - the architecture ends with `ForSequenceClassification`;
     - `id2label` contains an `entailment` label (case-insensitive);
     - `AutoTokenizer.from_pretrained(staging)` loads.
  4. Write `.complete.json` (model id, resolved revision, UTC timestamp, file list, total
     bytes) inside the staging directory.
  5. `os.replace` the staging directory to `MODEL_DIR/<slug>/<revision>/` (same filesystem,
     so the move is atomic).
  6. On any failure, remove the staging directory. On startup, delete stale `.staging/*`
     entries older than 1 hour.
- **Rationale**: no partial download is ever visible at the final path (Principle V,
  NFR-03). The marker is the single source of truth for "ready".
- **Alternatives considered**: a marker file without a staging directory, rejected because a
  crash between file writes could still leave a half-populated final path.

## R8. Active-use coordination and pipeline eviction

Streamlit runs every session in **one process** with multiple threads.

- **Decision**: a small thread-safe `ModelUsageRegistry` (in `zeroshot/core/usage.py`) that
  both the model manager and the classifier depend on (they do not depend on each other):
  - `lease(model_key)` is a context manager that increments a reference count, and is
    refused if the model is being deleted.
  - `begin_exclusive(model_key)` is used by deletion. It is refused with `ModelInUseError`
    if the count is above 0; otherwise it blocks new leases until it is released.
  - `begin_global_exclusive()` is used by cache cleanup. It is refused if any lease or
    download is active.
  - `on_removed(callback)`: the classifier registers its eviction callback, and the model
    manager calls `registry.notify_removed(model_key)` after a deletion (FR-05: deletion
    releases the loaded model).
- **Decision**: the classifier keeps an LRU of loaded pipelines with
  `MAX_LOADED_MODELS=1` by default (configurable). Loading a second model evicts the first.
  On 16 GB machines this keeps peak memory to one model.
- **Rationale**: deletion of a model in use is refused, not deferred, as decided in the
  spec's Assumptions. The design avoids cross-process locks and extra services (Principle
  XI).
- **Limitation (documented)**: the registry protects within one app process. Running
  several app replicas against a shared `MODEL_DIR` is out of scope.

## R9. Label-set persistence

- **Decision**: a seed file `configs/labels.yaml` (tracked in git, containing the
  `default` set) and a runtime file at `LABEL_SETS_FILE` (default `./data/label_sets.yaml`,
  git-ignored). On first run the runtime file is created from the seed. Writes take a
  `filelock` lock and use an atomic temp-file write followed by `os.replace`.
- **Rationale**: administrator edits survive restarts (US-06) without making the git
  working tree dirty. The seed guarantees that a fresh install can classify immediately.
- **Alternatives considered**: storing label sets in MinIO, rejected because MinIO's role
  is prediction output (Principle VI) and the spec asks for a simple configuration file;
  editing the tracked seed file in place was also rejected.

## R10. Structured logging

- **Decision**: standard-library `logging` with a small JSON formatter in
  `zeroshot/logging.py`. Every event has `event`, `level`, `ts` (UTC), and, where they
  apply, `request_id`, `model`, `revision`, `duration_ms`, `outcome`. Input text is logged
  only as `input_chars` plus `input_sha256_12` (the first 12 hex characters of the SHA-256
  hash). A redaction filter masks values whose keys match `token|secret|password|key`.
- **Alternatives considered**: structlog, rejected as an extra dependency the project does
  not need (Principle XI).

## R11. Testing strategy and CI

- **Decision**:
  - **Unit** (`tests/unit`): a fake pipeline, a fake storage service and `tmp_path`
    directories, with no network access. `pytest-socket` was considered but rejected. An
    autouse guard in **`tests/unit/conftest.py`** (not the root conftest) sets
    `HF_HUB_OFFLINE=1` and fails on any real download or pipeline call. It is scoped to
    unit tests so that the opt-in real-model smoke test in `tests/model/` can still load
    a real pipeline.
  - **Shared fixtures**: the MinIO fixtures live in the root `tests/conftest.py` (not
    autouse). pytest only supports `pytest_plugins` in the top-level conftest, so sharing
    fixtures through `tests/integration/conftest.py` would not reach `tests/e2e/`.
  - **Integration** (`tests/integration`, marker `integration`): run against a real MinIO.
    They are skipped locally when MinIO is unreachable, but **fail** when `REQUIRE_MINIO=1`
    is set (as it is in CI).
  - **End-to-end** (`tests/e2e`, marker `integration`): the real prediction service, a
    deterministic fake classifier and real MinIO. Verifies the stored record, its key and
    all fields (Quality Gate 4).
  - **Real-model smoke test** (marker `model`): excluded by default (`-m "not model"`) and
    run manually.
- **CI**: GitHub Actions on `ubuntu-latest` with `astral-sh/setup-uv` pinned to 0.13.7,
  then:
  - `uv sync --locked`;
  - `ruff format --check` and `ruff check`;
  - unit tests;
  - `docker compose up -d --wait minio`;
  - integration and e2e tests with `REQUIRE_MINIO=1`;
  - a hygiene script that fails if `git ls-files` contains `.env`, `models/`, `.cache/huggingface/`,
    `*.safetensors`, `*.bin`, or files over 5 MB.
- **Rationale**: GitHub service containers cannot pass the `server /data` command that MinIO
  needs, while Compose reuses the developer's exact setup.

## R12. Containers and ports

- **Decision**:
  - The Dockerfile builds from `python:3.14.10-slim-bookworm` and copies uv from
    `ghcr.io/astral-sh/uv:0.13.7`. It runs `uv sync --locked --no-dev` (which pulls the
    CPU torch on Linux), runs as a non-root user, and exposes Streamlit on port 8501. It
    contains no `.env`, models or cache; `MODEL_DIR`, `HF_HOME` and `data/` are volumes.
  - `.dockerignore` excludes those directories plus `.git`.
  - `docker-compose.yml` defines a `minio` service (pinned image, a named volume, a health
    check, and host ports `${MINIO_API_PORT:-9010}` and `${MINIO_CONSOLE_PORT:-9011}`) and
    an `app` service (host port `${APP_PORT:-8502}`) behind the Compose profile `app`, so
    `docker compose up -d minio` stays lightweight for local development.
- **Rationale**: on the reference machine, host ports 9000, 9001 and 8501 are already in use
  and 5000/8000 are known to be taken. Configurable, non-default ports avoid start-up
  failures for evaluators too.
- **Decision (file ownership)**: the `app` service runs as `user: "${APP_UID:-10001}:${APP_GID:-10001}"`.
  The Docker start command (README §9) first creates `models/`, `.cache/huggingface/` and `data/` on the host, then passes
  `APP_UID=$(id -u) APP_GID=$(id -g)`. The image sets `HOME=/tmp`, so Streamlit can write
  its config for any UID.
- **Rationale**: on Linux hosts, folders Docker creates for bind mounts are owned by root,
  so a fixed non-root UID would get "permission denied" when writing models or label sets.
  Docker Desktop on macOS hides this problem. Bind mounts are kept, rather than named
  volumes, so that models downloaded by the host-run app are shared with the container.

## R13. Score semantics and tie-breaking

- **Decision**: the classifier asks the pipeline for every label, then re-sorts with the key
  `(-score, original_index)`, so equal scores keep their order in the candidate list
  (FR-07). Single-label mode applies a softmax over labels, so scores sum to 1 (a test
  checks the sum is within 1e-6). Multi-label mode scores each label independently in [0, 1].
  Non-finite scores raise `InferenceError`.
- **Rationale**: the pipeline's own sort is not guaranteed to be stable for ties, so the
  ordering is enforced by the classifier.

## R14. Explicit compute device

- **Decision**: a new setting `INFERENCE_DEVICE` (`cpu` | `mps`, default `cpu`), passed to
  the pipeline factory. It is never auto-detected. If `mps` is set but
  `torch.backends.mps.is_available()` is false, startup fails with
  `ConfigError(code="device_unavailable")`. The effective device is stored in
  `inference_parameters.device` in every record and logged on `pipeline.loaded`.
- **Rationale**: constitution Principle III requires that inference parameters which can
  affect results be explicit, configured and documented, and that known platform-dependent
  behavior be documented. CPU and MPS kernels can give slightly different scores (float
  precision and kernel differences), so an automatic choice would make the same request
  give different records on different machines without leaving a trace. A `cpu` default
  also makes SC-006 measurable as specified.
- **Documented behavior** (README): with `mps`, scores may differ from CPU in the later
  decimal places, and very close labels could swap ranks. CI and Docker always use `cpu`.
- **Alternatives considered**: auto-selecting MPS, which was the previous plan, rejected
  because it was implicit and not recorded.

## Remaining open items

| Item | Owner | Blocking? |
|------|-------|-----------|
| ~~Approval of ADR-0002~~ | Project owner | Resolved: approved by the project owner (constitution v1.1.0, Principle VI) |
| ~~Developer upgrades local uv from 0.11.29 to 0.13.x~~ | Project owner | Resolved: upgraded to 0.13.7 |
