# Zero-Shot Text Classification Platform

A small, reproducible MLOps platform for **zero-shot text classification**, built with
**Python 3.14 · PyTorch · Hugging Face Transformers · Streamlit · MinIO**.

An **administrator** downloads and manages zero-shot models and candidate-label sets. A
**user** classifies text against a saved label set. **Every successful classification is
stored as a JSON record in MinIO**, the system of record.

> 🔓 **Security limitation:** the Administration page has **no authentication**. Run the app
> only for local development or in trusted environments; never expose it to untrusted users.

---

## Contents

1. [Purpose and objectives](#1-purpose-and-objectives)
2. [Architecture](#2-architecture)
3. [Project structure](#3-project-structure)
4. [Supported models](#4-supported-models)
5. [Prerequisites](#5-prerequisites)
6. [Environment variables](#6-environment-variables)
7. [Quick start](#7-quick-start)
8. [Running MinIO](#8-running-minio)
9. [Running the app](#9-running-the-app)
10. [Model download, caching, deletion and cache cleanup](#10-model-download-caching-deletion-and-cache-cleanup)
11. [Candidate labels](#11-candidate-labels)
12. [Classifying text](#12-classifying-text)
13. [Verifying the stored prediction](#13-verifying-the-stored-prediction)
14. [Testing](#14-testing)
15. [Troubleshooting](#15-troubleshooting)
16. [Security, privacy, platform notes and deviations](#16-security-privacy-platform-notes-and-deviations)

---

## 1. Purpose and objectives

The project practises **AI-assisted, specification-driven development with Spec Kit** and
demonstrates core MLOps practices:

| Practice | How it is implemented |
|----------|-----------------------|
| Reproducible dependencies | `uv.lock` (universal lock; CPU-only torch on Linux), `.python-version` = 3.14.10 |
| Model version tracking | Every model is pinned to an immutable 40-character Hugging Face commit SHA; each record stores it |
| Model lifecycle | Atomic, validated downloads; reuse of existing downloads; safe deletion; cache cleanup |
| Configuration | Environment variables (`.env`) plus YAML catalogues; no hard-coded credentials |
| Persistence | One schema-validated JSON object per prediction in MinIO |
| Observability | Structured logs with a request ID; submitted text is never logged |
| Testing | Unit tests (offline), MinIO integration tests, an end-to-end persistence test, an opt-in real-model smoke test |
| Delivery | Docker Compose (MinIO and the app), a CPU-only app image, GitHub Actions CI |

The Spec Kit artifacts that drove the implementation are in the repository:
`.specify/memory/constitution.md` and `specs/001-zero-shot-platform/` (spec, plan, research,
data model, contracts, tasks, quickstart).

## 2. Architecture

```text
                         ┌──────────────────────────────┐
                         │  Streamlit app (ui/app.py)   │  st.navigation
                         └──────┬────────────────┬──────┘
                 ┌──────────────▼───┐        ┌───▼───────────────────┐
                 │ Administration   │        │ Classification page   │
                 │ page             │        │ text · model · labels │
                 └──┬────────────┬──┘        └───────────┬───────────┘
          ┌─────────▼──────┐ ┌───▼────────────┐ ┌────────▼─────────────┐
          │ ModelManager   │ │ LabelSetService│ │ PredictionService    │
          │ download/cache │ │ YAML label sets│ │ validate→lease→      │
          │ delete/cleanup │ └────────────────┘ │ classify→record→save │
          └───┬────────┬───┘                    └───┬──────────────┬───┘
              │        └──── ModelUsageRegistry ────┤              │
   ┌──────────▼───────┐       (leases, eviction)  ┌─▼────────────┐ ┌▼────────────────┐
   │ Hugging Face Hub │                           │ Classifier   │ │ StorageService  │
   │ → MODEL_DIR      │                           │ transformers │ │ MinIO bucket    │
   └──────────────────┘                           │ + PyTorch    │ └─────────────────┘
                                                  └──────────────┘
      config.py (settings, YAML loaders) and logging.py (JSON logs) are used by all
```

How the parts relate:

- **Streamlit pages** only render and collect input. They call services and show the
  `user_message` of any error. They never import `transformers`, `huggingface_hub` or
  `minio`, and a test enforces this.
- **`ModelManager`** owns model files. It never runs inference.
- **`Classifier`** loads pipelines and runs inference. It never downloads anything and
  never touches MinIO.
- **`PredictionService`** is the only path from the Classification page to inference and
  storage. It returns exactly one of three outcomes: **failed**, **computed but not
  persisted**, or **fully successful**.
- **`StorageService`** owns the MinIO bucket and the record format.
- **`ModelUsageRegistry`** prevents a model from being deleted, or the cache from being
  cleared, while a classification uses it. The model manager and the classifier both
  depend on it, but not on each other.

## 3. Project structure

```text
.
├── src/zeroshot/
│   ├── config.py            # Settings from env/.env; model catalogue loader; bucket-name check
│   ├── errors.py            # Typed errors with safe user messages and stable codes
│   ├── logging.py           # Structured JSON/console logging, secret redaction, text fingerprint
│   ├── bootstrap.py         # build_services(): wires everything once per process
│   ├── domain/              # Pydantic entities: models, label sets, predictions
│   ├── core/                # paths.py (path safety), usage.py (registry), validation.py
│   ├── services/
│   │   ├── model_manager.py # download / availability / delete / HF cache cleanup
│   │   ├── label_sets.py    # YAML label sets (atomic writes, file lock)
│   │   ├── classifier.py    # transformers zero-shot pipeline, LRU of loaded models
│   │   ├── storage.py       # MinIO bucket and record persistence
│   │   └── prediction.py    # orchestration: the three outcomes
│   ├── tools/show_latest.py # prints the newest stored prediction
│   └── ui/                  # app.py (entry point), admin_page.py, classify_page.py
├── configs/models.yaml      # model catalogue (pinned revisions)
├── configs/labels.yaml      # seed label sets (default, extended)
├── tests/                   # unit/ (offline) · integration/ + e2e/ (real MinIO) · model/ (opt-in)
├── scripts/check_hygiene.py # fails if secrets, models or caches are tracked
├── docs/adr/                # architecture decision records (deviations)
├── docker-compose.yml       # minio service (+ app service under the "app" profile)
├── Dockerfile               # CPU-only app image, non-root, no secrets or models inside
├── pyproject.toml, uv.lock  # dependencies (uv.lock is authoritative)
├── requirements.txt         # exported from uv.lock for reference; install with uv
├── .specify/memory/constitution.md       # project constitution
└── specs/001-zero-shot-platform/         # spec, plan, research, data model, contracts, tasks
```

Created at runtime and **never committed**: `models/` (downloaded models), `.cache/huggingface/`
(`HF_HOME`), `data/label_sets.yaml` (label sets edited by the administrator) and `.env`.

## 4. Supported models

| Name | Hugging Face id | Pinned revision | Licence | Hypothesis template | Download size |
|------|-----------------|-----------------|---------|---------------------|---------------|
| DeBERTa Zero-Shot v2 | `MoritzLaurer/deberta-v3-base-zeroshot-v2.0` | `8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8` | MIT | `This text is about {}` | ≈ 380 MB |
| BART MNLI | `facebook/bart-large-mnli` | `d7645e127eaf1aefc7862fd59a17a5aa8558b8ce` | MIT | `This example is {}.` | ≈ 1.6 GB |

Both use `multi_label: false`, so scores are normalized across the labels and sum to 1.

**To add a model**, append an entry to `configs/models.yaml`; no code changes are needed:

```yaml
  - key: my-model                       # stable app id
    name: My Model
    repo_id: org/my-nli-model           # must be an NLI model with an "entailment" label
    task: zero-shot-classification
    revision: <40-char commit SHA>      # branch names such as "main" are rejected
    license: apache-2.0
    inference_defaults: {multi_label: false, hypothesis_template: "This text is about {}."}
```

## 5. Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| [uv](https://docs.astral.sh/uv/) | **0.13.7 or newer** (`uv self update`) | `uv --version` |
| Docker with Compose v2 | Docker 20.10+, Compose 2.1+ | `docker compose version` |
| Disk | ≈ 2.5 GB for both models, plus ≈ 1.5 GB for the environment | `df -h .` |
| RAM | 8 GB minimum, 16 GB recommended | |

uv installs Python 3.14.10 automatically from `.python-version`.

Supported platforms: development on **macOS 14+ (Apple Silicon)** and **Linux x86_64**;
Docker and CI run on Linux x86_64, CPU only.

## 6. Environment variables

Configure the app with `cp .env.example .env`, then edit `.env`. It is git-ignored.

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_VERSION` | `0.1.0` | Stored as `application_version` in each record |
| `MODEL_DIR` | `./models` | Application-managed model directory |
| `HF_HOME` | `./.cache/huggingface` | Hugging Face library home (caches, token); target of *Clear HF cache* |
| `HF_TOKEN` | *(empty)* | Only for gated or private models; never logged |
| `MODELS_FILE` | `./configs/models.yaml` | Model catalogue |
| `LABEL_SETS_FILE` | `./data/label_sets.yaml` | Runtime label sets (created from the seed) |
| `LABEL_SETS_SEED_FILE` | `./configs/labels.yaml` | Seed label sets |
| `MAX_INPUT_CHARS` | `5000` | Maximum text length (1–100000) |
| `MAX_LOADED_MODELS` | `1` | Models kept in memory at once |
| `INFERENCE_DEVICE` | `cpu` | `cpu` or `mps` (Apple Silicon, opt-in); see §16 |
| `MINIO_ENDPOINT` | `localhost:9010` | `host:port` (inside Compose: `minio:9000`) |
| `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` | — (required) | Must equal `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` |
| `MINIO_SECURE` | `false` | `true` for TLS |
| `MINIO_BUCKET` | `zeroshot-learning-output` | Prediction bucket; see §16, *Deviations* |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | `console` gives human-readable logs |
| `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` | local-only values | Used by Compose to create the MinIO server (password ≥ 8 characters) |
| `MINIO_API_PORT` / `MINIO_CONSOLE_PORT` / `APP_PORT` | `9010` / `9011` / `8502` | Host ports |

`TRANSFORMERS_CACHE` is **ignored**; a startup warning appears if it is set. See §16.

## 7. Quick start

Clone the repository, or unzip the submitted archive, then work from the project root (the
folder that contains `pyproject.toml`):

```bash
cd ml-ops-zeroshot
cp .env.example .env          # local-only credentials
uv sync --locked              # → "Installed 75 packages"
docker compose up -d --wait minio                                # → "Container …-minio-1  Healthy"
uv run streamlit run src/zeroshot/ui/app.py --server.port 8502   # → "You can now view your Streamlit app … http://localhost:8502"
```

Then, in the browser at <http://localhost:8502>:

1. **Administration** → select *DeBERTa Zero-Shot v2* → **Download selected**. It should
   show "downloaded and validated (380 MB)".
2. **Classification** → paste *"The team won the championship after a very close game."*,
   choose the *Default topics* label set → **Classify**. It should show ✅ "Saved to MinIO."
   and predict **sports**.
3. Run `uv run python -m zeroshot.tools.show_latest` to print the stored JSON record.

## 8. Running MinIO

```bash
docker compose up -d --wait minio   # start, and wait until healthy
docker compose stop minio           # stop (data is kept in the named volume "minio-data")
docker compose restart minio        # restart; stored objects survive restarts
```

The **MinIO console** is at <http://localhost:9011>. Log in with `MINIO_ROOT_USER` and
`MINIO_ROOT_PASSWORD` from `.env`.

## 9. Running the app

**On the host** (fastest to iterate):

```bash
uv run streamlit run src/zeroshot/ui/app.py --server.port 8502   # http://localhost:8502; stop with Ctrl+C
```

**In Docker** (CPU-only image, runs as your UID so bind mounts stay writable on Linux):

```bash
# build the image and start minio + app on http://localhost:8502, running as your UID
mkdir -p models .cache/huggingface data
APP_UID=$(id -u) APP_GID=$(id -g) docker compose --profile app up -d --build

docker compose --profile app down   # stop both (the MinIO data volume is kept)
```

The container uses `MINIO_ENDPOINT=minio:9000` and mounts `./models`,
`./.cache/huggingface` and `./data`. Models downloaded by the host-run app are therefore
reused by the container, and vice versa.

## 10. Model download, caching, deletion and cache cleanup

**Two separate locations:**

| Location | Variable | Contains | Removed by |
|----------|----------|----------|------------|
| Managed model directory | `MODEL_DIR` (`./models`) | The app's own copy of each model: `models/<org>--<name>/<revision>/` plus a `.complete.json` marker | **Delete** (one model) |
| Hugging Face home | `HF_HOME` (`./.cache/huggingface`) | Library caches (`hub/`, `xet/`) and the optional token | **Clear HF cache** (`hub/` and `xet/` only) |

**How a download works:**

1. If `models/<org>--<name>/<revision>/.complete.json` exists and matches, the model is
   already present and nothing is downloaded.
2. Otherwise only the needed files are fetched, at the pinned revision, into
   `models/.staging/…`. The filter keeps `*.json`, `*.safetensors`, `*.model` and `*.txt`,
   and drops ONNX, Flax, Rust and `.bin` duplicates. For BART that is 1.6 GB instead of
   6.9 GB.
3. The files are validated without running inference: the config loads, the architecture
   is a sequence classifier, an `entailment` label exists, and the tokenizer loads.
4. The marker is written, and the directory is moved into place atomically.

Any failure deletes the staging directory, so a **partial download is never shown as
ready**. Files are written directly to `MODEL_DIR`, so there is no second copy in the HF
cache.

**Deleting a model** (*Model details* → **Delete…** → confirm) removes only that model's
directory. It is **refused while a classification is using the model**, and the loaded
pipeline is released from memory. The HF cache is not touched.

**Clearing the HF cache** (**Clear HF cache…** → confirm) empties `HF_HOME/hub` and
`HF_HOME/xet`. It never touches `MODEL_DIR` or `HF_HOME/token`, so **downloaded models stay
ready**. A missing cache directory gives "Nothing to clear". Cleanup is refused while a
classification or download is running.

## 11. Candidate labels

- **Seed:** `configs/labels.yaml` (git-tracked) contains `default` (sports, politics,
  technology, finance) and `extended` (the same plus health and education).
- **Runtime:** on first start the seed is copied to `data/label_sets.yaml`. All edits made
  on the Administration page go there, are written atomically under a file lock, and
  survive restarts.
- **Rules:** at least two labels; no empty labels; no duplicates, compared
  case-insensitively, so `Sports` duplicates `sports`; each label 100 characters at most.
  Labels are trimmed, but otherwise kept exactly as entered.
- On the Administration page, under **Candidate-label sets**, choose *➕ New label set* or
  an existing key, enter one label per line, then **Save label set**. Invalid sets are
  rejected with a message naming the label, for example `Duplicate label 'Sports'`.

## 12. Classifying text

The Classification page offers only **ready** models and **valid** saved label sets. Enter
up to `MAX_INPUT_CHARS` characters; a character counter is shown. Then click **Classify**.

Example result (DeBERTa, *Default topics*):

```text
✅ Saved to MinIO.
Predicted class: sports      Confidence: 99.9%
1. sports       0.9992
2. technology   0.0003
3. politics     0.0002
4. finance      0.0002
Model: DeBERTa Zero-Shot v2 (MoritzLaurer/deberta-v3-base-zeroshot-v2.0) · Revision 8e7e5af…
Timestamp (UTC) · Request ID · Latency ≈ 0.6–0.7 s on cpu · Object key predictions/2026/11/11/<id>.json
```

The three possible outcomes:

| Outcome | Shown as |
|---------|----------|
| Fully successful | ✅ "Saved to MinIO." plus the result |
| Computed but not persisted (MinIO down) | ⚠️ "Prediction produced but **not persisted** — storage error: MinIO is not reachable…", with the result labelled *(not saved)* |
| Failed (validation, model or inference) | ❌ a clear message, for example "Text exceeds the 5000-character limit (5001 characters)." Nothing is stored |

## 13. Verifying the stored prediction

Each successful classification is stored once, in bucket **`zeroshot-learning-output`**,
at `predictions/YYYY/MM/DD/{request_id}.json`. The date is the UTC date of the prediction.

```bash
uv run python -m zeroshot.tools.show_latest   # prints the newest record's key and JSON
```

Or open the console at <http://localhost:9011> → *Object Browser* →
`zeroshot-learning-output` → `predictions/<year>/<month>/<day>/`.

Record format (schema `specs/001-zero-shot-platform/contracts/prediction-record.schema.json`):

```json
{
  "schema_version": "1.0",
  "request_id": "3c1bce34-88f8-485f-97a2-bbb932a1dcac",
  "timestamp": "2026-11-11T18:17:13.688201Z",
  "input_text": "The team won the championship after a very close game.",
  "predicted_class": "sports",
  "confidence": 0.9992216229438782,
  "candidate_labels": ["sports", "politics", "technology", "finance"],
  "ranked_predictions": [
    {"label": "sports", "score": 0.9992216229438782},
    {"label": "technology", "score": 0.0003404801245778799},
    {"label": "politics", "score": 0.00022069089754950255},
    {"label": "finance", "score": 0.00021726940758526325}
  ],
  "model_name": "DeBERTa Zero-Shot v2",
  "model": "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
  "model_revision": "8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8",
  "inference_parameters": {"multi_label": false, "hypothesis_template": "This text is about {}", "device": "cpu"},
  "label_set": "default",
  "application_version": "0.1.0",
  "inference_latency_ms": 686.2,
  "session_id": null
}
```

The same `request_id` appears in the app logs (`prediction.persisted`), so a record can be
traced back to its log lines.

## 14. Testing

```bash
uv run ruff format --check . && uv run ruff check .                    # → "All checks passed!"
uv run pytest tests/unit                                               # offline → "183 passed"
REQUIRE_MINIO=1 uv run pytest -m integration tests/integration tests/e2e   # MinIO running → "9 passed"
uv run python scripts/check_hygiene.py                                 # → "hygiene: OK …"
uv run pytest -m model tests/model -s      # opt-in, DeBERTa downloaded → "predicted=sports … latency_ms≈600"
```

- **Unit tests** use fakes (`tests/fakes.py`) and an offline guard: any real download or
  pipeline call fails the test.
- **Integration tests** run against the real MinIO. Locally they are *skipped* if MinIO is
  not running; with `REQUIRE_MINIO=1`, as in CI, they *fail* instead. They also reproduce
  the bucket-name evidence for ADR-0002.
- **End-to-end** (`tests/e2e/`): prediction service, real MinIO and a deterministic fake
  model. It checks the stored object against the JSON Schema and the key format.
- **CI** (`.github/workflows/ci.yml`):
  - `uv sync --locked` and the `requirements.txt` sync check;
  - ruff, hygiene and unit tests;
  - MinIO via Compose, then the integration and e2e tests.

## 15. Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| `uv sync` cannot find Python 3.14.10, or says "required-version" | uv is too old: `uv self update` (≥ 0.13.7) |
| `Bind for 0.0.0.0:9010 failed: port is already allocated` | Set `MINIO_API_PORT`, `MINIO_CONSOLE_PORT` or `APP_PORT` in `.env` to free ports |
| ⚠️ "not persisted — MinIO is not reachable" / "Storage initialization failed" | Start MinIO: `docker compose up -d --wait minio`. Classification recovers without a restart |
| `MINIO_BUCKET '…' is not a valid S3 bucket name` | Bucket names allow only lowercase letters, digits, `-` and `.` (see §16) |
| `Access Denied` / `SignatureDoesNotMatch` in MinIO | `MINIO_ACCESS_KEY`/`MINIO_SECRET_KEY` must equal `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD`. If you changed them after the first start, run `docker compose down -v` (this deletes stored data) |
| "No model is ready — an administrator must download one…" | Download a model on the Administration page |
| Model shows **Incomplete installation** | Files exist without a valid marker; **Delete…** it, or download it again |
| Download fails with a network error | Check connectivity and retry; partial files are removed automatically |
| Download fails with a disk error | Free disk space (≈ 2 GB for both models) |
| `PermissionError` writing `models/` in Docker on Linux | Start the container with `APP_UID=$(id -u) APP_GID=$(id -g)` as shown in §9, so it runs as your UID |
| `error getting credentials … No stored credential` when building | Docker's credential helper problem. `docker pull python:3.14.10-slim-bookworm` first, then start the container again (§9) |
| "Configuration error: …" at the top of the app | A value in `.env` is missing or invalid; the message names the variable |

## 16. Security, privacy, platform notes and deviations

### Security

- **No authentication** on the Administration page: local or trusted use only.
- The credentials in `.env.example` are **local development placeholders**, not
  production-safe.
- Secrets come only from the environment. They are never logged, shown in errors or
  baked into the image. `.env`, models and caches are git-ignored, and CI checks this.
- Model identifiers and paths are validated, and deletion is restricted to the managed
  directories.

### Privacy

Prediction records **contain the submitted text** (`input_text`) and are stored in the
MinIO bucket above. Logs never contain the text, only its length and a short hash. To
remove stored predictions:

- **Console:** select the objects under `predictions/` and choose *Delete*.
- **CLI** (runs inside the MinIO container, where the root credentials are set):

  ```bash
  docker compose exec minio sh -c 'mc alias set local http://localhost:9000 "$MINIO_ROOT_USER" "$MINIO_ROOT_PASSWORD" && mc rm --recursive --force local/zeroshot-learning-output/predictions/'
  ```
- **Everything:** `docker compose down -v` deletes the MinIO data volume.

### Platform-dependent behaviour

- **Compute device:** `INFERENCE_DEVICE` defaults to `cpu`. `mps` (Apple Silicon) is
  opt-in. It may change scores in the later decimal places, and so the order of labels
  whose scores are almost equal. The device is recorded in each record as
  `inference_parameters.device`. CI and Docker always use `cpu`.
- **Active-use protection** is in-process: an in-memory registry refuses deletion and
  cache cleanup while a classification or download runs. It protects **one app process
  only**, so do not run the host app and the Docker app against the same `models/` directory
  at the same time.
- **Docker on Linux:** the §9 command runs the container as your UID so the bind-mounted
  directories stay writable.

### Performance on the reference machine

Apple M1 Pro, 16 GB, `INFERENCE_DEVICE=cpu`, measured 2026-11-11:

- **Classification** of the sample sentence with DeBERTa, once the model is loaded:
  **≈ 0.6–0.7 s on the host**, and **≈ 2.2 s inside the Docker container** (Linux VM). The
  SC-006 target is 10 s.
- **Downloads:** DeBERTa (380 MB) took 36 s; BART (1.6 GB) took 84 s.
- **From a fresh copy of the repository to all tests passing** (SC-004, target under 30
  minutes, excluding model downloads):
  - `uv sync --locked`: 1 s, because uv's package cache on this machine was already warm;
    a cold cache adds the download of about 1 GB of wheels;
  - lint: under 1 s;
  - MinIO start: 7 s;
  - 183 unit tests: 21 s;
  - 9 integration and e2e tests: 2 s;
  - **total about 30 s**, well inside the target even with a cold cache.
- **App image:** `zeroshot-app:0.1.0` is 1.63 GB (CPU-only torch, no CUDA).

### Deviations from the assignment brief (documented decisions)

| Brief | This project | Why |
|-------|--------------|-----|
| Bucket `zeroshot_learning_output` | **`zeroshot-learning-output`** (configurable) | Underscores are invalid in S3 bucket names. The MinIO server answers `InvalidBucketName` and the SDK refuses to create it. [ADR-0002](docs/adr/0002-bucket-name-hyphens.md) |
| `TRANSFORMERS_CACHE=…` | `HF_HOME` only | `TRANSFORMERS_CACHE` is deprecated. [ADR-0001](docs/adr/0001-hf-home-instead-of-transformers-cache.md) |
| `image: minio/minio`, ports 9000/9001 | `pgsty/minio:RELEASE.2026-08-04T00-00-00Z`, ports 9010/9011 (configurable) | Official images are unavailable, and the image must be pinned. [ADR-0003](docs/adr/0003-minio-image-and-ports.md) |
| `models/{model_id}` | `models/<org>--<name>/<revision>/` | One directory per pinned revision; `/` in repo ids is not path-safe |
| `model_revision: "main"` (example) | 40-character commit SHA | Reproducibility: branch names move |
| `app/pages/1_Admin.py` layout, `specification.md` | `src/zeroshot/ui/…`, `spec.md` | `src/` layout keeps the core testable without Streamlit; Spec Kit requires `spec.md` |
