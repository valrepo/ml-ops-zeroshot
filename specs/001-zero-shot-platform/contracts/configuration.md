# Contract: Configuration

The application reads three external inputs: **environment variables**, **`configs/models.yaml`**, and the
**label-set file**. Field rules are in [data-model.md](../data-model.md); this file fixes names, defaults
and formats.

## 1. Environment variables

Loaded by `zeroshot.config.Settings` (pydantic-settings). A local `.env` is read if present (git-ignored).
`.env.example` lists every variable below with placeholder or local-only values.

| Variable | Default | Secret | Description |
|----------|---------|--------|-------------|
| `APP_VERSION` | value of `zeroshot.__version__` | no | Stored as `application_version` in records |
| `MODEL_DIR` | `./models` | no | Application-managed model root (authoritative copies) |
| `HF_HOME` | `./.cache/huggingface` | no | Hugging Face library home (Hub/Xet caches, token); target of cache cleanup |
| `HF_TOKEN` | unset | **yes** | Only needed for gated/private models; never logged or shown |
| `MODELS_FILE` | `./configs/models.yaml` | no | Model catalogue |
| `LABEL_SETS_FILE` | `./data/label_sets.yaml` | no | Runtime label sets (created from seed if missing) |
| `LABEL_SETS_SEED_FILE` | `./configs/labels.yaml` | no | Seed for the above |
| `MAX_INPUT_CHARS` | `5000` | no | Integer 1–100000 |
| `MAX_LOADED_MODELS` | `1` | no | Integer ≥1; pipeline LRU size |
| `INFERENCE_DEVICE` | `cpu` | no | `cpu` or `mps`. Explicit, never auto-detected. `mps` is opt-in on Apple Silicon; if it is requested but unavailable, startup fails with `ConfigError`. Recorded in every record as `inference_parameters.device`, because scores can differ slightly between devices (Principle III) |
| `MINIO_ENDPOINT` | `localhost:9010` | no | `host:port`, no scheme (inside Compose: `minio:9000`) |
| `MINIO_ACCESS_KEY` | — (required) | **yes** | |
| `MINIO_SECRET_KEY` | — (required) | **yes** | |
| `MINIO_SECURE` | `false` | no | `true` for TLS |
| `MINIO_BUCKET` | `zeroshot-learning-output` | no | Validated against S3 naming rules at startup — see ADR-0002 |
| `LOG_LEVEL` | `INFO` | no | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `LOG_FORMAT` | `json` | no | `json` or `console` (human-readable, for local dev) |

Compose-only (not read by Python): `MINIO_API_PORT` (9010), `MINIO_CONSOLE_PORT` (9011), `APP_PORT` (8502),
`MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, and `APP_UID`/`APP_GID` (default `10001`; the README §9 start command sets
them to the host user's `id -u`/`id -g` so bind-mounted `models/`, `.cache/huggingface/` and `data/` stay writable on Linux
hosts). The names avoid `UID`, which is read-only in bash.

`TRANSFORMERS_CACHE` is **not** read. If set, a startup warning explains that `HF_HOME` is used instead
(ADR-0001). `.env.example` contains it only as a commented-out line marked deprecated.

Errors: a missing or invalid variable raises `ConfigError("MINIO_SECRET_KEY is required")` / `ConfigError(
"MAX_INPUT_CHARS must be an integer between 1 and 100000")`. Secret values are never interpolated into
messages.

## 2. `configs/models.yaml`

```yaml
models:
  - key: deberta-v3-zeroshot-v2          # stable app identifier
    name: DeBERTa Zero-Shot v2
    repo_id: MoritzLaurer/deberta-v3-base-zeroshot-v2.0
    task: zero-shot-classification
    revision: 8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8
    license: mit
    inference_defaults:
      multi_label: false
      hypothesis_template: "This text is about {}"
  - key: bart-large-mnli
    name: BART MNLI
    repo_id: facebook/bart-large-mnli
    task: zero-shot-classification
    revision: d7645e127eaf1aefc7862fd59a17a5aa8558b8ce
    license: mit
    inference_defaults:
      multi_label: false
      hypothesis_template: "This example is {}."
    # download:                         # optional override of research R5 defaults
    #   allow_patterns: [...]
    #   ignore_patterns: [...]
```

Adding a compatible model = appending one entry (FR-02, Principle XIV). No code changes.

## 3. Label-set file (`LABEL_SETS_FILE`, seed `configs/labels.yaml`)

```yaml
label_sets:
  - key: default
    name: Default topics
    labels: [sports, politics, technology, finance]          # demo set, brief §13
  - key: extended
    name: Extended topics
    labels: [sports, politics, technology, finance, health, education]   # brief §3.3
```

The seed (`configs/labels.yaml`) is tracked in git and never written by the app. The runtime
file (`data/label_sets.yaml`, git-ignored) is created from the seed on first start and holds
the administrator's edits (research R9).

Written only by `LabelSetService` (atomic replace under a file lock). Hand edits are allowed while the app
is stopped; invalid entries are reported at load time and skipped, valid ones stay usable.

## 4. MinIO layout

| Item | Value |
|------|-------|
| Bucket | `MINIO_BUCKET` (default `zeroshot-learning-output`) |
| Object key | `predictions/YYYY/MM/DD/{request_id}.json` (UTC) |
| Body | UTF-8 JSON conforming to [prediction-record.schema.json](prediction-record.schema.json) |
| Content-Type | `application/json; charset=utf-8` |
