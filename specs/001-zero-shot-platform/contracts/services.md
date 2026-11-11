# Contract: Core Service Interfaces

These are the Python interfaces the Streamlit pages are allowed to call (Principle II). Pages import
**only** `zeroshot.services.*`, `zeroshot.domain.*` and `zeroshot.errors`; they never import
`transformers`, `huggingface_hub`, `minio`, or touch the filesystem. Signatures are normative; bodies are
not part of this contract.

## Dependency graph

```text
ui/pages ──> PredictionService ──> Classifier ──┐
   │               │                            ├──> ModelUsageRegistry (core/usage.py)
   │               └──> StorageService          │
   ├──> ModelManager ───────────────────────────┘
   └──> LabelSetService
All services ──> Settings (config.py), logging.py, domain/, errors.py
```

`Classifier` and `ModelManager` never import each other; `Classifier` and `StorageService` never import
each other. Wiring happens once in `zeroshot/bootstrap.py::build_services(settings)`, cached by the UI with
`st.cache_resource`. Core modules do not import `streamlit`.

## ModelManager

```python
class ModelManager:
    def list_models(self) -> list[ModelInstallation]: ...
    def get(self, model_key: str) -> ModelInstallation: ...  # raises UnknownModelError
    def ready_models(self) -> list[ModelInstallation]: ...
    def download(
        self, model_keys: Sequence[str], on_progress: Callable[[str, str], None] | None = None
    ) -> dict[str, DownloadOutcome]: ...  # never raises for one model's failure
    def delete(
        self, model_key: str
    ) -> None: ...  # ModelInUseError, PathSafetyError, ModelDeleteError
    def clear_hf_cache(self) -> CacheCleanupReport: ...  # CacheInUseError, CacheCleanupError
    def paths(self) -> StoragePaths: ...  # model_dir, hf_home, hf_cache_targets
```

- `download` processes keys **sequentially**; each key yields `DownloadOutcome(status=
  "downloaded" | "already_present" | "failed", error: str | None, marker: CompletionMarker | None)`.
- `CacheCleanupReport(status="cleared" | "nothing_to_clear", removed_bytes: int, targets: list[Path])`.

## LabelSetService

```python
class LabelSetService:
    def list(self) -> list[LabelSet]: ...
    def get(self, key: str) -> LabelSet: ...  # UnknownLabelSetError
    def save(
        self, label_set: LabelSet
    ) -> LabelSet: ...  # LabelSetValidationError (create or replace)
    def delete(self, key: str) -> None: ...  # UnknownLabelSetError
    def load_warnings(self) -> list[str]: ...  # invalid entries skipped at load
```

## Classifier

```python
class Classifier:
    def classify(
        self,
        text: str,
        labels: Sequence[str],
        model: ModelInstallation,
        params: InferenceParameters,
    ) -> ClassificationResult:
        ...
        # ModelNotReadyError, ModelLoadError, InferenceError; validates inputs → InputValidationError

    def evict(self, model_key: str) -> None: ...  # registered as registry.on_removed callback
```

The pipeline factory is injected (`pipeline_factory: Callable[[Path, str], ZeroShotPipeline]`, called as
`factory(model_path, device)`) so unit tests pass a fake. Production uses
`transformers.pipeline("zero-shot-classification", model=str(path), device=settings.inference_device)`.
The device is **never auto-detected**. It is reported in `ClassificationResult.inference_parameters.device`.

## StorageService

```python
class StorageService:
    def ensure_bucket(self) -> None: ...  # BucketInitError, StorageUnavailableError
    def object_key(self, record: PredictionRecord) -> str: ...
    def save(
        self, record: PredictionRecord
    ) -> str: ...  # returns key; StorageUnavailableError, StorageWriteError
    def get(self, key: str) -> PredictionRecord: ...  # used by tests and quickstart verification
```

Exception mapping (normative):

| Raised by the client | In `ensure_bucket` | In `save` / `get` |
|----------------------|--------------------|-------------------|
| `minio.error.S3Error` | `BucketInitError` | `StorageWriteError` (`get`: `StorageError`, code `storage_read_failed`) |
| `urllib3.exceptions.MaxRetryError`, `urllib3.exceptions.ProtocolError`, `ConnectionError`, `TimeoutError` | `StorageUnavailableError` | `StorageUnavailableError` |
| `PredictionRecord` validation failure | — | `StorageWriteError` (code `invalid_record`), raised **before** `put_object` |

## PredictionService (FR-12)

```python
class PredictionService:
    def predict(
        self, text: str, model_key: str, label_set_key: str, session_id: str | None = None
    ) -> PredictionOutcome: ...  # never raises ZeroShotError
```

Flow: validate → `registry.lease(model_key)` → `classifier.classify` → build `PredictionRecord` → validate →
`storage.save` → release lease (in `finally`). Mapping:

| Failure point | `status` | Stored? |
|---------------|----------|---------|
| validation / unknown or not-ready model / label set | `FAILED` | no |
| model load / inference | `FAILED` | no |
| record validation | `FAILED` (logged as bug) | no |
| `ensure_bucket` / `save` | `COMPUTED_NOT_PERSISTED` | no |
| all steps OK | `SUCCEEDED` | yes |

Unexpected (non-`ZeroShotError`) exceptions are logged with stack trace and returned as `FAILED` with a
generic `InternalError` message — the UI never shows a traceback.

## Errors (`zeroshot/errors.py`)

```text
ZeroShotError(user_message: str, *, code: str)
├── ConfigError
├── InputValidationError            codes: empty_text, text_too_long, too_few_labels, empty_label, duplicate_label
├── LabelSetError
│   ├── UnknownLabelSetError
│   └── LabelSetValidationError
├── ModelError
│   ├── UnknownModelError
│   ├── ModelNotReadyError          → UI hint: "download it on the Administration page"
│   ├── ModelInUseError
│   ├── ModelDownloadError
│   ├── ModelLoadError
│   └── ModelDeleteError
├── PathSafetyError
├── CacheError
│   ├── CacheInUseError
│   └── CacheCleanupError
├── InferenceError
├── StorageError
│   ├── BucketInitError
│   ├── StorageUnavailableError
│   └── StorageWriteError
└── InternalError
```

`user_message` is safe to display (no secrets, paths only when they are app-managed, no stack traces).
`code` is stable and asserted in tests.

### Error codes (complete list; the only allowed values)

| Exception | Codes (first one is the default) |
|-----------|----------------------------------|
| `ConfigError` | `config_invalid`, `config_missing`, `invalid_bucket_name`, `catalogue_invalid`, `device_unavailable` |
| `InputValidationError` | `empty_text`, `text_too_long`, `too_few_labels`, `empty_label`, `duplicate_label` |
| `UnknownLabelSetError` | `unknown_label_set` |
| `LabelSetValidationError` | `label_set_invalid`, `empty_label`, `duplicate_label`, `too_few_labels`, `label_too_long`, `invalid_key`, `invalid_name` |
| `UnknownModelError` | `unknown_model` |
| `ModelNotReadyError` | `model_not_ready` |
| `ModelInUseError` | `model_in_use`, `model_being_deleted` |
| `ModelDownloadError` | `download_failed`, `repo_not_found`, `revision_not_found`, `network_error`, `disk_error`, `validation_failed` |
| `ModelLoadError` | `model_load_failed` |
| `ModelDeleteError` | `delete_failed` |
| `PathSafetyError` | `unsafe_path` |
| `CacheInUseError` | `cache_in_use` |
| `CacheCleanupError` | `cache_cleanup_failed` |
| `InferenceError` | `inference_failed`, `invalid_scores` |
| `BucketInitError` | `bucket_init_failed` |
| `StorageUnavailableError` | `storage_unavailable` |
| `StorageWriteError` | `storage_write_failed`, `invalid_record` |
| `StorageError` (base, used directly only by `get`) | `storage_read_failed` |
| `InternalError` | `internal_error` |

## Log events (Principle IX)

`app.startup`, `config.validated` (always emitted once by `build_services` after logging is configured, with
a `warnings` list, e.g. the ignored `TRANSFORMERS_CACHE`), `model.download.started|succeeded|skipped|failed`, `model.deleted`,
`model.delete.refused`, `cache.cleared`, `cache.cleanup.failed`, `pipeline.loaded|evicted` (`loaded` carries
`device` and `duration_ms`), `pipeline.load_failed` (with `exc_info`, server-side log only),
`prediction.started`, `prediction.inferred`, `prediction.persisted`, `prediction.persist_failed`,
`prediction.failed`, `storage.bucket.created|exists|failed`. Prediction events carry `request_id`,
`model`, `revision`, `label_set`, `input_chars`, `input_sha256_12`, `duration_ms`, `outcome` — never
`input_text`.
