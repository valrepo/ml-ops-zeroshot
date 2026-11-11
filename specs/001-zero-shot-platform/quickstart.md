# Quickstart & Validation Guide: Zero-Shot Text Classification Platform

**Feature**: `001-zero-shot-platform` | **Spec**: [spec.md](spec.md) | **Plan**: [plan.md](plan.md)

This guide runs the platform end to end and proves the acceptance criteria. It is the
outline for the README (Principle XV). Expected outcomes refer to AC-xx in the spec.

## Prerequisites

| Tool | Version | Check |
|------|---------|-------|
| uv | 0.13.x (`uv self update` if older) | `uv --version` |
| Docker + Compose v2 | Docker 20.10+ | `docker compose version` |
| Disk | about 2.5 GB free for both models, plus about 1 GB for the environment | `df -h .` |
| RAM | 8 GB minimum, 16 GB recommended (BART loads about 1.6 GB of weights) | |

Python 3.14.10 is installed by uv from `.python-version`; no system Python is needed.

## 1. Set up from a clean checkout (AC-01, Quality Gate 7)

Get the project in one of two ways: clone the repository, or unzip the submitted archive
(the brief's submission is by email). Then work from the project root, the folder that
contains `pyproject.toml`:

```bash
cd ml-ops-zeroshot                   # or wherever you cloned/unzipped it
cp .env.example .env                 # local-only credentials; edit if you like
uv sync --locked                     # installs exactly what is in uv.lock
```

Expected: `uv sync --locked` finishes without "lockfile needs to be updated".

## 2. Start MinIO (AC-02)

```bash
docker compose up -d --wait minio
```

Expected: the container `minio` is `healthy`. The console is at http://localhost:9011
(log in with `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` from `.env`).

Persistence check: `docker compose restart minio`. Objects written
earlier are still listed afterwards.

## 3. Run the tests (AC-19, AC-20; Quality Gates 1–6)

```bash
uv run ruff format --check . && uv run ruff check .                        # lint
uv run pytest tests/unit                                                   # no network, no model downloads
REQUIRE_MINIO=1 uv run pytest -m integration tests/integration tests/e2e   # needs step 2
uv run python scripts/check_hygiene.py                                     # fails if secrets, models or caches are tracked
```

Expected: all pass. `test-integration` includes:
- the bucket test (AC-03, ADR-0002): asserts that `zeroshot_learning_output` is rejected
  and the default `zeroshot-learning-output` is created;
- the E2E persistence test (Quality Gate 4): prediction service, fake classifier and real
  MinIO, then the stored object is checked against
  [prediction-record.schema.json](contracts/prediction-record.schema.json) and its key
  format `predictions/YYYY/MM/DD/<uuid>.json`.

## 4. Start the application

```bash
uv run streamlit run src/zeroshot/ui/app.py --server.port 8502
# or fully containerized:
mkdir -p models .cache/huggingface data && APP_UID=$(id -u) APP_GID=$(id -g) docker compose --profile app up -d --build
```

Open http://localhost:8502.

## 5. Administration walkthrough

| Step | Action | Expected | AC |
|------|--------|----------|----|
| 5.1 | Open **Administration** | Both models are listed as `Not available`; `MODEL_DIR` and `HF_HOME` are shown separately; the "local/trusted use only" banner is visible | AC-04 |
| 5.2 | Select **DeBERTa Zero-Shot v2** and **BART MNLI**, then click **Download selected** | Each shows its own status, then `Ready`; `models/<slug>/<sha>/.complete.json` exists for each | AC-05, AC-06 |
| 5.3 | Click **Download selected** for DeBERTa again | The outcome is `already present`, there is no network transfer, and the log shows `model.download.skipped` | AC-07 |
| 5.4 | Create the label set `news` with the labels `sports`, `Sports`, `politics` | Rejected: duplicate label "Sports" | AC-16 |
| 5.5 | Fix it to `sports, politics, science` and save; restart the app | `news` is still listed | AC-10 |

## 6. Classification walkthrough

| Step | Action | Expected | AC |
|------|--------|----------|----|
| 6.1 | Open **Classification** | Only `Ready` models are offered; the label sets `default`, `extended` and `news` are offered | AC-11 |
| 6.2 | Classify "The team won the championship after a very close game." with DeBERTa + `default` | ✅ Saved; predicted class `sports`; 4 labels sorted by score; model, revision, UTC timestamp, request ID and object key are shown | AC-12, AC-13 |
| 6.3 | Submit empty text, then 5001 characters | Rejected before inference, with messages naming the problem and the limit | AC-16 |
| 6.4 | `docker compose stop minio`, then classify again | ⚠️ "Prediction produced but **not persisted**"; "Saved" is never shown | AC-18 |
| 6.5 | `docker compose up -d --wait minio` and classify again | ✅ Saved | AC-14 |

## 7. Verify the stored JSON (AC-14, AC-15)

Console: bucket `zeroshot-learning-output`, then `predictions/<today UTC>/`, then open
`<request_id>.json`.

CLI alternative:

```bash
uv run python -m zeroshot.tools.show_latest
```

Expected: the JSON matches the schema. `model_revision` is
`8e7e5af5983a0ddb1a5b45a38b129ab69e2258e8`, `inference_parameters.hypothesis_template` is
`"This text is about {}"`, `inference_parameters.device` is `"cpu"` (unless you opted
into `INFERENCE_DEVICE=mps`), `label_set` is `"default"`, and the request ID equals the one in
the UI and in the logs (`grep <request_id>` on the app logs shows `prediction.persisted`).
The logs contain `input_chars` but not the text.

## 8. Lifecycle operations (AC-08, AC-09)

| Step | Action | Expected |
|------|--------|----------|
| 8.1 | Start a BART classification on long text and, while it runs, click **Delete** on BART in a second tab | Refused: "model is in use" |
| 8.2 | Delete BART when idle, after confirming | The directory `models/facebook--bart-large-mnli/` is removed; the status becomes `Not available`; `.cache/huggingface/` is untouched; the log shows `pipeline.evicted` |
| 8.3 | Click **Clear HF cache** | The target directories are listed before confirmation; the result is `cleared (X MB)` or `nothing to clear`; DeBERTa is still `Ready` |

## 9. Remove stored predictions (NFR-04)

Delete them in the console, or run `docker compose exec minio mc rm --recursive --force
local/zeroshot-learning-output/predictions/`, or wipe everything with
`docker compose down -v` (this deletes the MinIO volume).

## 10. Optional real-model smoke test

```bash
INFERENCE_DEVICE=cpu uv run pytest -m model tests/model -s   # needs DeBERTa downloaded
```

Expected: the real pipeline classifies the 6.2 sentence as `sports`; latency is printed
(target: SC-006).

## Stop

```bash
docker compose --profile app down   # the MinIO data volume is kept
```
