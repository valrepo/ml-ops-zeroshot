# ADR-0003: MinIO server image `pgsty/minio` and default host ports 9010/9011

- **Status**: Accepted
- **Owner**: Project owner
- **Date**: 2026-11-11
- **Review date**: 2027-05-11, or as soon as an official MinIO image is published again
- **Affected principle**: I (Fixed Technology Stack, MinIO server component); Technology
  Baseline ("MinIO server image: explicitly pinned, never `latest`"); brief §7 example
  `docker-compose.yml`

## Context

The brief's example Compose file uses `image: minio/minio` (with no tag, which means
`latest`) and host ports `9000:9000` and `9001:9001`.

- The official `minio/minio` images on Docker Hub have not been available since October
  2025. `pgsty/minio` is a maintained redistribution of the MinIO server. Its tag
  `RELEASE.2026-08-04T00-00-00Z` is published for amd64 and arm64 and was checked on
  2026-11-11 (research R2).
- The constitution forbids `latest`, so a pinned tag is required in any case.
- On the reference development machine, host ports 9000, 9001 and 8501 are already in use
  (research R12).

## Decision

- Use `pgsty/minio:RELEASE.2026-08-04T00-00-00Z`. The server is still MinIO, so this changes
  only where the image comes from, not the storage technology.
- Default host ports: `MINIO_API_PORT=9010` and `MINIO_CONSOLE_PORT=9011`, plus
  `APP_PORT=8502` for Streamlit. All three are configurable through `.env`. Inside the
  Compose network, MinIO still listens on 9000/9001, so `MINIO_ENDPOINT=minio:9000` from the
  brief's `.env.example` stays valid for the containerized app.

## Rationale

- An image that can actually be pulled and is pinned is required for reproducibility
  (Principle III and Quality Gate 7).
- Non-default ports avoid start-up failures on machines where common ports are taken.
  Evaluators who prefer the brief's ports can set `MINIO_API_PORT=9000` and
  `MINIO_CONSOLE_PORT=9001`.

## Benefits

A Compose stack that starts reliably on arm64 and amd64, with a pinned server version.

## Risks

- A community redistribution could lag behind upstream security fixes. Mitigation: pin the
  tag, review it on the review date, and keep the stack local or trusted only.
- Graders following the brief may open `localhost:9001`. Mitigation: the README Quick start
  and Deviations sections state 9011, and the ports are configurable.

## Impact on reproducibility

Positive. The server version is fixed by tag, and integration tests in CI run against this
exact image.
