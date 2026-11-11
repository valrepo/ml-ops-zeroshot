# ADR-0001: Use `HF_HOME` instead of `TRANSFORMERS_CACHE`

- **Status**: Accepted
- **Owner**: Project owner
- **Date**: 2026-11-11
- **Review date**: 2027-05-11, or on the next Transformers major-version upgrade
- **Affected principle**: IV (Configuration and Secrets); spec FR-04

## Context

The assignment's example environment lists `TRANSFORMERS_CACHE`. Transformers has deprecated
that variable in favor of the Hugging Face home directory `HF_HOME`, which the pinned
`huggingface_hub` 2.4.0 reads (checked on 2026-11-11). Relying on a deprecated variable risks
silent behavior changes when the library is upgraded.

## Decision

The application configures only `HF_HOME`. `TRANSFORMERS_CACHE` is not read. If it is set,
the app logs a startup warning saying it is ignored. `.env.example` mentions it only as a
commented-out, deprecated line. The README explains this deviation. Model weights live in
`MODEL_DIR`; `HF_HOME` holds only the library's caches and token (see plan research R6).

## Rationale

- Uses the supported configuration (constitution Principle IV).
- One clearly named variable for the cleanup target makes "Clear HF cache" unambiguous.

## Benefits

Future-proof against removal of the deprecated variable; clearer separation of model storage
from cache.

## Risks

An evaluator who sets only `TRANSFORMERS_CACHE` from the assignment example may expect it to
apply. Mitigation: the startup warning and the README note.

## Impact on reproducibility

None negative. Cache location is fully determined by `HF_HOME`, which is documented and
defaulted.
