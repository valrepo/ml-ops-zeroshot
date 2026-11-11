# ADR-0002: Default bucket `zeroshot-learning-output` instead of `zeroshot_learning_output`

- **Status**: **Accepted** by the project owner on 2026-11-11, which constitution v1.1.0
  Principle VI allows (approval by the instructor or the project owner)
- **Owner**: Project owner
- **Date**: 2026-11-11
- **Review date**: if the instructor raises the bucket name in grading feedback
- **Affected principle**: VI (MinIO Is the System of Record); spec FR-09, AC-03; integration
  tests in spec "Testing Requirements"

## Context

The assignment and constitution require the default bucket `zeroshot_learning_output`. The
brief (§1) calls it "the corrected and recommended bucket name", so the instructor expects
this exact string. On
2026-11-11 this was tested against the pinned server image
`pgsty/minio:RELEASE.2026-08-04T00-00-00Z` with the `minio` Python SDK 7.2.20:

```text
zeroshot_learning_output: REJECTED -> S3Error: code: InvalidBucketName,
                          message: The specified bucket is not valid.
zeroshot-learning-output: OK
```

Implementation testing showed that the rejection happens at **two** levels. The server
answers `InvalidBucketName` (seen through `bucket_exists`). The SDK's `make_bucket` refuses
the name before sending any request (`ValueError: invalid bucket name`). The integration
test `tests/integration/test_bucket.py::test_required_underscore_name_is_rejected_by_server`
reproduces both.

S3 bucket-naming rules, which MinIO enforces, allow only lowercase letters, digits, hyphens
and dots, so underscores are invalid. The constitution forbids switching to another object
store.

## Decision

- The default is `MINIO_BUCKET=zeroshot-learning-output`, which is the required name with
  each `_` replaced by `-`.
- The bucket stays configurable. The storage service validates the configured name at
  startup and reports an actionable `ConfigError` that explains the S3 naming rule.
- An integration test asserts that the required name is rejected and that the default is
  accepted, so the evidence is reproducible.

## Rationale

This is the smallest possible change that keeps the intent: the same words, recognizable to
the grader, and a valid name.

## Benefits

The application can actually start and persist predictions, which AC-03 and AC-14 depend on.

## Risks

The grader may check for the exact string. Mitigation: the README states the deviation
prominently, and the integration test reproduces the server's and the SDK's rejection.
Since constitution v1.1.0, Principle VI accepts the project owner's approval, so instructor
confirmation is not required. The owner accepts the residual risk that a grader expects the
exact string.

## Impact on reproducibility

None negative. The name is fixed by default and verified by tests.

## Alternatives considered

- **Use the required name anyway**: impossible, because the server rejects it.
- **Use a single valid bucket with the prefix `zeroshot_learning_output/`**: rejected because
  it changes the required key layout and hides the deviation.
- **Use another S3-compatible server**: rejected because Principle I fixes MinIO and the
  naming rules are the same.

## Confirmation log

| Date | Who | Outcome |
|------|-----|---------|
| 2026-11-11 | Project owner | Accepted. The required name cannot be created on any S3-compatible server, so there is no viable alternative. Approval by the project owner satisfies constitution v1.1.0 Principle VI |
