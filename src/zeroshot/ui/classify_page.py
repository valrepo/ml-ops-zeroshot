"""Classification page (contracts/ui-pages.md). Presentation only: classification goes
exclusively through ``PredictionService.predict`` (FR-08, FR-12)."""

from __future__ import annotations

import streamlit as st

from zeroshot.bootstrap import Services
from zeroshot.domain.predictions import ClassificationResult, OutcomeStatus, PredictionOutcome


def render(services: Services) -> None:
    st.title("Classification")
    settings = services.settings
    for message in services.startup_errors:
        st.error(message)

    ready = services.model_manager.ready_models()
    label_sets = services.label_sets.list()
    if not ready:
        st.warning(
            "No model is ready — an administrator must download one on the Administration page."
        )
    if not label_sets:
        st.warning("No valid label set is available — create one on the Administration page.")

    text = st.text_area(
        "Text to classify",
        height=180,
        max_chars=None,
        placeholder="The team won the championship after a very close game.",
        key="classify_text",
    )
    over = len(text) > settings.max_input_chars
    st.caption(
        f"{len(text)} / {settings.max_input_chars} characters"
        + (" — over the limit" if over else "")
    )

    left, right = st.columns(2)
    with left:
        model_key = st.selectbox(
            "Model",
            options=[m.key for m in ready],
            format_func=lambda key: next(m.definition.name for m in ready if m.key == key),
            disabled=not ready,
            key="classify_model",
        )
    with right:
        label_set_key = st.selectbox(
            "Candidate-label set",
            options=[s.key for s in label_sets],
            format_func=lambda key: next(s.name for s in label_sets if s.key == key),
            disabled=not label_sets,
            key="classify_label_set",
        )
    if label_set_key:
        chosen = next(s for s in label_sets if s.key == label_set_key)
        st.caption("Labels: " + ", ".join(chosen.labels))

    clicked = st.button(
        "Classify", type="primary", disabled=not (ready and label_sets), key="classify_button"
    )
    if clicked:
        with st.spinner("Classifying… (the first request loads the model)"):
            outcome = services.prediction.predict(text, model_key, label_set_key)
        st.session_state["last_outcome"] = outcome
    outcome = st.session_state.get("last_outcome")
    if outcome is not None:
        _render_outcome(outcome)


def _render_outcome(outcome: PredictionOutcome) -> None:
    st.divider()
    if outcome.status is OutcomeStatus.FAILED:
        st.error(outcome.error.user_message if outcome.error else "Classification failed.")
        st.caption(f"Request ID: `{outcome.request_id}`")
        return
    if outcome.status is OutcomeStatus.SUCCEEDED:
        st.success("Saved to MinIO.", icon="✅")
    else:  # COMPUTED_NOT_PERSISTED — never shown as a success (FR-08, AC-18)
        st.warning(
            "Prediction produced but **not persisted** — storage error: "
            f"{outcome.error.user_message if outcome.error else 'unknown'}",
            icon="⚠️",
        )
    _render_result(outcome, saved=outcome.status is OutcomeStatus.SUCCEEDED)


def _render_result(outcome: PredictionOutcome, *, saved: bool) -> None:
    result: ClassificationResult = outcome.result
    record = outcome.record
    title = "Result" if saved else "Result (not saved)"
    st.subheader(title)
    c1, c2 = st.columns(2)
    c1.metric("Predicted class", result.predicted_class)
    c2.metric("Confidence", f"{result.confidence:.1%}")
    st.dataframe(
        [{"Rank": i, "Label": p.label, "Score": p.score} for i, p in enumerate(result.ranked, 1)],
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score", min_value=0.0, max_value=1.0, format="%.4f"
            )
        },
    )
    details = {
        "Model": f"{record.model_name} (`{record.model}`)",
        "Revision": f"`{record.model_revision}`",
        "Timestamp (UTC)": record.model_dump(mode="json")["timestamp"],
        "Request ID": f"`{outcome.request_id}`",
        "Latency": f"{result.latency_ms:.0f} ms on {result.inference_parameters.device}",
    }
    if outcome.object_key:
        details["Object key"] = f"`{outcome.object_key}`"
    st.markdown("\n".join(f"- **{k}:** {v}" for k, v in details.items()))
