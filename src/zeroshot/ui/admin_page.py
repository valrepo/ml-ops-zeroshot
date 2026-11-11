"""Administration page (contracts/ui-pages.md). Presentation only: every action goes
through ``ModelManager`` / ``LabelSetService``; errors are shown via ``user_message``."""

from __future__ import annotations

import streamlit as st

from zeroshot.bootstrap import Services
from zeroshot.domain.labels import LabelSet
from zeroshot.domain.models import ModelInstallation, ModelStatus
from zeroshot.errors import LabelSetError, ZeroShotError


def render(services: Services) -> None:
    st.title("Administration")
    st.warning(
        "No authentication — this page is intended for local development or trusted "
        "environments only.",
        icon="🔓",
    )
    for message in services.startup_errors:
        st.error(message)
    _flash()
    _models_section(services)
    st.divider()
    _label_sets_section(services)


# ---- models ---------------------------------------------------------------------------


def _model_rows(models: list[ModelInstallation]) -> list[dict[str, object]]:
    return [
        {
            "Name": m.definition.name,
            "Hugging Face id": m.definition.repo_id,
            "Revision": m.definition.revision[:12],
            "Licence": m.definition.license or "—",
            "Status": m.status_label,
            "Local path": str(m.local_path) if m.local_path else "—",
            "In use": "yes" if m.in_use else "",
        }
        for m in models
    ]


def _models_section(services: Services) -> None:
    manager = services.model_manager
    st.header("Models")
    models = manager.list_models()
    if not models:
        st.info("No models configured — add one to `configs/models.yaml`.")
        return
    header_left, header_right = st.columns([4, 1])
    with header_right:
        if st.button("Refresh", icon="🔄", width="stretch"):
            st.rerun()
    with header_left:
        ready = sum(m.status is ModelStatus.READY for m in models)
        st.caption(f"{ready} of {len(models)} configured models are ready.")
    st.dataframe(_model_rows(models), hide_index=True, width="stretch")
    _storage_locations(services)
    _model_details(services, models)
    _download_controls(services, models)
    _cache_controls(services)


def _storage_locations(services: Services) -> None:
    paths = services.model_manager.paths()
    with st.expander("Storage locations", icon="📁"):
        st.markdown(
            f"- **Managed model directory (`MODEL_DIR`):** `{paths.model_dir}` — the app's own "
            "copies; *Delete* removes one model from here.\n"
            f"- **Hugging Face home (`HF_HOME`):** `{paths.hf_home}` — library caches; "
            "*Clear HF cache* empties:\n" + "\n".join(f"  - `{t}`" for t in paths.hf_cache_targets)
        )
        if paths.hf_home.is_relative_to(paths.model_dir) or paths.model_dir.is_relative_to(
            paths.hf_home
        ):
            st.warning("MODEL_DIR and HF_HOME are nested inside each other; keep them separate.")


def _model_details(services: Services, models: list[ModelInstallation]) -> None:
    st.subheader("Model details")
    for m in models:
        with st.expander(f"{m.definition.name} — {m.status_label}"):
            d = m.definition
            lines = [
                f"- **Hugging Face id:** `{d.repo_id}`",
                f"- **Revision:** `{d.revision}`",
                f"- **Licence:** {d.license or 'not declared'}",
                f"- **Hypothesis template:** `{d.inference_defaults.hypothesis_template}`, "
                f"multi-label: {d.inference_defaults.multi_label}",
            ]
            if m.marker and m.local_path:
                lines += [
                    f"- **Local path:** `{m.local_path}`",
                    f"- **Downloaded:** {m.marker.downloaded_at}",
                    f"- **Files:** {len(m.marker.files)} ({m.marker.total_bytes / 1e6:.0f} MB)",
                ]
            st.markdown("\n".join(lines))
            if m.in_use:
                st.info("In use by a classification right now.")
            if m.status in (ModelStatus.READY, ModelStatus.INCOMPLETE):
                _delete_model_controls(services, m)


def _delete_model_controls(services: Services, model: ModelInstallation) -> None:
    action = f"delete_model:{model.key}"
    if st.button("Delete…", key=f"delete_{model.key}", icon="🗑️"):
        _ask(action)
    target = services.model_manager.paths().model_dir / model.definition.slug
    prompt = (
        f"Delete **{model.definition.name}** from local storage? This removes "
        f"`{target / model.definition.revision}`. The Hugging Face cache (`HF_HOME`) is "
        "**not** touched; you can download the model again later."
    )
    if _confirmed(action, prompt, "Delete"):
        try:
            services.model_manager.delete(model.key)
        except ZeroShotError as exc:
            _flash_and_rerun("error", exc.user_message)
        _flash_and_rerun("success", f"{model.definition.name} deleted.")


def _cache_controls(services: Services) -> None:
    st.subheader("Hugging Face cache")
    st.caption(
        "Clearing the cache never deletes models under MODEL_DIR; it only empties the library "
        "caches listed under *Storage locations*."
    )
    if st.button("Clear HF cache…", icon="🧹"):
        _ask("clear_cache")
    targets = services.model_manager.paths().hf_cache_targets
    prompt = (
        "The contents of these directories will be removed: "
        + ", ".join(f"`{t}`" for t in targets)
        + ". Downloaded models in `MODEL_DIR` are **not** affected."
    )
    if _confirmed("clear_cache", prompt, "Clear cache"):
        try:
            report = services.model_manager.clear_hf_cache()
        except ZeroShotError as exc:
            _flash_and_rerun("error", exc.user_message)
        if report.status == "cleared":
            _flash_and_rerun("success", f"Cache cleared ({report.removed_bytes / 1e6:.1f} MB).")
        _flash_and_rerun("info", "Nothing to clear — the cache is already empty.")


def _download_controls(services: Services, models: list[ModelInstallation]) -> None:
    candidates = {m.key: m for m in models if m.status is not ModelStatus.READY}
    st.subheader("Download")
    if not candidates:
        st.success("All configured models are downloaded and ready.")
        return
    selected = st.multiselect(
        "Models to download",
        options=list(candidates),
        format_func=lambda key: (
            f"{candidates[key].definition.name} ({candidates[key].status_label})"
        ),
        key="download_selection",
    )
    if not st.button("Download selected", type="primary", disabled=not selected):
        return
    labels = {
        "started": "⏳ downloading…",
        "downloaded": "✅ downloaded",
        "already_present": "✔️ already present",
        "failed": "❌ failed",
    }
    with st.status("Downloading models…", expanded=True) as status:
        outcomes = services.model_manager.download(
            selected,
            on_progress=lambda key, state: st.write(
                f"{candidates[key].definition.name}: {labels[state]}"
            ),
        )
        failed = [k for k, o in outcomes.items() if o.status == "failed"]
        status.update(
            label="Download finished with errors" if failed else "Download finished",
            state="error" if failed else "complete",
        )
    report = []
    for key, outcome in outcomes.items():
        name = candidates[key].definition.name
        if outcome.status == "failed":
            report.append(("error", f"{name}: {outcome.error}"))
        elif outcome.status == "already_present":
            report.append(("info", f"{name}: already downloaded at the configured revision."))
        else:
            size = outcome.marker.total_bytes / 1e6 if outcome.marker else 0
            report.append(("success", f"{name}: downloaded and validated ({size:.0f} MB)."))
    # Rerun so the model table shows the new states; the report is shown once afterwards.
    st.session_state["admin_flash"] = report
    st.rerun()


def _flash() -> None:
    for kind, message in st.session_state.pop("admin_flash", []):
        getattr(st, kind)(message)


def _flash_and_rerun(kind: str, message: str) -> None:
    st.session_state["admin_flash"] = [(kind, message)]
    st.rerun()


# Destructive actions use an explicit two-step confirmation kept in session state
# (works identically in the browser and in headless AppTest runs).
def _ask(action: str) -> None:
    st.session_state["admin_confirm"] = action


def _confirmed(action: str, prompt: str, confirm_label: str) -> bool:
    if st.session_state.get("admin_confirm") != action:
        return False
    with st.container(border=True):
        st.warning(prompt, icon="⚠️")
        yes, no = st.columns(2)
        if yes.button(confirm_label, type="primary", key=f"confirm_{action}"):
            st.session_state.pop("admin_confirm", None)
            return True
        if no.button("Cancel", key=f"cancel_{action}"):
            st.session_state.pop("admin_confirm", None)
            st.rerun()
    return False


# ---- label sets -----------------------------------------------------------------------


def _label_sets_section(services: Services) -> None:
    svc = services.label_sets
    st.header("Candidate-label sets")
    for warning in svc.load_warnings():
        st.warning(warning)
    sets = svc.list()
    if sets:
        st.dataframe(
            [{"Key": s.key, "Name": s.name, "Labels": ", ".join(s.labels)} for s in sets],
            hide_index=True,
            width="stretch",
        )
    else:
        st.info("No label sets yet — create one below.")

    choices = ["➕ New label set", *[s.key for s in sets]]
    target = st.selectbox("Create or edit", choices, key="label_set_target")
    existing = next((s for s in sets if s.key == target), None)
    with st.form("label_set_form", clear_on_submit=False):
        key = st.text_input(
            "Key (lowercase, digits, '-' or '_'; cannot be changed later)",
            value=existing.key if existing else "",
            disabled=existing is not None,
        )
        name = st.text_input("Display name", value=existing.name if existing else "")
        labels_text = st.text_area(
            "Labels — one per line (at least two, no duplicates)",
            value="\n".join(existing.labels) if existing else "",
            height=150,
        )
        submitted = st.form_submit_button("Save label set", type="primary")
    if submitted:
        _save_label_set(services, existing.key if existing else key, name, labels_text)
    if existing is not None:
        action = f"delete_label_set:{existing.key}"
        if st.button(f"Delete '{existing.key}'…", key="delete_label_set"):
            _ask(action)
        prompt = f"Delete the label set **{existing.key}**? Stored predictions are not affected."
        if _confirmed(action, prompt, "Delete label set"):
            services.label_sets.delete(existing.key)
            _flash_and_rerun("success", f"Label set '{existing.key}' deleted.")


def _save_label_set(services: Services, key: str, name: str, labels_text: str) -> None:
    labels = [line for line in labels_text.splitlines() if line.strip()]  # skip blank lines
    try:
        saved = services.label_sets.save(LabelSet(key=key, name=name, labels=labels))
    except LabelSetError as exc:
        st.error(exc.user_message)
        return
    _flash_and_rerun("success", f"Label set '{saved.key}' saved ({len(saved.labels)} labels).")
