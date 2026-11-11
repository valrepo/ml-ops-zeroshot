"""Streamlit entry point: ``streamlit run src/zeroshot/ui/app.py``.

Presentation only — every action goes through the services built by ``build_services``.
"""

from __future__ import annotations

import streamlit as st

from zeroshot.bootstrap import Services, build_services
from zeroshot.config import load_settings
from zeroshot.errors import ConfigError
from zeroshot.ui import admin_page, classify_page

st.set_page_config(page_title="Zero-Shot Classifier", layout="wide")


@st.cache_resource(show_spinner="Starting services…")
def get_services() -> Services:
    return build_services(load_settings())


try:
    services = get_services()
except ConfigError as exc:
    st.error(f"Configuration error: {exc.user_message}")
    st.info("Copy `.env.example` to `.env`, fix the value above, then restart the app.")
    st.stop()

navigation = st.navigation(
    [
        st.Page(
            lambda: admin_page.render(services), title="Administration", icon="🛠️", url_path="admin"
        ),
        st.Page(
            lambda: classify_page.render(services),
            title="Classification",
            icon="🔎",
            url_path="classify",
            default=True,
        ),
    ]
)
navigation.run()
