"""Race-control board custom component wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ModuleNotFoundError:  # pragma: no cover - import-only fallback
    st = None


COMPONENT_DIR = Path(__file__).resolve().parent


def _load_asset(name: str) -> str:
    """Load one local component asset."""
    return (COMPONENT_DIR / name).read_text(encoding="utf-8")


_HTML = _load_asset("race_control_board.html")
_CSS = _load_asset("race_control_board.css")
_JS = _load_asset("race_control_board.js")
_COMPONENT = None


def race_control_board_available() -> bool:
    """Return whether the Streamlit v2 component runtime is available."""
    return bool(
        st is not None
        and hasattr(st, "components")
        and hasattr(st.components, "v2")
        and hasattr(st.components.v2, "component")
    )


def _get_component():
    """Register and memoize the component factory."""
    global _COMPONENT
    if _COMPONENT is None:
        if not race_control_board_available():
            raise RuntimeError("Streamlit Components v2 is not available in this environment.")
        _COMPONENT = st.components.v2.component(
            "race_control_board",
            html=_HTML,
            css=_CSS,
            js=_JS,
        )
    return _COMPONENT


def render_race_control_board(
    payload: dict[str, Any],
    *,
    selected_driver: str,
    key: str,
) -> str:
    """Render the custom race-control board and return the selected driver key."""
    component = _get_component()
    result = component(
        data=payload,
        default={"selected_driver": selected_driver},
        key=key,
        width="stretch",
        height="content",
        on_selected_driver_change=lambda: None,
    )
    return getattr(result, "selected_driver", None) or selected_driver
