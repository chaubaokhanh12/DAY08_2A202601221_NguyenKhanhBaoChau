"""Source-level contracts for the Streamlit interface.

These checks intentionally avoid importing ``app.py`` because importing a
Streamlit entry point executes the application and may initialize the RAG
stack during test collection.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
APP_SOURCE = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")


def test_uniassist_brand_and_palette_are_present():
    assert "UniAssist AI" in APP_SOURCE
    assert "#002b5a" in APP_SOURCE
    assert "#006a65" in APP_SOURCE
    assert "max-width: 800px" in APP_SOURCE


def test_retrieval_settings_live_in_collapsible_sidebar():
    assert "with st.sidebar" in APP_SOURCE
    assert "sidebar-retrieval" in APP_SOURCE
    assert "uniassist-header" in APP_SOURCE
    assert "uniassist-welcome" in APP_SOURCE
    assert 'initial_sidebar_state="expanded"' in APP_SOURCE
    assert 'min_value=3' in APP_SOURCE
    assert 'max_value=10' in APP_SOURCE


def test_native_streamlit_interactions_and_rag_call_remain():
    assert "st.chat_input" in APP_SOURCE
    assert "st.chat_message" in APP_SOURCE
    assert "st.slider" in APP_SOURCE
    assert "generate_with_citation(query, top_k=top_k)" in APP_SOURCE
    assert "except NotImplementedError" in APP_SOURCE
    assert "except Exception as" in APP_SOURCE


def test_sources_have_a_dedicated_renderer():
    assert "def render_sources(" in APP_SOURCE
    assert "source-card" in APP_SOURCE
    assert "metadata" in APP_SOURCE
