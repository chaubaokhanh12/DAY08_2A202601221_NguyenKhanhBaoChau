# UniAssist Streamlit UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the interface in `app.py` with the approved UniAssist Academic Minimalism design while preserving the existing RAG chat behavior.

**Architecture:** Keep Streamlit as the only UI runtime and retain its native session state, buttons, slider, chat messages, and chat input. Reorganize `app.py` into small rendering helpers, inject a single responsive CSS theme, and keep the call to `generate_with_citation(query, top_k=top_k)` unchanged.

**Tech Stack:** Python 3.11+, Streamlit 1.35+, HTML/CSS injected through `st.markdown`, pytest.

## Global Constraints

- Modify presentation and interaction behavior only; do not alter retrieval, ranking, generation, or evaluation modules.
- Preserve `generate_with_citation(query, top_k=top_k)`, chat history, suggested queries, source collections, and friendly error messages.
- Use Inter with system sans-serif fallback, `#f9f9f9` surfaces, `#002b5a` primary navy, and `#006a65` AI/status teal.
- Keep the conversation centered at a maximum width of 800px.
- Keep `top_k` in the inclusive range 3 through 10 with a default of 5.
- Do not depend on externally hosted images from the reference HTML.
- Preserve the user's existing uncommitted `README.md` changes.

## File Structure

- Modify `app.py`: page setup, CSS theme, session state, UniAssist header, empty state, chat thread, source cards, suggestions, settings, composer, and RAG query flow.
- Create `tests/test_app_ui.py`: source-level UI contract tests that do not execute Streamlit or require external services.

---

### Task 1: Establish the UniAssist UI Contract

**Files:**
- Create: `tests/test_app_ui.py`
- Test: `tests/test_app_ui.py`

**Interfaces:**
- Consumes: UTF-8 source text from `PROJECT_ROOT / "app.py"`.
- Produces: regression checks for required branding, palette, layout hooks, Streamlit widgets, RAG call, and error paths.

- [ ] **Step 1: Write failing source-contract tests**

```python
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
APP_SOURCE = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")


def test_uniassist_brand_and_palette_are_present():
    assert "UniAssist AI" in APP_SOURCE
    assert "#002b5a" in APP_SOURCE
    assert "#006a65" in APP_SOURCE
    assert "max-width: 800px" in APP_SOURCE


def test_custom_shell_replaces_default_sidebar():
    assert "st.sidebar" not in APP_SOURCE
    assert "uniassist-header" in APP_SOURCE
    assert "uniassist-welcome" in APP_SOURCE
    assert "initial_sidebar_state=\"collapsed\"" in APP_SOURCE


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
```

- [ ] **Step 2: Run the tests and confirm the old UI fails the contract**

Run: `python -m pytest tests/test_app_ui.py -v`

Expected: failures for missing `UniAssist AI`, the new palette/layout hooks, collapsed sidebar, welcome state, and dedicated source renderer.

- [ ] **Step 3: Commit the failing contract tests**

```powershell
git add -- tests/test_app_ui.py
git commit -m "test: define UniAssist UI contract"
```

---

### Task 2: Replace the Streamlit Shell and Conversation Presentation

**Files:**
- Modify: `app.py:1-151`
- Test: `tests/test_app_ui.py`

**Interfaces:**
- Consumes: `st.session_state.messages: list[dict]`, where each item has `role`, `content`, and optional `sources`.
- Produces: `render_sources(sources: list[dict]) -> None` and `render_message(message: dict) -> None`, both writing to the active Streamlit page.

- [ ] **Step 1: Replace page configuration and initialize state before widgets**

Set `page_title="UniAssist AI"`, use a local Unicode graduation-cap page icon, retain `layout="wide"`, set `initial_sidebar_state="collapsed"`, and initialize `messages`, `pending_query`, and `top_k` before rendering interactive controls.

- [ ] **Step 2: Add the responsive Academic Minimalism CSS theme**

Add one `st.markdown(..., unsafe_allow_html=True)` block that:

```css
:root {
  --surface: #f9f9f9;
  --surface-lowest: #ffffff;
  --surface-low: #f3f3f4;
  --primary: #002b5a;
  --secondary: #006a65;
  --text: #1a1c1c;
  --muted: #424750;
  --border: #e2e2e2;
}
.block-container { max-width: 800px; padding-top: 6rem; padding-bottom: 10rem; }
.uniassist-header { position: fixed; inset: 0 0 auto 0; height: 64px; z-index: 999; }
.source-card { border-left: 4px solid var(--primary); border-radius: 16px; }
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .uniassist-nav, .uniassist-profile-label { display: none; }
}
```

The complete block must also hide Streamlit chrome/sidebar, style user and assistant chat containers, create pill-shaped buttons, elevate the chat input, style expanders/sliders, and preserve horizontal scrolling for suggestions on narrow screens.

- [ ] **Step 3: Render the fixed UniAssist header and empty state**

Use inline SVG/CSS shapes or Unicode for the brand mark so no remote image is required. Render a fixed header containing the UniAssist AI wordmark, Home label, theme icon, and profile avatar. When `messages` is empty, render `.uniassist-welcome` with a concise university-services introduction.

- [ ] **Step 4: Implement safe source and message renderers**

Implement:

```python
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.markdown('<div class="sources-heading">▤ Sources</div>', unsafe_allow_html=True)
    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata") or {}
        source_name = str(metadata.get("source") or source.get("source") or "Unknown source")
        document_type = str(metadata.get("type") or "document")
        score = float(source.get("score") or 0.0)
        st.markdown(
            f'<div class="source-card"><strong>{index}. {html.escape(source_name)}</strong>'
            f'<span>{html.escape(document_type)} · score {score:.4f}</span></div>',
            unsafe_allow_html=True,
        )
        excerpt = str(source.get("content") or "")[:500]
        if excerpt:
            with st.expander(f"View excerpt {index}"):
                st.text(excerpt)


def render_message(message: dict) -> None:
    role = message.get("role", "assistant")
    with st.chat_message(role, avatar="🎓" if role == "assistant" else None):
        st.markdown(str(message.get("content", "")))
        if role == "assistant":
            render_sources(message.get("sources") or [])
```

Import Python's `html` module so source metadata inserted into custom HTML is escaped.

- [ ] **Step 5: Render history through the helpers**

Replace duplicate history/source rendering with:

```python
for message in st.session_state.messages:
    render_message(message)
```

- [ ] **Step 6: Run the focused UI contract tests**

Run: `python -m pytest tests/test_app_ui.py -v`

Expected: all contract tests pass except any interaction assertion completed in Task 3.

- [ ] **Step 7: Commit the shell and conversation presentation**

```powershell
git add -- app.py tests/test_app_ui.py
git commit -m "feat: recreate UniAssist conversation interface"
```

---

### Task 3: Restore Settings, Suggestions, Composer, and RAG Flow

**Files:**
- Modify: `app.py`
- Test: `tests/test_app_ui.py`

**Interfaces:**
- Consumes: a typed chat value or `st.session_state.pending_query`, plus `st.session_state.top_k`.
- Produces: appended user and assistant message dictionaries and a call to `generate_with_citation(query, top_k=top_k)`.

- [ ] **Step 1: Add compact settings and suggestion controls**

Place the retrieval slider in `st.expander("Retrieval settings")` with `min_value=3`, `max_value=10`, and `key="top_k"`. Render four short suggestion buttons in columns: Course registration, Tuition deadline, Library booking, and IT Support desk. Each button assigns a complete English university-services question to `pending_query`.

- [ ] **Step 2: Add the native chat composer and disclaimer**

Use:

```python
user_input = st.chat_input("Ask anything about your university...")
query = user_input or st.session_state.pending_query
st.caption("UniAssist AI can make mistakes. Check important institutional info.")
```

Apply the composer styling from Task 2 without replacing `st.chat_input` with raw HTML.

- [ ] **Step 3: Preserve and simplify query processing**

Clear `pending_query`, append the user message, render it, then call the existing generator under an assistant chat message and spinner. Retain these exact behaviors:

```python
from src.task10_generation import generate_with_citation

response = generate_with_citation(query, top_k=top_k)
answer = response.get("answer", "Unable to answer this question.")
sources = response.get("sources", [])
```

Catch `NotImplementedError` with student guidance and catch `Exception as error` with a concise pipeline message. Render sources using `render_sources(sources)` and append `{"role": "assistant", "content": answer, "sources": sources}` to session history.

- [ ] **Step 4: Run the focused UI contract tests**

Run: `python -m pytest tests/test_app_ui.py -v`

Expected: all tests pass.

- [ ] **Step 5: Verify Python syntax without importing unavailable dependencies**

Run:

```powershell
python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text(encoding='utf-8')); print('app.py syntax OK')"
```

Expected: `app.py syntax OK`.

- [ ] **Step 6: Run the existing project test suite**

Run: `python -m pytest tests -v`

Expected: the new UI contract passes; existing lab tests pass or report their pre-existing dependency/data skips without a new failure caused by `app.py`.

- [ ] **Step 7: Launch Streamlit for visual verification when the environment is available**

Run: `python -m streamlit run app.py --server.headless true`.

Verify the empty state, a user message, an assistant response, source cards, the retrieval setting, suggestion submission, error output, and a viewport narrower than 640px. If the current virtual environment remains broken, report that constraint and provide the exact launch command after recreating the environment.

- [ ] **Step 8: Commit the completed interaction flow**

```powershell
git add -- app.py tests/test_app_ui.py
git commit -m "feat: complete UniAssist chat interactions"
```

---

### Task 4: Final Regression Review

**Files:**
- Review: `app.py`
- Review: `tests/test_app_ui.py`
- Review: `docs/superpowers/specs/2026-08-04-uniassist-streamlit-ui-design.md`

**Interfaces:**
- Consumes: completed implementation and verification output.
- Produces: evidence that the implementation matches the approved design without changing the RAG contract.

- [ ] **Step 1: Compare implementation against every design section**

Confirm header, 800px conversation layout, empty state, user/assistant differentiation, source cards, suggestions, bottom composer, compact `top_k`, responsive CSS, safe source defaults, and the three non-goals are all respected.

- [ ] **Step 2: Inspect the final diff and unrelated working-tree changes**

Run: `git diff --check` and `git status --short`.

Expected: no whitespace errors; the user's pre-existing `README.md` modification remains separate.

- [ ] **Step 3: Run final verification commands once more**

Run: `python -m pytest tests/test_app_ui.py -v` and the AST syntax command from Task 3.

Expected: all focused tests pass and syntax parsing succeeds.
