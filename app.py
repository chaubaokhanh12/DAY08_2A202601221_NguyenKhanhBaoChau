"""UniAssist AI — Streamlit interface for the University Services RAG app."""

import html
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="UniAssist AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --surface: #f9f9f9;
        --surface-lowest: #ffffff;
        --surface-low: #f3f3f4;
        --surface-high: #e8e8e8;
        --primary: #002b5a;
        --primary-container: #004182;
        --primary-fixed: #d6e3ff;
        --secondary: #006a65;
        --text: #1a1c1c;
        --muted: #424750;
        --border: #e2e2e2;
        --error: #ba1a1a;
    }

    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--surface);
    }

    #MainMenu, footer {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    [data-testid="collapsedControl"] {
        top: 72px;
        left: 12px;
        z-index: 1001;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: var(--surface-lowest);
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.08);
    }

    [data-testid="stSidebar"] {
        top: 64px;
        height: calc(100vh - 64px);
        border-right: 1px solid var(--border);
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 8px 0 30px rgba(0, 43, 90, 0.045);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.15rem 2rem;
    }

    .sidebar-retrieval {
        margin: 0.25rem 0 1.2rem;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(0, 106, 101, 0.16);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(214, 227, 255, 0.58), rgba(111, 247, 238, 0.13));
    }

    .sidebar-kicker {
        margin-bottom: 5px;
        color: var(--secondary);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-retrieval h2 {
        margin: 0;
        color: var(--primary);
        font-size: 20px;
        line-height: 1.3;
        letter-spacing: -0.02em;
    }

    .sidebar-retrieval p {
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.55;
    }

    .pipeline-steps {
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border);
    }

    .pipeline-step {
        display: flex;
        align-items: center;
        gap: 9px;
        margin: 9px 0;
        color: var(--muted);
        font-size: 12px;
    }

    .pipeline-dot {
        width: 8px;
        height: 8px;
        flex: 0 0 8px;
        border-radius: 999px;
        background: var(--secondary);
        box-shadow: 0 0 0 4px rgba(0, 106, 101, 0.09);
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0;
    }

    .block-container {
        max-width: 800px;
        padding-top: 6rem;
        padding-bottom: 10rem;
    }

    .uniassist-header {
        position: fixed;
        inset: 0 0 auto 0;
        z-index: 999;
        height: 64px;
        display: flex;
        align-items: center;
        background: rgba(255, 255, 255, 0.92);
        border-bottom: 1px solid rgba(226, 226, 226, 0.72);
        box-shadow: 0 1px 8px rgba(0, 0, 0, 0.04);
        backdrop-filter: blur(14px);
    }

    .uniassist-header-inner {
        width: min(100% - 48px, 1248px);
        margin: 0 auto;
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
    }

    .uniassist-brand {
        display: flex;
        align-items: center;
        gap: 11px;
        color: var(--primary);
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    .uniassist-logo {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        background: linear-gradient(145deg, var(--primary-container), var(--primary));
        box-shadow: 0 6px 16px rgba(0, 43, 90, 0.18);
    }

    .uniassist-nav {
        color: var(--primary);
        font-size: 14px;
        font-weight: 600;
    }

    .uniassist-tools {
        justify-self: end;
        display: flex;
        align-items: center;
        gap: 14px;
        color: var(--muted);
    }

    .theme-icon {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border-radius: 999px;
        font-size: 17px;
    }

    .profile-avatar {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        border: 1px solid var(--border);
        border-radius: 999px;
        color: var(--primary);
        background: var(--primary-fixed);
        font-size: 13px;
        font-weight: 700;
    }

    .uniassist-welcome {
        margin: min(14vh, 120px) auto 44px;
        text-align: center;
        animation: welcome-in 0.5s ease-out;
    }

    .welcome-mark {
        width: 56px;
        height: 56px;
        margin: 0 auto 20px;
        display: grid;
        place-items: center;
        color: var(--primary);
        background: var(--primary-fixed);
        border-radius: 18px;
        font-size: 26px;
        box-shadow: 0 8px 28px rgba(0, 43, 90, 0.10);
    }

    .uniassist-welcome h1 {
        margin: 0 0 10px;
        color: var(--primary);
        font-size: clamp(28px, 4vw, 38px);
        line-height: 1.2;
        letter-spacing: -0.035em;
    }

    .uniassist-welcome p {
        max-width: 580px;
        margin: 0 auto;
        color: var(--muted);
        font-size: 16px;
        line-height: 1.65;
    }

    @keyframes welcome-in {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stExpander"] {
        margin-bottom: 1.25rem;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.74);
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.025);
    }

    [data-testid="stExpander"] summary {
        color: var(--muted);
        font-size: 13px;
        font-weight: 600;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--primary-container);
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 1.3rem;
        padding: 0;
        background: transparent;
    }

    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarUser"] {
        display: none;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        width: fit-content;
        max-width: 78%;
        margin-left: auto;
        padding: 0.85rem 1.15rem;
        border: 1px solid rgba(195, 198, 210, 0.55);
        border-radius: 18px 5px 18px 18px;
        background: var(--surface-low);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.035);
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        color: var(--primary) !important;
        background: var(--primary-fixed) !important;
        border: 1px solid rgba(169, 199, 255, 0.55);
    }

    [data-testid="stChatMessageContent"] {
        color: var(--text);
        font-size: 16px;
        line-height: 1.7;
    }

    [data-testid="stChatMessageContent"] p {
        margin-bottom: 0.75rem;
    }

    .ai-status {
        width: fit-content;
        margin: 0 0 12px;
        padding: 6px 10px;
        color: var(--secondary);
        background: rgba(111, 247, 238, 0.22);
        border-radius: 999px;
        font-size: 12px;
        font-weight: 600;
    }

    .sources-heading {
        margin: 22px 0 10px;
        padding-top: 16px;
        border-top: 1px solid var(--border);
        color: var(--muted);
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 0.01em;
    }

    .source-card {
        position: relative;
        margin: 0 0 8px;
        padding: 12px 14px 12px 16px;
        overflow: hidden;
        border: 1px solid var(--border);
        border-left: 4px solid var(--primary);
        border-radius: 14px;
        background: var(--surface-lowest);
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.035);
    }

    .source-card:nth-of-type(even) {
        border-left-color: var(--secondary);
    }

    .source-card strong,
    .source-card span {
        display: block;
    }

    .source-card strong {
        padding-right: 10px;
        color: var(--text);
        font-size: 13px;
        font-weight: 600;
    }

    .source-card span {
        margin-top: 3px;
        color: var(--muted);
        font-size: 11px;
    }

    .assistant-actions {
        display: flex;
        gap: 6px;
        margin-top: 10px;
        color: #737781;
        font-size: 15px;
        letter-spacing: 8px;
        user-select: none;
    }

    .suggestion-label {
        margin: 24px 0 9px;
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
    }

    [data-testid="stButton"] button {
        min-height: 40px;
        border: 1px solid transparent;
        border-radius: 999px;
        color: var(--text);
        background: var(--surface-low);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.035);
        font-size: 12px;
        font-weight: 500;
        transition: border-color 0.18s ease, background 0.18s ease, transform 0.18s ease;
    }

    [data-testid="stButton"] button:hover {
        border-color: rgba(0, 106, 101, 0.28);
        color: var(--primary);
        background: #edf6f5;
        transform: translateY(-1px);
    }

    [data-testid="stChatInput"] {
        border: 1px solid var(--border);
        border-radius: 17px;
        background: var(--surface-lowest);
        box-shadow: 0 6px 30px rgba(0, 43, 90, 0.09);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: rgba(0, 106, 101, 0.55);
        box-shadow: 0 7px 32px rgba(0, 106, 101, 0.11);
    }

    [data-testid="stChatInput"] button {
        color: white;
        border-radius: 11px;
        background: var(--primary);
    }

    [data-testid="stBottom"] {
        background: linear-gradient(to top, var(--surface) 70%, rgba(249, 249, 249, 0));
    }

    .uniassist-disclaimer {
        margin: 8px 0 0;
        color: #737781;
        text-align: center;
        font-size: 10px;
    }

    @media (max-width: 640px) {
        .block-container {
            padding: 5.25rem 1rem 9rem;
        }

        .uniassist-header-inner {
            width: calc(100% - 30px);
            grid-template-columns: 1fr auto;
        }

        .uniassist-nav,
        .theme-icon,
        .uniassist-profile-label {
            display: none;
        }

        .uniassist-welcome {
            margin-top: 8vh;
        }

        [data-testid="stSidebar"] {
            width: min(86vw, 320px) !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            max-width: 88%;
        }

        [data-testid="column"] {
            min-width: max-content !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <div class="uniassist-header">
        <div class="uniassist-header-inner">
            <div class="uniassist-brand">
                <span class="uniassist-logo">U</span>
                <span>UniAssist AI</span>
            </div>
            <div class="uniassist-nav">Home</div>
            <div class="uniassist-tools">
                <span class="theme-icon" aria-label="Theme">☾</span>
                <span class="profile-avatar" aria-label="Profile">UA</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        """
        <section class="sidebar-retrieval">
            <div class="sidebar-kicker">Knowledge controls</div>
            <h2>Retrieval Settings</h2>
            <p>Control how much evidence UniAssist collects before composing an answer.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Sources to retrieve",
        min_value=3,
        max_value=10,
        key="top_k",
        help="Choose how many knowledge-base chunks are sent to the answer generator.",
    )
    st.caption(f"Using the top **{top_k}** knowledge chunks")
    st.markdown(
        """
        <div class="pipeline-steps">
            <div class="sidebar-kicker">Retrieval flow</div>
            <div class="pipeline-step"><span class="pipeline-dot"></span>Semantic + BM25 search</div>
            <div class="pipeline-step"><span class="pipeline-dot"></span>RRF reranking</div>
            <div class="pipeline-step"><span class="pipeline-dot"></span>PageIndex fallback</div>
            <div class="pipeline-step"><span class="pipeline-dot"></span>Citation-aware answer</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict]) -> None:
    """Render retrieved chunks as compact, safely escaped source cards."""
    if not sources:
        return

    st.markdown(
        '<div class="sources-heading">▤ &nbsp; Sources</div>',
        unsafe_allow_html=True,
    )
    for index, source in enumerate(sources, start=1):
        metadata = source.get("metadata") or {}
        source_name = str(
            metadata.get("source") or source.get("source") or "Unknown source"
        )
        document_type = str(metadata.get("type") or "document")
        try:
            score = float(source.get("score") or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        st.markdown(
            '<div class="source-card">'
            f"<strong>{index}. {html.escape(source_name)}</strong>"
            f"<span>{html.escape(document_type)} &nbsp;·&nbsp; score {score:.4f}</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        excerpt = str(source.get("content") or "")[:500]
        if excerpt:
            with st.expander(f"View excerpt {index}"):
                st.text(excerpt)


def render_message(message: dict) -> None:
    """Render one persisted chat message and its optional sources."""
    role = message.get("role", "assistant")
    avatar = "🎓" if role == "assistant" else None
    with st.chat_message(role, avatar=avatar):
        st.markdown(str(message.get("content", "")))
        if role == "assistant":
            render_sources(message.get("sources") or [])
            st.markdown(
                '<div class="assistant-actions" title="Response actions">⧉ ↻ ♡</div>',
                unsafe_allow_html=True,
            )


if not st.session_state.messages:
    st.markdown(
        """
        <section class="uniassist-welcome">
            <div class="welcome-mark">✦</div>
            <h1>How can I help today?</h1>
            <p>
                Ask UniAssist about course registration, tuition, scholarships,
                library services, accommodation, and student support at RMIT Vietnam.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


for persisted_message in st.session_state.messages:
    render_message(persisted_message)


st.markdown(
    '<div class="suggestion-label">Popular questions</div>',
    unsafe_allow_html=True,
)

suggestions = [
    ("Course registration", "What are the key dates for course registration?"),
    ("Tuition deadline", "When is the tuition payment deadline?"),
    ("Library booking", "How can I book a group study room in the library?"),
    ("IT Support desk", "How can I contact the IT Support desk?"),
]

suggestion_columns = st.columns(len(suggestions))
for column, (label, question) in zip(suggestion_columns, suggestions):
    with column:
        if st.button(label, key=f"suggestion_{label}", use_container_width=True):
            st.session_state.pending_query = question


user_input = st.chat_input("Ask anything about your university...")
query = user_input or st.session_state.pending_query

st.markdown(
    '<div class="uniassist-disclaimer">'
    "UniAssist AI can make mistakes. Check important institutional info."
    "</div>",
    unsafe_allow_html=True,
)


if query:
    st.session_state.pending_query = None
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant", avatar="🎓"):
        st.markdown(
            '<div class="ai-status">✦ &nbsp; Searching the knowledge base</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Reviewing university sources and preparing an answer..."):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Unable to answer this question.")
                sources = response.get("sources", [])
            except NotImplementedError:
                answer = (
                    "⚠️ **Task 10 has not been implemented.** Complete "
                    "`src/task10_generation.py` to connect the RAG pipeline to this interface."
                )
                sources = []
            except Exception as error:
                answer = f"❌ **The RAG pipeline could not complete this request:** {error}"
                sources = []

        st.markdown(answer)
        render_sources(sources)
        st.markdown(
            '<div class="assistant-actions" title="Response actions">⧉ ↻ ♡</div>',
            unsafe_allow_html=True,
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
