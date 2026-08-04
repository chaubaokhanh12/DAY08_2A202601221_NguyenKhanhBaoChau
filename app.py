"""UniAssist AI — evidence-led Streamlit interface for University Services RAG."""

import html
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

PROJECT_ROOT = Path(__file__).parent
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"
DOCUMENT_COUNT = len(list(STANDARDIZED_DIR.rglob("*.md")))
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="UniAssist AI",
    page_icon="U",
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
    :root {
        --navy-950: #001b3a;
        --navy-800: #002b5a;
        --navy-700: #0b4778;
        --teal-700: #006a65;
        --teal-100: #d9f2ef;
        --canvas: #f4f7fa;
        --surface: #ffffff;
        --surface-muted: #edf2f6;
        --text: #172033;
        --text-muted: #526173;
        --border: #d9e2ec;
        --danger: #b42318;
        --shadow-sm: 0 1px 2px rgba(16, 24, 40, 0.04), 0 4px 12px rgba(16, 24, 40, 0.04);
        --shadow-md: 0 12px 32px rgba(16, 24, 40, 0.09);
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 22px;
    }

    html,
    body,
    [class*="css"] {
        color: var(--text);
        font-family: Inter, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    html,
    body,
    [data-testid="stAppViewContainer"],
    .stApp {
        background: var(--canvas);
    }

    #MainMenu,
    footer {
        display: none !important;
    }

    header[data-testid="stHeader"] {
        height: 0;
        background: transparent;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0;
    }

    .block-container {
        max-width: 800px;
        padding-top: 6.75rem;
        padding-bottom: 10rem;
    }

    .academic-copilot-shell {
        position: fixed;
        inset: 0 0 auto 0;
        z-index: 999;
        height: 72px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid rgba(217, 226, 236, 0.92);
        background: rgba(255, 255, 255, 0.96);
        box-shadow: 0 1px 12px rgba(16, 24, 40, 0.04);
        backdrop-filter: blur(14px);
    }

    .appbar-inner {
        width: min(100% - 48px, 1360px);
        margin: 0 auto;
        display: grid;
        grid-template-columns: 1fr auto;
        align-items: center;
        gap: 24px;
    }

    .brand-lockup,
    .appbar-meta,
    .brand-copy,
    .verified-state,
    .knowledge-item,
    .assistant-label,
    .evidence-title,
    .pipeline-step {
        display: flex;
        align-items: center;
    }

    .brand-lockup {
        gap: 12px;
        min-width: 0;
    }

    .brand-mark {
        width: 40px;
        height: 40px;
        flex: 0 0 40px;
        display: grid;
        place-items: center;
        color: white;
        border-radius: 12px;
        background: var(--navy-800);
        box-shadow: 0 6px 16px rgba(0, 43, 90, 0.18);
    }

    .brand-mark svg,
    .inline-icon,
    .status-icon {
        display: block;
        fill: none;
        stroke: currentColor;
        stroke-linecap: round;
        stroke-linejoin: round;
        stroke-width: 1.8;
    }

    .brand-mark svg {
        width: 22px;
        height: 22px;
    }

    .brand-copy {
        min-width: 0;
        align-items: baseline;
        gap: 10px;
    }

    .brand-title {
        color: var(--navy-950);
        font-size: 18px;
        font-weight: 700;
        letter-spacing: -0.025em;
        white-space: nowrap;
    }

    .brand-context {
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 500;
        white-space: nowrap;
    }

    .appbar-meta {
        justify-content: flex-end;
        gap: 16px;
    }

    .verified-state {
        gap: 8px;
        min-height: 36px;
        padding: 0 12px;
        color: var(--teal-700);
        border: 1px solid #b8ded9;
        border-radius: 999px;
        background: #f2fbfa;
        font-size: 12px;
        font-weight: 650;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: var(--teal-700);
        box-shadow: 0 0 0 3px rgba(0, 106, 101, 0.11);
    }

    .profile-avatar {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        color: var(--navy-800);
        border: 1px solid var(--border);
        border-radius: 999px;
        background: var(--surface-muted);
        font-size: 12px;
        font-weight: 750;
    }

    [data-testid="collapsedControl"] {
        top: 82px;
        left: 12px;
        z-index: 1001;
        width: 44px;
        height: 44px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stSidebar"] {
        top: 72px;
        height: calc(100vh - 72px);
        border-right: 1px solid var(--border);
        background: var(--surface);
        box-shadow: 8px 0 30px rgba(16, 24, 40, 0.04);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding: 1.5rem 1.15rem 2rem;
    }

    .sidebar-retrieval {
        margin-bottom: 20px;
        padding: 16px;
        border: 1px solid #c8e4e1;
        border-radius: var(--radius-md);
        background: linear-gradient(145deg, #f8fcfc, #edf8f7);
    }

    .sidebar-kicker {
        margin-bottom: 6px;
        color: var(--teal-700);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-title-row {
        display: flex;
        align-items: center;
        gap: 9px;
        color: var(--navy-950);
    }

    .sidebar-title-row svg {
        width: 20px;
        height: 20px;
    }

    .sidebar-retrieval h2 {
        margin: 0;
        font-size: 19px;
        line-height: 1.3;
        letter-spacing: -0.02em;
    }

    .sidebar-retrieval p {
        margin: 9px 0 0;
        color: var(--text-muted);
        font-size: 12px;
        line-height: 1.55;
    }

    [data-testid="stSidebar"] [data-testid="stSlider"] {
        padding: 8px 2px 4px;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--teal-700);
    }

    .pipeline-steps {
        margin-top: 20px;
        padding-top: 18px;
        border-top: 1px solid var(--border);
    }

    .pipeline-step {
        position: relative;
        gap: 10px;
        min-height: 36px;
        color: var(--text-muted);
        font-size: 12px;
        font-weight: 550;
    }

    .pipeline-number {
        width: 24px;
        height: 24px;
        display: grid;
        place-items: center;
        flex: 0 0 24px;
        color: var(--teal-700);
        border: 1px solid #b8ded9;
        border-radius: 8px;
        background: #f2fbfa;
        font-size: 10px;
        font-weight: 750;
    }

    .knowledge-strip {
        margin-bottom: 34px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        border: 1px solid var(--border);
        border-radius: 14px;
        background: rgba(255, 255, 255, 0.78);
        box-shadow: var(--shadow-sm);
    }

    .knowledge-items {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px 18px;
    }

    .knowledge-item {
        gap: 7px;
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 600;
        white-space: nowrap;
    }

    .knowledge-item svg {
        width: 15px;
        height: 15px;
        color: var(--teal-700);
    }

    .knowledge-badge {
        padding: 5px 9px;
        color: var(--navy-800);
        border-radius: 999px;
        background: var(--surface-muted);
        font-size: 10px;
        font-weight: 700;
        white-space: nowrap;
    }

    .uniassist-welcome {
        margin: min(8vh, 72px) auto 34px;
        text-align: center;
        animation: content-in 220ms ease-out;
    }

    .welcome-symbol {
        width: 52px;
        height: 52px;
        margin: 0 auto 18px;
        display: grid;
        place-items: center;
        color: var(--teal-700);
        border: 1px solid #b8ded9;
        border-radius: 16px;
        background: #f2fbfa;
    }

    .welcome-symbol svg {
        width: 25px;
        height: 25px;
    }

    .welcome-eyebrow {
        margin-bottom: 8px;
        color: var(--teal-700);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.11em;
        text-transform: uppercase;
    }

    .uniassist-welcome h1 {
        max-width: 660px;
        margin: 0 auto 12px;
        color: var(--navy-950);
        font-size: clamp(30px, 5vw, 42px);
        line-height: 1.14;
        letter-spacing: -0.045em;
    }

    .uniassist-welcome p {
        max-width: 590px;
        margin: 0 auto;
        color: var(--text-muted);
        font-size: 15px;
        line-height: 1.65;
    }

    .trust-line {
        margin-top: 18px;
        color: var(--navy-700);
        font-size: 12px;
        font-weight: 650;
    }

    @keyframes content-in {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    [data-testid="stChatMessage"] {
        margin-bottom: 22px;
        padding: 0;
        background: transparent;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        width: fit-content;
        max-width: 78%;
        margin-left: auto;
        padding: 13px 17px;
        color: white;
        border-radius: 18px 6px 18px 18px;
        background: var(--navy-800);
        box-shadow: 0 8px 20px rgba(0, 43, 90, 0.14);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        color: white;
    }

    [data-testid="stChatMessageAvatarUser"] {
        display: none;
    }

    [data-testid="stChatMessageAvatarAssistant"] {
        color: var(--navy-800) !important;
        border: 1px solid var(--border);
        background: var(--surface) !important;
        box-shadow: var(--shadow-sm);
    }

    [data-testid="stChatMessageContent"] {
        color: var(--text);
        font-size: 16px;
        line-height: 1.66;
    }

    [data-testid="stChatMessageContent"] p {
        margin-bottom: 12px;
    }

    .assistant-card {
        margin: 0 0 10px;
    }

    .assistant-label {
        gap: 8px;
        color: var(--navy-800);
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }

    .assistant-label svg {
        width: 15px;
        height: 15px;
        color: var(--teal-700);
    }

    .search-state {
        width: fit-content;
        margin: 0 0 12px;
        padding: 7px 11px;
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--teal-700);
        border: 1px solid #c8e4e1;
        border-radius: 999px;
        background: #f2fbfa;
        font-size: 11px;
        font-weight: 700;
    }

    .search-state svg {
        width: 14px;
        height: 14px;
        animation: status-pulse 1.4s ease-in-out infinite;
    }

    @keyframes status-pulse {
        0%, 100% { opacity: 0.55; }
        50% { opacity: 1; }
    }

    .evidence-panel {
        margin-top: 24px;
        padding-top: 18px;
        border-top: 1px solid var(--border);
    }

    .evidence-title {
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 11px;
    }

    .evidence-heading {
        display: flex;
        align-items: center;
        gap: 8px;
        color: var(--navy-950);
        font-size: 12px;
        font-weight: 750;
    }

    .evidence-heading svg {
        width: 16px;
        height: 16px;
        color: var(--teal-700);
    }

    .evidence-count {
        color: var(--text-muted);
        font-size: 10px;
        font-weight: 600;
    }

    .source-card {
        margin-bottom: 9px;
        padding: 13px 14px;
        display: grid;
        grid-template-columns: 28px 1fr;
        gap: 11px;
        border: 1px solid var(--border);
        border-radius: 13px;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
    }

    .source-index {
        width: 28px;
        height: 28px;
        display: grid;
        place-items: center;
        color: var(--teal-700);
        border: 1px solid #b8ded9;
        border-radius: 9px;
        background: #f2fbfa;
        font-size: 10px;
        font-weight: 800;
    }

    .source-copy strong,
    .source-copy span {
        display: block;
    }

    .source-copy strong {
        overflow: hidden;
        color: var(--text);
        font-size: 12px;
        font-weight: 700;
        text-overflow: ellipsis;
        white-space: nowrap;
    }

    .source-copy span {
        margin-top: 3px;
        color: var(--text-muted);
        font-size: 10px;
    }

    [data-testid="stExpander"] {
        margin-top: -3px;
        margin-bottom: 9px;
        overflow: hidden;
        border: 1px solid var(--border);
        border-radius: 11px;
        background: rgba(255, 255, 255, 0.72);
    }

    [data-testid="stExpander"] summary {
        min-height: 44px;
        color: var(--text-muted);
        font-size: 11px;
        font-weight: 650;
    }

    .suggestion-label {
        margin: 26px 0 10px;
        color: var(--navy-950);
        font-size: 12px;
        font-weight: 750;
    }

    .suggestion-help {
        margin: -5px 0 13px;
        color: var(--text-muted);
        font-size: 11px;
    }

    [data-testid="stButton"] button {
        min-height: 44px;
        padding: 8px 13px;
        color: var(--navy-800);
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
        font-size: 11px;
        font-weight: 650;
        transition: color 180ms ease, border-color 180ms ease, background 180ms ease, box-shadow 180ms ease;
    }

    [data-testid="stButton"] button:hover {
        color: var(--teal-700);
        border-color: #9bcfc9;
        background: #f8fcfc;
        box-shadow: 0 5px 16px rgba(0, 106, 101, 0.08);
    }

    [data-testid="stButton"] button:active {
        background: #edf8f7;
        box-shadow: none;
    }

    button:focus-visible,
    [role="slider"]:focus-visible,
    textarea:focus-visible,
    summary:focus-visible {
        outline: 3px solid var(--teal-700) !important;
        outline-offset: 2px !important;
    }

    [data-testid="stChatInput"] {
        min-height: 58px;
        border: 1px solid #c6d3df;
        border-radius: 17px;
        background: var(--surface);
        box-shadow: var(--shadow-md);
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: var(--teal-700);
        box-shadow: 0 0 0 3px rgba(0, 106, 101, 0.11), var(--shadow-md);
    }

    [data-testid="stChatInput"] button {
        width: 44px;
        height: 44px;
        color: white;
        border-radius: 12px;
        background: var(--navy-800);
    }

    [data-testid="stChatInput"] button:hover {
        background: var(--navy-700);
    }

    [data-testid="stBottom"] {
        background: linear-gradient(to top, var(--canvas) 74%, rgba(244, 247, 250, 0));
    }

    .uniassist-disclaimer {
        margin-top: 9px;
        color: var(--text-muted);
        text-align: center;
        font-size: 10px;
        line-height: 1.45;
    }

    @media (max-width: 720px) {
        .block-container {
            padding: 6rem 1rem 9rem;
        }

        .appbar-inner {
            width: calc(100% - 30px);
        }

        .brand-context,
        .verified-copy {
            display: none;
        }

        [data-testid="stSidebar"] {
            width: min(86vw, 320px) !important;
        }

        .knowledge-strip,
        .evidence-title {
            align-items: flex-start;
        }

        .knowledge-strip {
            flex-direction: column;
        }

        .uniassist-welcome {
            margin-top: 5vh;
        }

        [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
            max-width: 90%;
        }

        [data-testid="column"] {
            min-width: min(100%, 160px) !important;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            scroll-behavior: auto !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <header class="academic-copilot-shell">
        <div class="appbar-inner">
            <div class="brand-lockup">
                <div class="brand-mark" aria-hidden="true">
                    <svg viewBox="0 0 24 24">
                        <path d="M4 5.5h5.5A2.5 2.5 0 0 1 12 8v11a3 3 0 0 0-3-3H4z"></path>
                        <path d="M20 5.5h-5.5A2.5 2.5 0 0 0 12 8v11a3 3 0 0 1 3-3h5z"></path>
                    </svg>
                </div>
                <div class="brand-copy">
                    <span class="brand-title">UniAssist AI</span>
                    <span class="brand-context">Academic Research Copilot</span>
                </div>
            </div>
            <div class="appbar-meta">
                <div class="verified-state">
                    <span class="status-dot" aria-hidden="true"></span>
                    <span class="verified-copy">Evidence mode active</span>
                </div>
                <div class="profile-avatar" aria-label="UniAssist profile">UA</div>
            </div>
        </div>
    </header>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown(
        """
        <section class="sidebar-retrieval">
            <div class="sidebar-kicker">Knowledge controls</div>
            <div class="sidebar-title-row">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 7h16M7 12h10M10 17h4"></path>
                    <circle cx="8" cy="7" r="2"></circle>
                    <circle cx="15" cy="12" r="2"></circle>
                    <circle cx="12" cy="17" r="2"></circle>
                </svg>
                <h2>Retrieval settings</h2>
            </div>
            <p>Choose how much evidence UniAssist reviews before composing an answer.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    top_k = st.slider(
        "Sources to retrieve",
        min_value=3,
        max_value=10,
        key="top_k",
        help="Number of knowledge-base chunks sent to the answer generator.",
    )
    st.caption(f"Reviewing up to **{top_k}** knowledge chunks per question.")
    st.markdown(
        """
        <div class="pipeline-steps">
            <div class="sidebar-kicker">Retrieval flow</div>
            <div class="pipeline-step"><span class="pipeline-number">01</span>Semantic search</div>
            <div class="pipeline-step"><span class="pipeline-number">02</span>BM25 keyword search</div>
            <div class="pipeline-step"><span class="pipeline-number">03</span>RRF reranking</div>
            <div class="pipeline-step"><span class="pipeline-number">04</span>PageIndex fallback</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <section class="knowledge-strip" aria-label="Knowledge base status">
        <div class="knowledge-items">
            <span class="knowledge-item">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M5 4h14v16H5z"></path><path d="M8 8h8M8 12h8M8 16h5"></path>
                </svg>
                {DOCUMENT_COUNT} standardized documents
            </span>
            <span class="knowledge-item">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="12" cy="12" r="9"></circle><path d="m9 12 2 2 4-5"></path>
                </svg>
                RMIT policies and services
            </span>
        </div>
        <span class="knowledge-badge">Coverage: 2025–2026</span>
    </section>
    """,
    unsafe_allow_html=True,
)


def render_sources(sources: list[dict]) -> None:
    """Render retrieved chunks as safely escaped evidence cards."""
    if not sources:
        return

    st.markdown(
        f"""
        <div class="evidence-panel">
            <div class="evidence-title">
                <span class="evidence-heading">
                    <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                        <path d="M4 5h16v14H4z"></path><path d="M8 9h8M8 13h8M8 17h5"></path>
                    </svg>
                    Evidence used
                </span>
                <span class="evidence-count">{len(sources)} retrieved chunks</span>
            </div>
        </div>
        """,
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
            f'<span class="source-index">{index:02d}</span>'
            '<span class="source-copy">'
            f"<strong>{html.escape(source_name)}</strong>"
            f"<span>{html.escape(document_type)} &nbsp;·&nbsp; relevance {score:.4f}</span>"
            "</span></div>",
            unsafe_allow_html=True,
        )

        excerpt = str(source.get("content") or "")[:500]
        if excerpt:
            with st.expander(f"Read evidence excerpt {index}"):
                st.text(excerpt)


def render_assistant_label() -> None:
    """Render the consistent assistant identity above an answer."""
    st.markdown(
        """
        <div class="assistant-card">
            <div class="assistant-label">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M12 3v3M5.6 5.6l2.1 2.1M3 12h3M18 12h3M16.3 7.7l2.1-2.1"></path>
                    <path d="M8 14a4 4 0 1 1 8 0c0 1.7-1 2.5-2 3.5V20h-4v-2.5C9 16.5 8 15.7 8 14z"></path>
                </svg>
                UniAssist response
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(message: dict) -> None:
    """Render one persisted chat message and its optional evidence."""
    role = message.get("role", "assistant")
    with st.chat_message(role):
        if role == "assistant":
            render_assistant_label()
        st.markdown(str(message.get("content", "")))
        if role == "assistant":
            render_sources(message.get("sources") or [])


if not st.session_state.messages:
    st.markdown(
        """
        <section class="uniassist-welcome">
            <div class="welcome-symbol" aria-hidden="true">
                <svg class="inline-icon" viewBox="0 0 24 24">
                    <path d="M12 3v3M5.6 5.6l2.1 2.1M3 12h3M18 12h3M16.3 7.7l2.1-2.1"></path>
                    <path d="M8 14a4 4 0 1 1 8 0c0 1.7-1 2.5-2 3.5V20h-4v-2.5C9 16.5 8 15.7 8 14z"></path>
                </svg>
            </div>
            <div class="welcome-eyebrow">RMIT University Services</div>
            <h1>Reliable answers, grounded in university sources.</h1>
            <p>
                Ask about tuition, scholarships, accommodation, library services,
                and student wellbeing. UniAssist finds evidence before it answers.
            </p>
            <div class="trust-line">Answers include the documents used whenever evidence is available.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


for persisted_message in st.session_state.messages:
    render_message(persisted_message)


st.markdown(
    """
    <div class="suggestion-label">Start with a verified topic</div>
    <div class="suggestion-help">These questions match the documents currently in the knowledge base.</div>
    """,
    unsafe_allow_html=True,
)

suggestions = [
    (
        "Tuition and Census Date",
        "What is the Census Date, and how does it affect my tuition fee liability?",
    ),
    (
        "Scholarship requirements",
        "What GPA and study load must RMIT scholarship recipients maintain?",
    ),
    (
        "Housing checklist",
        "What should international students check before signing a rental contract in Ho Chi Minh City?",
    ),
    (
        "Library study rooms",
        "How can I book a study room through the RMIT Library?",
    ),
]

suggestion_columns = st.columns(2)
for index, (label, question) in enumerate(suggestions):
    with suggestion_columns[index % 2]:
        if st.button(label, key=f"suggestion_{index}", use_container_width=True):
            st.session_state.pending_query = question


user_input = st.chat_input("Ask a question about RMIT services and policies")
query = user_input or st.session_state.pending_query

st.markdown(
    """
    <div class="uniassist-disclaimer">
        UniAssist may make mistakes. Verify high-impact decisions against official university information.
    </div>
    """,
    unsafe_allow_html=True,
)


if query:
    st.session_state.pending_query = None
    user_message = {"role": "user", "content": query}
    st.session_state.messages.append(user_message)
    render_message(user_message)

    with st.chat_message("assistant"):
        render_assistant_label()
        st.markdown(
            """
            <div class="search-state">
                <svg class="status-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <circle cx="11" cy="11" r="6"></circle><path d="m16 16 4 4"></path>
                </svg>
                Searching the university knowledge base
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.spinner("Reviewing retrieved evidence and preparing an answer..."):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(query, top_k=top_k)
                answer = response.get("answer", "Unable to answer this question.")
                sources = response.get("sources", [])
            except NotImplementedError:
                answer = (
                    "**Generation is not connected yet.** Complete "
                    "`src/task10_generation.py` to connect the RAG pipeline to this interface."
                )
                sources = []
            except Exception as error:
                answer = f"**The RAG pipeline could not complete this request.** {error}"
                sources = []

        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
