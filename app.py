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
LEGAL_DOCUMENT_COUNT = len(list((STANDARDIZED_DIR / "legal").glob("*.md")))
NEWS_DOCUMENT_COUNT = len(list((STANDARDIZED_DIR / "news").glob("*.md")))
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
        --navy-950: #0f172a;
        --navy-800: #1e3a5f;
        --navy-700: #274c77;
        --cobalt-600: #2563eb;
        --cobalt-700: #1d4ed8;
        --blue-100: #dbeafe;
        --teal-700: #0f766e;
        --teal-100: #ccfbf1;
        --orange-600: #ea580c;
        --orange-100: #ffedd5;
        --canvas: #f3f6fb;
        --surface: #ffffff;
        --surface-muted: #e9eef5;
        --text: #0f172a;
        --text-muted: #52647a;
        --border: #cbd5e1;
        --danger: #b42318;
        --shadow-sm: 0 2px 4px rgba(15, 23, 42, 0.04), 0 8px 20px rgba(30, 58, 95, 0.06);
        --shadow-md: 0 18px 44px rgba(30, 58, 95, 0.13);
        --shadow-lg: 0 28px 70px rgba(30, 58, 95, 0.18);
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 24px;
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
        background:
            radial-gradient(circle at 16% 10%, rgba(37, 99, 235, 0.10), transparent 27rem),
            radial-gradient(circle at 88% 18%, rgba(234, 88, 12, 0.07), transparent 24rem),
            var(--canvas);
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
        background: linear-gradient(145deg, var(--cobalt-600), var(--navy-800));
        box-shadow: 0 8px 20px rgba(37, 99, 235, 0.26);
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
        border: 1px solid #99d5ce;
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
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: var(--radius-md);
        background:
            radial-gradient(circle at 90% 0%, rgba(37, 99, 235, 0.55), transparent 10rem),
            linear-gradient(145deg, #162d4d, var(--navy-800));
        box-shadow: 0 16px 34px rgba(30, 58, 95, 0.18);
    }

    .sidebar-kicker {
        margin-bottom: 6px;
        color: #99f6e4;
        font-size: 11px;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-title-row {
        display: flex;
        align-items: center;
        gap: 9px;
        color: white;
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
        color: #d8e5f2;
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
        margin-bottom: 18px;
    }

    .insight-rail {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
    }

    .insight-card {
        min-height: 82px;
        padding: 14px 15px;
        display: flex;
        align-items: center;
        gap: 12px;
        border: 1px solid rgba(203, 213, 225, 0.78);
        border-radius: 16px;
        background: rgba(255, 255, 255, 0.84);
        box-shadow: var(--shadow-sm);
        backdrop-filter: blur(12px);
    }

    .insight-icon {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        flex: 0 0 38px;
        border-radius: 12px;
    }

    .insight-icon svg {
        width: 19px;
        height: 19px;
    }

    .insight-icon.blue {
        color: var(--cobalt-700);
        background: var(--blue-100);
    }

    .insight-icon.teal {
        color: var(--teal-700);
        background: var(--teal-100);
    }

    .insight-icon.orange {
        color: var(--orange-600);
        background: var(--orange-100);
    }

    .insight-copy strong,
    .insight-copy span {
        display: block;
    }

    .insight-copy strong {
        color: var(--navy-950);
        font-size: 14px;
        font-weight: 750;
        line-height: 1.2;
    }

    .insight-copy span {
        margin-top: 4px;
        color: var(--text-muted);
        font-size: 10px;
        font-weight: 600;
    }

    .bento-hero {
        display: grid;
        grid-template-columns: minmax(0, 1.65fr) minmax(210px, 0.85fr);
        gap: 14px;
        margin: 0 0 30px;
    }

    .hero-primary {
        position: relative;
        min-height: 286px;
        padding: 32px;
        overflow: hidden;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 28px;
        background:
            radial-gradient(circle at 88% 12%, rgba(96, 165, 250, 0.62), transparent 13rem),
            radial-gradient(circle at 8% 100%, rgba(15, 118, 110, 0.36), transparent 14rem),
            linear-gradient(145deg, #172f50 0%, var(--navy-800) 48%, #1e4f91 100%);
        box-shadow: var(--shadow-lg);
    }

    .hero-primary::after {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        right: -70px;
        bottom: -90px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255, 255, 255, 0.035), 0 0 0 58px rgba(255, 255, 255, 0.025);
    }

    .hero-topline,
    .hero-proof,
    .metric-topline {
        display: flex;
        align-items: center;
    }

    .hero-topline {
        position: relative;
        z-index: 1;
        justify-content: space-between;
        gap: 16px;
    }

    .hero-pill {
        padding: 7px 10px;
        color: #fff7ed;
        border: 1px solid rgba(255, 237, 213, 0.32);
        border-radius: 999px;
        background: rgba(234, 88, 12, 0.78);
        font-size: 10px;
        font-weight: 750;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .hero-year {
        color: #dbeafe;
        font-size: 10px;
        font-weight: 650;
    }

    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 520px;
        margin: 30px 0 26px;
    }

    .hero-content h1 {
        margin: 0;
        color: white;
        font-size: clamp(31px, 5vw, 46px);
        line-height: 1.08;
        letter-spacing: -0.052em;
    }

    .hero-content p {
        max-width: 470px;
        margin: 14px 0 0;
        color: #dce9f7;
        font-size: 14px;
        line-height: 1.65;
    }

    .hero-proof {
        position: relative;
        z-index: 1;
        gap: 8px;
        color: #ccfbf1;
        font-size: 11px;
        font-weight: 650;
    }

    .hero-proof svg {
        width: 16px;
        height: 16px;
    }

    .hero-metrics {
        display: grid;
        grid-template-rows: repeat(3, 1fr);
        gap: 12px;
    }

    .metric-card {
        min-height: 82px;
        padding: 15px 16px;
        overflow: hidden;
        border: 1px solid rgba(203, 213, 225, 0.78);
        border-radius: 20px;
        background: var(--surface);
        box-shadow: var(--shadow-sm);
    }

    .metric-card.blue { background: linear-gradient(145deg, #eff6ff, #dbeafe); }
    .metric-card.teal { background: linear-gradient(145deg, #f0fdfa, #ccfbf1); }
    .metric-card.orange { background: linear-gradient(145deg, #fff7ed, #ffedd5); }

    .metric-topline {
        justify-content: space-between;
        gap: 12px;
        color: var(--text-muted);
        font-size: 9px;
        font-weight: 750;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }

    .metric-value {
        margin-top: 5px;
        color: var(--navy-950);
        font-size: 23px;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.04em;
    }

    .metric-note {
        margin-top: 5px;
        color: var(--text-muted);
        font-size: 9px;
        font-weight: 600;
    }

    .knowledge-meter {
        height: 5px;
        margin-top: 9px;
        overflow: hidden;
        border-radius: 999px;
        background: rgba(100, 116, 139, 0.16);
    }

    .knowledge-meter span {
        width: 78%;
        height: 100%;
        display: block;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--cobalt-600), var(--teal-700));
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

    .evidence-insight-card {
        position: relative;
        overflow: hidden;
    }

    .evidence-insight-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: linear-gradient(180deg, var(--cobalt-600), var(--teal-700));
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
        margin: 30px 0 10px;
        color: var(--navy-950);
        font-size: 18px;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    .suggestion-help {
        margin: -5px 0 16px;
        color: var(--text-muted);
        font-size: 12px;
    }

    .topic-card-grid {
        margin-top: 10px;
    }

    .topic-card {
        position: relative;
        min-height: 132px;
        margin-bottom: 8px;
        padding: 17px;
        overflow: hidden;
        border: 1px solid rgba(203, 213, 225, 0.78);
        border-radius: 18px;
        box-shadow: var(--shadow-sm);
    }

    .topic-card::after {
        content: "";
        position: absolute;
        width: 90px;
        height: 90px;
        right: -35px;
        bottom: -45px;
        border: 16px solid rgba(255, 255, 255, 0.36);
        border-radius: 50%;
    }

    .topic-card.blue { background: linear-gradient(145deg, #eff6ff, #dbeafe); }
    .topic-card.teal { background: linear-gradient(145deg, #f0fdfa, #ccfbf1); }
    .topic-card.orange { background: linear-gradient(145deg, #fff7ed, #ffedd5); }
    .topic-card.navy { background: linear-gradient(145deg, #eef2f7, #dce6f1); }

    .topic-card .topic-icon {
        width: 34px;
        height: 34px;
        display: grid;
        place-items: center;
        margin-top: 0;
        border-radius: 11px;
        background: rgba(255, 255, 255, 0.68);
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.06);
    }

    .topic-card.blue .topic-icon { color: var(--cobalt-700); }
    .topic-card.teal .topic-icon { color: var(--teal-700); }
    .topic-card.orange .topic-icon { color: var(--orange-600); }
    .topic-card.navy .topic-icon { color: var(--navy-800); }

    .topic-icon svg {
        width: 18px;
        height: 18px;
        fill: none;
        stroke: currentColor;
        stroke-width: 1.8;
        stroke-linecap: round;
        stroke-linejoin: round;
    }

    .topic-card strong {
        position: relative;
        z-index: 1;
        display: block;
        margin-top: 13px;
        color: var(--navy-950);
        font-size: 13px;
        font-weight: 800;
    }

    .topic-card .topic-description {
        position: relative;
        z-index: 1;
        display: block;
        margin-top: 4px;
        color: var(--text-muted);
        font-size: 10px;
        font-weight: 600;
        line-height: 1.45;
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
        color: white;
        border-color: var(--cobalt-600);
        background: var(--cobalt-600);
        box-shadow: 0 8px 18px rgba(37, 99, 235, 0.20);
    }

    [data-testid="stButton"] button:active {
        color: white;
        border-color: var(--cobalt-700);
        background: var(--cobalt-700);
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
        box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.13), var(--shadow-md);
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

        .bento-hero {
            grid-template-columns: 1fr;
        }

        .hero-primary {
            min-height: 265px;
            padding: 25px;
        }

        .hero-metrics {
            grid-template-columns: repeat(3, 1fr);
            grid-template-rows: 1fr;
            gap: 8px;
        }

        .metric-card {
            min-height: 100px;
            padding: 12px;
        }

        .metric-topline {
            align-items: flex-start;
            min-height: 24px;
        }

        .insight-rail {
            grid-template-columns: 1fr;
        }

        .evidence-title {
            align-items: flex-start;
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
    <section class="knowledge-strip insight-rail" aria-label="Knowledge base status">
        <div class="insight-card">
            <span class="insight-icon blue">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M5 4h14v16H5z"></path><path d="M8 8h8M8 12h8M8 16h5"></path>
                </svg>
            </span>
            <span class="insight-copy"><strong>{DOCUMENT_COUNT} sources</strong><span>Standardized knowledge</span></span>
        </div>
        <div class="insight-card">
            <span class="insight-icon teal">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M4 6h6l2 2h8v10H4z"></path><path d="m9 13 2 2 4-4"></path>
                </svg>
            </span>
            <span class="insight-copy"><strong>{LEGAL_DOCUMENT_COUNT} guides</strong><span>Policies and services</span></span>
        </div>
        <div class="insight-card">
            <span class="insight-icon orange">
                <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                    <path d="M5 5h14v14H5z"></path><path d="M8 9h8M8 13h5M8 16h7"></path>
                </svg>
            </span>
            <span class="insight-copy"><strong>{NEWS_DOCUMENT_COUNT} stories</strong><span>Student news and updates</span></span>
        </div>
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
            '<div class="source-card evidence-insight-card">'
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
        f"""
        <section class="bento-hero" aria-label="UniAssist overview">
            <article class="hero-primary">
                <div class="hero-topline">
                    <span class="hero-pill">Academic Copilot</span>
                    <span class="hero-year">Knowledge coverage · 2025–2026</span>
                </div>
                <div class="hero-content">
                    <h1>Find the answer.<br>See the evidence.</h1>
                    <p>
                        Explore tuition, scholarships, accommodation, library services,
                        and student wellbeing through verified RMIT source material.
                    </p>
                </div>
                <div class="hero-proof">
                    <svg class="inline-icon" viewBox="0 0 24 24" aria-hidden="true">
                        <circle cx="12" cy="12" r="9"></circle><path d="m8 12 2.5 2.5L16 9"></path>
                    </svg>
                    Evidence is shown alongside every supported answer
                </div>
            </article>
            <aside class="hero-metrics" aria-label="Knowledge metrics">
                <div class="metric-card blue">
                    <div class="metric-topline"><span>Knowledge base</span><span>Live</span></div>
                    <div class="metric-value">{DOCUMENT_COUNT}</div>
                    <div class="metric-note">standardized documents</div>
                </div>
                <div class="metric-card teal">
                    <div class="metric-topline"><span>Evidence depth</span><span>Top-k</span></div>
                    <div class="metric-value">{st.session_state.top_k}</div>
                    <div class="knowledge-meter"><span></span></div>
                </div>
                <div class="metric-card orange">
                    <div class="metric-topline"><span>Focus areas</span><span>RMIT</span></div>
                    <div class="metric-value">5</div>
                    <div class="metric-note">student service domains</div>
                </div>
            </aside>
        </section>
        <section class="uniassist-welcome">
            <div class="welcome-eyebrow">Ask with confidence</div>
            <h1>What would you like to understand?</h1>
            <p>Choose a verified topic below or ask your own question in the composer.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


for persisted_message in st.session_state.messages:
    render_message(persisted_message)


suggestions = [
    (
        "blue",
        "Tuition and Census Date",
        "Understand fee liability and the deadline that changes it.",
        "Ask about tuition",
        "What is the Census Date, and how does it affect my tuition fee liability?",
        '<path d="M4 19.5V6.8L12 3l8 3.8v12.7"></path><path d="M7 10h10M7 14h10M8 19.5v-3h8v3"></path>',
    ),
    (
        "teal",
        "Scholarship requirements",
        "Check the academic standards needed to keep an award.",
        "Check scholarship rules",
        "What GPA and study load must RMIT scholarship recipients maintain?",
        '<path d="m3 9 9-5 9 5-9 5-9-5Z"></path><path d="M7 11.5V16c2.8 2 7.2 2 10 0v-4.5M21 9v6"></path>',
    ),
    (
        "orange",
        "Housing checklist",
        "Review the contract, landlord and property before signing.",
        "Review housing steps",
        "What should international students check before signing a rental contract in Ho Chi Minh City?",
        '<path d="m3 11 9-7 9 7"></path><path d="M5 10v10h14V10M9 20v-6h6v6"></path>',
    ),
    (
        "navy",
        "Library study rooms",
        "Find the booking flow, conditions and access guidance.",
        "Explore room booking",
        "How can I book a study room through the RMIT Library?",
        '<path d="M4 5h16v14H4z"></path><path d="M8 9h8M8 13h8M8 17h5"></path>',
    ),
]

if not st.session_state.messages:
    st.markdown(
        """
        <section class="topic-card-grid">
            <div class="suggestion-label">Start with a verified topic</div>
            <div class="suggestion-help">Four useful entry points grounded in the current knowledge base.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    suggestion_columns = st.columns(2)
    for index, (tone, title, description, action, question, icon_paths) in enumerate(
        suggestions
    ):
        with suggestion_columns[index % 2]:
            st.markdown(
                f"""
                <article class="topic-card {tone}">
                    <span class="topic-icon" aria-hidden="true">
                        <svg viewBox="0 0 24 24">{icon_paths}</svg>
                    </span>
                    <strong>{html.escape(title)}</strong>
                    <span class="topic-description">{html.escape(description)}</span>
                </article>
                """,
                unsafe_allow_html=True,
            )
            if st.button(action, key=f"suggestion_{index}", use_container_width=True):
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
