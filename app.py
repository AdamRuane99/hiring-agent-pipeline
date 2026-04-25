import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.cv_parser import parse_directory, parse_uploaded_files
from pipeline.crew import run_pipeline

SAMPLE_JD = """Role: Senior Marketing Manager — Consumer Brands

Required:
- 4+ years of marketing experience in a B2C environment
- Proven campaign management across digital and traditional channels
- Experience owning and managing a marketing budget (£1M+)
- Team leadership — managing direct reports and agency partners
- Proficiency with CRM platforms and marketing analytics tools

Nice to have:
- Experience in FMCG or consumer goods
- Hands-on paid media management (Google Ads, Meta Ads)
- Market research and brand tracking experience

We are a fast-growing consumer brand looking for a senior marketer to own our
campaign strategy, lead a team of 5, and drive brand growth across the UK market.
"""

PROVIDERS = {
    "HuggingFace Inference API — free tier": "hf_api",
    "Ollama — local, free, no API key": "ollama",
    "Google Gemini — free tier (AI Studio, non-EU)": "gemini",
    "OpenAI": "openai",
    "Local HuggingFace — no API key (CPU, slow)": "local_hf",
}


# ── SVG BOT ────────────────────────────────────────────────────────────────────

_BOT_SVG_LARGE = """
<svg width="80" height="80" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="40" y1="3" x2="40" y2="16" stroke="#7c3aed" stroke-width="2.5" stroke-linecap="round"/>
  <circle cx="40" cy="3" r="3.5" fill="#06b6d4">
    <animate attributeName="opacity" values="1;0.25;1" dur="2s" repeatCount="indefinite"/>
  </circle>
  <rect x="12" y="16" width="56" height="38" rx="12" fill="url(#hg)"/>
  <circle cx="29" cy="33" r="6" fill="#080d1a"/>
  <circle cx="51" cy="33" r="6" fill="#080d1a"/>
  <circle cx="29" cy="33" r="3.5" fill="url(#eg)">
    <animate attributeName="opacity" values="1;0.35;1" dur="2.6s" repeatCount="indefinite"/>
  </circle>
  <circle cx="51" cy="33" r="3.5" fill="url(#eg)">
    <animate attributeName="opacity" values="1;0.35;1" dur="2.6s" begin="0.35s" repeatCount="indefinite"/>
  </circle>
  <rect x="26" y="44" width="28" height="5" rx="2.5" fill="rgba(255,255,255,0.12)"/>
  <rect x="30" y="45.5" width="5" height="2" rx="1" fill="url(#eg)"/>
  <rect x="37.5" y="45.5" width="5" height="2" rx="1" fill="url(#eg)"/>
  <rect x="45" y="45.5" width="5" height="2" rx="1" fill="url(#eg)"/>
  <rect x="32" y="53" width="16" height="7" rx="3.5" fill="url(#hg)" opacity="0.8"/>
  <rect x="18" y="58" width="44" height="20" rx="10" fill="url(#bg)" opacity="0.9"/>
  <circle cx="30" cy="68" r="3" fill="rgba(255,255,255,0.18)"/>
  <circle cx="40" cy="68" r="3" fill="url(#eg)" opacity="0.85"/>
  <circle cx="50" cy="68" r="3" fill="rgba(255,255,255,0.18)"/>
  <defs>
    <linearGradient id="hg" x1="12" y1="16" x2="68" y2="54" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b3f6e"/>
      <stop offset="100%" stop-color="#1c2050"/>
    </linearGradient>
    <linearGradient id="bg" x1="18" y1="58" x2="62" y2="78" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#282d60"/>
      <stop offset="100%" stop-color="#141838"/>
    </linearGradient>
    <linearGradient id="eg" x1="0" y1="0" x2="1" y2="1" gradientUnits="objectBoundingBox">
      <stop offset="0%" stop-color="#7c3aed"/>
      <stop offset="100%" stop-color="#06b6d4"/>
    </linearGradient>
  </defs>
</svg>
"""

_BOT_SVG_SMALL = """
<svg width="22" height="22" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="12" y="16" width="56" height="38" rx="12" fill="rgba(255,255,255,0.2)"/>
  <circle cx="29" cy="33" r="6" fill="rgba(255,255,255,0.7)"/>
  <circle cx="51" cy="33" r="6" fill="rgba(255,255,255,0.7)"/>
  <line x1="40" y1="3" x2="40" y2="16" stroke="white" stroke-width="3" stroke-linecap="round"/>
  <circle cx="40" cy="3" r="3" fill="white" opacity="0.6"/>
  <rect x="18" y="58" width="44" height="20" rx="10" fill="rgba(255,255,255,0.15)"/>
</svg>
"""

_BOT_SVG_THINKING = """
<svg width="32" height="32" viewBox="0 0 80 80" fill="none" xmlns="http://www.w3.org/2000/svg">
  <rect x="12" y="16" width="56" height="38" rx="12" fill="url(#tg)"/>
  <circle cx="29" cy="33" r="6" fill="#080d1a"/>
  <circle cx="51" cy="33" r="6" fill="#080d1a"/>
  <circle cx="29" cy="33" r="3.5" fill="#7c3aed">
    <animate attributeName="opacity" values="1;0.2;1" dur="1.2s" repeatCount="indefinite"/>
  </circle>
  <circle cx="51" cy="33" r="3.5" fill="#06b6d4">
    <animate attributeName="opacity" values="1;0.2;1" dur="1.2s" begin="0.15s" repeatCount="indefinite"/>
  </circle>
  <defs>
    <linearGradient id="tg" x1="12" y1="16" x2="68" y2="54" gradientUnits="userSpaceOnUse">
      <stop offset="0%" stop-color="#3b3f6e"/>
      <stop offset="100%" stop-color="#1c2050"/>
    </linearGradient>
  </defs>
</svg>
"""


# ── CSS ────────────────────────────────────────────────────────────────────────

def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── GLOBAL ──────────────────────────────────────────────────────── */
html, body, .stApp {
    background: #080d1a !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #f8fafc !important;
}

[data-testid="stHeader"]  { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
#MainMenu { display: none !important; }
footer    { display: none !important; }

.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 3rem !important;
    max-width: 1200px !important;
}

/* ── ANIMATED MESH BACKGROUND ────────────────────────────────────── */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 70% 55% at 12% 12%, rgba(124,58,237,.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 60% at 88% 82%, rgba(6,182,212,.12) 0%, transparent 60%),
        radial-gradient(ellipse 55% 80% at 50% 42%, rgba(59,130,246,.07) 0%, transparent 60%);
    animation: meshDrift 11s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes meshDrift {
    0%   { opacity: .7; transform: scale(1)     translateY(0px);    }
    100% { opacity: 1;  transform: scale(1.07)  translateY(-18px);  }
}

/* ── SIDEBAR ─────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(11,17,37,.98) 0%, rgba(8,13,26,.98) 100%) !important;
    border-right: 1px solid rgba(124,58,237,.22) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #e2e8f0 !important;
    font-size: 11px !important;
    letter-spacing: .09em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
    margin-top: 20px !important;
}

[data-testid="stSidebar"] .stMarkdown p { color: #64748b !important; line-height: 1.6 !important; }
[data-testid="stSidebar"] .stCaption p  { color: #475569 !important; font-size: 11px !important; }
[data-testid="stSidebar"] .stMarkdown strong { color: #a78bfa !important; }
[data-testid="stSidebar"] .stMarkdown code {
    background: rgba(124,58,237,.15) !important;
    color: #a78bfa !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
    font-size: 11px !important;
}

/* Sidebar mascot header */
.sb-mascot {
    display: flex;
    align-items: center;
    gap: 11px;
    padding: 22px 16px 16px;
    border-bottom: 1px solid rgba(124,58,237,.18);
    margin-bottom: 4px;
}
.sb-icon {
    width: 38px; height: 38px;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    box-shadow: 0 0 0 rgba(124,58,237,0);
    animation: iconGlow 3s ease-in-out infinite;
}
@keyframes iconGlow {
    0%,100% { box-shadow: 0 0 18px rgba(124,58,237,.35); }
    50%      { box-shadow: 0 0 32px rgba(6,182,212,.45);  }
}
.sb-name    { color: #f1f5f9; font-weight: 700; font-size: 15px; line-height: 1.2; }
.sb-tagline { color: #475569; font-size: 11px; margin-top: 1px; }

/* ── TEXT INPUTS & TEXT AREAS ────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > textarea {
    background: rgba(255,255,255,.04) !important;
    border: 1.5px solid rgba(124,58,237,.25) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    caret-color: #a78bfa !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color .2s, box-shadow .2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > textarea:focus {
    border-color: rgba(124,58,237,.65) !important;
    box-shadow: 0 0 0 3px rgba(124,58,237,.1) !important;
    outline: none !important;
}
.stTextArea > div > textarea:disabled {
    opacity: .55 !important;
    cursor: default !important;
}

/* ── SELECTBOX ───────────────────────────────────────────────────── */
.stSelectbox > div > div {
    background: rgba(255,255,255,.04) !important;
    border: 1.5px solid rgba(124,58,237,.25) !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    transition: border-color .2s !important;
}
.stSelectbox > div > div:focus-within {
    border-color: rgba(124,58,237,.6) !important;
}
.stSelectbox svg { color: #a78bfa !important; }

/* ── CHECKBOXES ──────────────────────────────────────────────────── */
.stCheckbox label p { color: #94a3b8 !important; font-size: 14px !important; }
.stCheckbox svg    { color: #7c3aed !important; }

/* ── PRIMARY BUTTON ──────────────────────────────────────────────── */
.stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 50%, #06b6d4 100%) !important;
    background-size: 200% 200% !important;
    border: none !important;
    border-radius: 12px !important;
    color: #fff !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    letter-spacing: .02em !important;
    padding: 14px 28px !important;
    transition: transform .2s ease, box-shadow .2s ease !important;
    animation: btnGrad 5s ease infinite !important;
}
.stButton > button:hover {
    transform: translateY(-2px) scale(1.02) !important;
    box-shadow: 0 14px 42px rgba(124,58,237,.42) !important;
}
.stButton > button:active {
    transform: translateY(0) scale(.99) !important;
    box-shadow: none !important;
}
@keyframes btnGrad {
    0%   { background-position: 0%   50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0%   50%; }
}

/* ── DOWNLOAD BUTTON ─────────────────────────────────────────────── */
.stDownloadButton > button {
    background: rgba(255,255,255,.03) !important;
    border: 1.5px solid rgba(124,58,237,.35) !important;
    border-radius: 10px !important;
    color: #a78bfa !important;
    font-weight: 500 !important;
    transition: all .2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(124,58,237,.12) !important;
    border-color: rgba(124,58,237,.65) !important;
    color: #c4b5fd !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 22px rgba(124,58,237,.22) !important;
}

/* ── FILE UPLOADER ───────────────────────────────────────────────── */
[data-testid="stFileUploader"] > div {
    background: rgba(255,255,255,.025) !important;
    border: 2px dashed rgba(124,58,237,.38) !important;
    border-radius: 14px !important;
    transition: all .3s ease !important;
    animation: uploadPulse 4s ease-in-out infinite !important;
}
[data-testid="stFileUploader"] > div:hover {
    background: rgba(124,58,237,.06) !important;
    border-color: rgba(124,58,237,.72) !important;
    animation: none !important;
}
@keyframes uploadPulse {
    0%,100% { border-color: rgba(124,58,237,.38); }
    50%      { border-color: rgba(6,182,212,.55);  }
}

/* ── COLUMNS (glass cards) ───────────────────────────────────────── */
[data-testid="column"] {
    background: rgba(255,255,255,.025) !important;
    border: 1px solid rgba(255,255,255,.065) !important;
    border-radius: 18px !important;
    padding: 28px 28px 24px !important;
    transition: transform .25s ease, box-shadow .25s ease !important;
}
[data-testid="column"]:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 24px 60px rgba(0,0,0,.5), 0 0 0 1px rgba(124,58,237,.16) !important;
}

/* ── DIVIDER ─────────────────────────────────────────────────────── */
[data-testid="stDivider"] hr, hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg,
        transparent 0%,
        rgba(124,58,237,.48) 30%,
        rgba(6,182,212,.48)  70%,
        transparent 100%) !important;
    margin: 28px 0 !important;
}

/* ── ALERTS ──────────────────────────────────────────────────────── */
.stSuccess > div {
    background: rgba(16,185,129,.08) !important;
    border-left: 4px solid #10b981 !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
.stInfo > div {
    background: rgba(59,130,246,.08) !important;
    border-left: 4px solid #3b82f6 !important;
    border-radius: 10px !important;
    color: #93c5fd !important;
}
.stError > div {
    background: rgba(239,68,68,.08) !important;
    border-left: 4px solid #ef4444 !important;
    border-radius: 10px !important;
    color: #fca5a5 !important;
}

/* ── SPINNER ─────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #7c3aed !important; }
.stSpinner p     { color: #64748b !important; font-size: 13px !important; }

/* ── CAPTION ─────────────────────────────────────────────────────── */
.stCaption p { color: #475569 !important; font-size: 12px !important; }

/* ── MARKDOWN CONTENT ────────────────────────────────────────────── */
[data-testid="stMarkdownContainer"] p      { color: #94a3b8 !important; line-height: 1.72 !important; }
[data-testid="stMarkdownContainer"] li     { color: #94a3b8 !important; }
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3     { color: #e2e8f0 !important; font-weight: 600 !important; }
[data-testid="stMarkdownContainer"] strong { color: #c4b5fd !important; }
[data-testid="stMarkdownContainer"] code {
    background: rgba(124,58,237,.14) !important;
    color: #a78bfa !important;
    border-radius: 4px !important;
    padding: 1px 5px !important;
}

/* ── SUBHEADINGS ─────────────────────────────────────────────────── */
h1 { color: #f8fafc !important; font-weight: 800 !important; }
h2 { color: #e2e8f0 !important; font-weight: 600 !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; }

/* ── SCROLLBAR ───────────────────────────────────────────────────── */
::-webkit-scrollbar             { width: 5px; height: 5px; }
::-webkit-scrollbar-track       { background: rgba(255,255,255,.02); }
::-webkit-scrollbar-thumb       { background: rgba(124,58,237,.38); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(124,58,237,.65); }

/* ── HERO SECTION ────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 60px 20px 44px;
    position: relative;
    z-index: 1;
}
.hero-bot {
    display: inline-block;
    animation: botFloat 4.2s ease-in-out infinite;
    margin-bottom: 22px;
    filter: drop-shadow(0 0 26px rgba(124,58,237,.5));
}
@keyframes botFloat {
    0%,100% { transform: translateY(0px);   }
    50%      { transform: translateY(-13px); }
}
.hero-title {
    font-size: clamp(2rem, 5vw, 3.4rem);
    font-weight: 800;
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 45%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 12px;
    line-height: 1.13;
    letter-spacing: -.025em;
}
.hero-sub {
    color: #64748b;
    font-size: 17px;
    max-width: 540px;
    margin: 0 auto 10px;
    line-height: 1.65;
}
.gradient-bar {
    width: 72px; height: 3px;
    background: linear-gradient(90deg, #7c3aed, #06b6d4);
    border-radius: 2px;
    margin: 0 auto 26px;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(124,58,237,.14);
    border: 1px solid rgba(124,58,237,.28);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    color: #a78bfa;
    margin: 3px;
    transition: background .2s;
}
.badge:hover { background: rgba(124,58,237,.24); }

/* ── AGENT PROCESSING BANNER ─────────────────────────────────────── */
.agent-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    background: rgba(124,58,237,.07);
    border: 1px solid rgba(124,58,237,.2);
    border-radius: 14px;
    padding: 16px 22px;
    margin: 14px 0 10px;
    animation: bannerGlow 2.2s ease-in-out infinite alternate;
}
@keyframes bannerGlow {
    0%   { box-shadow: none; }
    100% { box-shadow: 0 0 32px rgba(124,58,237,.18); }
}
.agent-banner-text   { color: #a78bfa; font-weight: 500; font-size: 14px; margin: 0; }
.agent-banner-detail { color: #475569; font-size: 12px; margin: 3px 0 0; }
.dots {
    display: inline-flex; gap: 4px; align-items: center; margin-left: 6px;
}
.dots span {
    width: 5px; height: 5px;
    background: linear-gradient(135deg, #7c3aed, #06b6d4);
    border-radius: 50%;
    display: inline-block;
    animation: dotBounce 1.4s ease-in-out infinite;
}
.dots span:nth-child(2) { animation-delay: .2s; }
.dots span:nth-child(3) { animation-delay: .4s; }
@keyframes dotBounce {
    0%,80%,100% { transform: scale(.55); opacity: .35; }
    40%          { transform: scale(1);   opacity: 1;   }
}

/* ── REPORT AREA ─────────────────────────────────────────────────── */
.report-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 24px 0 8px;
    border-bottom: 1px solid rgba(255,255,255,.06);
    margin-bottom: 16px;
}
.report-header-title {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9;
}
.report-tag {
    background: rgba(16,185,129,.12);
    border: 1px solid rgba(16,185,129,.25);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    color: #6ee7b7;
}
</style>
""", unsafe_allow_html=True)


# ── HERO RENDER ────────────────────────────────────────────────────────────────

def _render_hero():
    st.markdown(f"""
<div class="hero">
    <div class="hero-bot">{_BOT_SVG_LARGE}</div>
    <div class="hero-title">Your AI Recruiting Agent</div>
    <div class="gradient-bar"></div>
    <div class="hero-sub">
        Multi-agent CV screening and ranking — swap LLM providers without changing a line of pipeline code.
    </div>
    <div style="margin-top:18px;">
        <span class="badge">&#9889; 3-Agent Pipeline</span>
        <span class="badge">&#129504; RAG + FAISS</span>
        <span class="badge">&#128301; LLM-Agnostic</span>
        <span class="badge">&#128202; Ranked Shortlist</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── SIDEBAR MASCOT ─────────────────────────────────────────────────────────────

def _sidebar_mascot():
    st.markdown(f"""
<div class="sb-mascot">
    <div class="sb-icon">{_BOT_SVG_SMALL}</div>
    <div>
        <div class="sb-name">HireBot</div>
        <div class="sb-tagline">AI Hiring Agent</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="AI Hiring Agent", page_icon="🤖", layout="wide")
    _inject_css()
    _render_hero()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        _sidebar_mascot()
        st.header("LLM Provider")
        env_provider = os.getenv("LLM_PROVIDER", "local_hf")
        default_label = next(
            (k for k, v in PROVIDERS.items() if v == env_provider),
            list(PROVIDERS.keys())[0],
        )
        provider_label = st.selectbox(
            "Select provider",
            list(PROVIDERS.keys()),
            index=list(PROVIDERS.keys()).index(default_label),
        )
        provider = PROVIDERS[provider_label]

        if provider == "gemini":
            os.environ["GEMINI_API_KEY"] = st.text_input(
                "Gemini API Key", type="password", value=os.getenv("GEMINI_API_KEY", "")
            )
            os.environ["GEMINI_MODEL"] = st.text_input(
                "Model", value=os.getenv("GEMINI_MODEL", "gemini/gemini-1.5-flash")
            )

        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = st.text_input(
                "OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", "")
            )
            os.environ["OPENAI_MODEL"] = st.text_input(
                "Model", value=os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            )

        elif provider == "hf_api":
            os.environ["HUGGINGFACE_API_TOKEN"] = st.text_input(
                "HuggingFace Token", type="password", value=os.getenv("HUGGINGFACE_API_TOKEN", "")
            )
            os.environ["HF_MODEL"] = st.text_input(
                "Model ID", value=os.getenv("HF_MODEL", "HuggingFaceH4/zephyr-7b-beta")
            )

        elif provider == "local_hf":
            os.environ["HF_MODEL"] = st.text_input(
                "Model ID", value=os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
            )
            st.caption("Model downloads once (~1 GB) and is cached locally.")

        elif provider == "ollama":
            os.environ["OLLAMA_MODEL"] = st.text_input(
                "Model", value=os.getenv("OLLAMA_MODEL", "llama3.2:3b")
            )
            os.environ["OLLAMA_BASE_URL"] = st.text_input(
                "Base URL", value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            )
            st.caption("Install Ollama from ollama.com, then run: `ollama pull llama3.2:3b`")

        os.environ["LLM_PROVIDER"] = provider

        st.divider()
        st.header("About")
        st.markdown(
            "**Agents**\n"
            "1. **Screener** — PASS/FAIL per candidate\n"
            "2. **Scorer** — 1-10 on skills, experience, alignment\n"
            "3. **Reporter** — ranked markdown shortlist\n\n"
            "**Embeddings**: `all-MiniLM-L6-v2` + FAISS (always local)"
        )

    # ── Two-column input layout ───────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Job Description")
        use_sample_jd = st.checkbox("Use sample job description", value=True)
        if use_sample_jd:
            jd = SAMPLE_JD
            st.text_area("Preview", SAMPLE_JD, height=320, disabled=True)
        else:
            jd = st.text_area("Paste job description", height=320, placeholder="Describe the role...")

    with col2:
        st.subheader("Candidate CVs")
        use_sample_cvs = st.checkbox("Use sample CVs (4 candidates)", value=True)
        uploaded = []
        if not use_sample_cvs:
            uploaded = st.file_uploader(
                "Upload CVs (.txt, .pdf, .csv)",
                accept_multiple_files=True,
                type=["txt", "pdf", "csv"],
            )
            if uploaded:
                st.caption(f"{len(uploaded)} file(s) selected: {', '.join(f.name for f in uploaded)}")

    st.divider()

    # ── Run button & pipeline ─────────────────────────────────────────────────
    if st.button("Run Hiring Pipeline", type="primary", use_container_width=True):
        if not jd.strip():
            st.error("Please provide a job description.")
            return

        with st.spinner("Loading CVs and building vector index..."):
            if use_sample_cvs:
                documents = parse_directory("sample_data")
            else:
                if not uploaded:
                    st.error("Please upload at least one CV.")
                    return
                documents = parse_uploaded_files(uploaded)

        if not documents:
            st.error("No CV documents loaded. Check sample_data/ or upload files.")
            return

        st.info(
            f"Loaded {len(documents)} candidate(s): "
            + ", ".join(f"**{n}**" for n, _ in documents)
        )

        est = "~2-5 min on CPU" if provider == "local_hf" else "~30-60 sec"
        st.markdown(f"""
<div class="agent-banner">
    {_BOT_SVG_THINKING}
    <div>
        <p class="agent-banner-text">
            Agents are working
            <span class="dots"><span></span><span></span><span></span></span>
        </p>
        <p class="agent-banner-detail">{est} &nbsp;·&nbsp; {provider}</p>
    </div>
</div>
""", unsafe_allow_html=True)

        with st.spinner(
            f"Running 3-agent pipeline via `{provider}` ({est})..."
        ):
            try:
                report = run_pipeline(jd, documents)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)
                return

        st.success("Pipeline complete!")

        st.markdown("""
<div class="report-header">
    <div class="report-header-title">Hiring Report</div>
    <span class="report-tag">&#10003; Complete</span>
</div>
""", unsafe_allow_html=True)

        st.markdown(report)

        st.download_button(
            label="Download report as Markdown",
            data=report,
            file_name="hiring_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
