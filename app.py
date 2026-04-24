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


def main():
    st.set_page_config(page_title="AI Hiring Agent", page_icon="🤖", layout="wide")
    st.title("AI Hiring Agent Pipeline")
    st.caption(
        "Multi-agent CV screening and ranking — LLM-agnostic. "
        "Swap providers via the sidebar without changing any pipeline code."
    )

    with st.sidebar:
        st.header("LLM Provider")
        env_provider = os.getenv("LLM_PROVIDER", "local_hf")
        default_label = next((k for k, v in PROVIDERS.items() if v == env_provider), list(PROVIDERS.keys())[0])
        provider_label = st.selectbox("Select provider", list(PROVIDERS.keys()), index=list(PROVIDERS.keys()).index(default_label))
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

        with st.spinner(
            f"Running 3-agent pipeline via `{provider}` "
            f"({'~2-5 min on CPU' if provider == 'local_hf' else '~30-60 sec'})..."
        ):
            try:
                report = run_pipeline(jd, documents)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.exception(exc)
                return

        st.success("Pipeline complete!")
        st.subheader("Hiring Report")
        st.markdown(report)

        st.download_button(
            label="Download report as Markdown",
            data=report,
            file_name="hiring_report.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
