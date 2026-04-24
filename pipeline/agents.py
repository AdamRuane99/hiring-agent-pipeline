from crewai import Agent
from core.llm_factory import get_llm
from pipeline.tools import search_candidates


def build_agents(llm=None):
    llm = llm or get_llm()

    screener = Agent(
        role="CV Screener",
        goal=(
            "Determine which candidates meet the minimum requirements of the job description. "
            "Return a clear PASS or FAIL with one sentence of reasoning for each candidate."
        ),
        backstory=(
            "You are a rigorous talent acquisition specialist who filters out candidates lacking "
            "required skills before deeper evaluation."
        ),
        tools=[search_candidates],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    scorer = Agent(
        role="Candidate Scorer",
        goal=(
            "Score each passing candidate 1-10 across three criteria: "
            "Technical Skills, Relevant Experience, and Role Alignment. "
            "Ground every score in evidence directly from the CV."
        ),
        backstory=(
            "You are a senior technical recruiter who evaluates candidates objectively, "
            "citing specific CV evidence for every score you assign."
        ),
        tools=[search_candidates],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    reporter = Agent(
        role="Hiring Report Writer",
        goal=(
            "Produce a ranked shortlist with a composite score and a two-sentence summary per candidate. "
            "Output clean markdown sorted by composite score descending."
        ),
        backstory=(
            "You are an executive recruiter who communicates findings clearly to hiring managers. "
            "Your reports are concise, evidence-based, and immediately actionable."
        ),
        tools=[],
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )

    return screener, scorer, reporter
