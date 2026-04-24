from crewai import Crew, Process

from core.llm_factory import get_llm
from core.rag import CVIndex
from pipeline import tools as _tools
from pipeline.agents import build_agents
from pipeline.tasks import build_tasks


def run_pipeline(
    job_description: str,
    documents: list,  # [(candidate_name, text)]
    llm=None,
) -> str:
    index = CVIndex()
    index.build(documents)
    _tools.set_index(index)

    llm = llm or get_llm()
    screener, scorer, reporter = build_agents(llm)

    candidate_names = [name for name, _ in documents]
    screen_task, score_task, report_task = build_tasks(
        screener, scorer, reporter, job_description, candidate_names
    )

    crew = Crew(
        agents=[screener, scorer, reporter],
        tasks=[screen_task, score_task, report_task],
        process=Process.sequential,
        verbose=False,
        memory=False,
    )

    result = crew.kickoff()
    return str(result)
