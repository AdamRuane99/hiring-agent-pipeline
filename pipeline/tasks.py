from crewai import Agent, Task


def build_tasks(
    screener: Agent,
    scorer: Agent,
    reporter: Agent,
    job_description: str,
    candidate_names: list,
):
    names_str = ", ".join(candidate_names)

    screen_task = Task(
        description=(
            f"Job description:\n{job_description}\n\n"
            f"Candidates to evaluate: {names_str}\n\n"
            "For each candidate, use the search tool to retrieve relevant CV sections, then "
            "return PASS or FAIL against the required skills in the job description. "
            "List every candidate — do not skip any."
        ),
        expected_output="A PASS/FAIL list with one-line reasoning per candidate.",
        agent=screener,
    )

    score_task = Task(
        description=(
            f"Job description:\n{job_description}\n\n"
            "Using the screening results, score each PASS candidate on:\n"
            "- Technical Skills (1-10)\n"
            "- Relevant Experience (1-10)\n"
            "- Role Alignment (1-10)\n"
            "Search the CV index for evidence and cite it for each score."
        ),
        expected_output="Scores with cited evidence per passing candidate.",
        agent=scorer,
        context=[screen_task],
    )

    report_task = Task(
        description=(
            "Using the scores, produce a final ranked markdown shortlist. "
            "For each candidate include: rank, name, composite score (mean of three criteria), "
            "and a two-sentence hiring summary. Sort descending by composite score. "
            "Also include a one-line note on any FAIL candidates at the bottom."
        ),
        expected_output="Markdown hiring report ready to send to the hiring manager.",
        agent=reporter,
        context=[score_task],
    )

    return screen_task, score_task, report_task
