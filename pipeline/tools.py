from crewai.tools import tool
from core.rag import CVIndex

_index: CVIndex | None = None


def set_index(index: CVIndex):
    global _index
    _index = index


@tool
def search_candidates(query: str) -> str:
    """Search candidate CVs for skills, experience, or keywords. Input: a natural language query."""
    if _index is None:
        return "No CV index loaded."
    results = _index.search(query, top_k=5)
    if not results:
        return "No relevant content found."
    lines = []
    for r in results:
        lines.append(f"[{r['meta']['name']} | score={r['score']:.3f}]\n{r['chunk']}\n")
    return "\n---\n".join(lines)
