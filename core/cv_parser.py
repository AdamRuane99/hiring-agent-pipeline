import io
from pathlib import Path
from typing import List, Tuple


def parse_uploaded_files(uploaded_files) -> List[Tuple[str, str]]:
    """Streamlit UploadedFile list → [(candidate_name, text)]"""
    results = []
    for f in uploaded_files:
        text = _extract(f)
        if text.strip():
            results.append((Path(f.name).stem, text))
    return results


def parse_directory(path: str) -> List[Tuple[str, str]]:
    """Load all .txt files from a directory → [(stem, text)]"""
    results = []
    for fp in sorted(Path(path).glob("*.txt")):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            results.append((fp.stem, text))
    return results


def _extract(file) -> str:
    name = file.name.lower()
    raw = file.read()

    if name.endswith(".txt"):
        return raw.decode("utf-8", errors="ignore")

    if name.endswith(".pdf"):
        try:
            import pdfminer.high_level as pdfminer
            return pdfminer.extract_text(io.BytesIO(raw))
        except Exception:
            return ""

    if name.endswith(".csv"):
        import csv
        import io as _io
        reader = csv.reader(_io.StringIO(raw.decode("utf-8", errors="ignore")))
        return "\n".join(",".join(row) for row in reader)

    return ""
