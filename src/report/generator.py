"""Auditable Markdown and HTML report generation."""

from pathlib import Path
from typing import Any
import html


def generate_report(
    title: str, sections: dict[str, Any], destination: str | Path, format: str = "markdown"
) -> Path:
    """Generate a deterministic report and return its path."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if format == "markdown":
        body = f"# {title}\n\n" + "\n\n".join(
            f"## {key}\n\n{value}" for key, value in sections.items()
        )
    elif format == "html":
        body = (
            "<!doctype html><meta charset='utf-8'><title>"
            + html.escape(title)
            + "</title><h1>"
            + html.escape(title)
            + "</h1>"
            + "".join(
                f"<h2>{html.escape(str(key))}</h2><pre>{html.escape(str(value))}</pre>"
                for key, value in sections.items()
            )
        )
    else:
        raise ValueError("format must be markdown or html")
    path.write_text(body, encoding="utf-8")
    return path
