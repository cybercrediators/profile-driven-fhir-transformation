"""Prompt template handling"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from llm.errors import LLMDependencyError

PROMPT_DIR = Path(__file__).parent

# Templates shipped with the package
AUTOMAPPING_SYSTEM = "automapping_system.j2"
AUTOMAPPING_USER = "automapping_user.j2"
AGENT_FIX_SYSTEM = "agent_fix_system.j2"
AGENT_FIX_USER = "agent_fix_user.j2"


@lru_cache(maxsize=1)
def _environment():
    """Build the Jinja2 environment once, importing the optional dependency lazily."""

    try:
        from jinja2 import Environment, FileSystemLoader, StrictUndefined
    except ImportError as exc:
        raise LLMDependencyError("jinja2") from exc

    return Environment(
        loader=FileSystemLoader(str(PROMPT_DIR)),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
        autoescape=False,
    )


def render(template_name: str, **context: Any) -> str:
    """Render template_name from this directory with corresponding context

    :raises LLMDependencyError: when the optional ``jinja2`` package is absent.
    :raises jinja2.TemplateNotFound: when no such template ships with the package.
    :raises jinja2.UndefinedError: when the template needs a variable that the
        caller did not supply.
    """

    return _environment().get_template(template_name).render(**context).strip()


def available_templates() -> list[str]:
    """return existing template names"""

    return sorted(path.name for path in PROMPT_DIR.glob("*.j2"))
