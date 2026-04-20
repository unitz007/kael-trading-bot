"""HTML template rendering for the web UI.

Uses Jinja2 templates stored in the ``templates/`` directory alongside
this package.  Provides helpers that are available inside every template.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

_TEMPLATES_DIR = Path(__file__).parent / "templates"

_env = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(default_for_string=False, default=True),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **context: object) -> str:
    """Render a Jinja2 template with the given context variables."""
    template = _env.get_template(template_name)
    return template.render(**context)


def get_template_names() -> list[str]:
    """Return a list of all available template names (for debugging)."""
    return _env.list_templates()
