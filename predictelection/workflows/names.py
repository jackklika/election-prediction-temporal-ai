"""Workflow names, importable without constructing an agent.

Temporal dispatches on a bare string, so a client only needs the name. Keeping
the names here rather than in `workflows/registry.py` is what lets the trigger
CLI start a run without an ANTHROPIC_API_KEY: importing a workflow class builds
its agent, because PydanticAIWorkflow reads `__pydantic_ai_agents__` at
class-definition time.
"""

from __future__ import annotations

from typing import Final


RESEARCH_WORKFLOWS: Final[dict[str, str]] = {
    "debates": "ResearchDebatesWorkflow",
    "structure": "ResearchStructureWorkflow",
    "candidacies": "ResearchCandidaciesWorkflow",
}
"""CLI name to workflow name. Add a research domain here and it is triggerable."""

DEFAULT_RESEARCH_WORKFLOW: Final[str] = "debates"
