"""Every workflow the worker runs.

Separate from the worker so adding a research domain does not mean editing the
runtime. The worker's job is to poll a task queue; which workflows exist is a
fact about this package, and it belongs next to them.

Importing this reaches an LLM client, because a PydanticAIWorkflow's agent has
to exist by class-definition time. Keep it out of module scope anywhere that
only handles contracts or database writes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from predictelection.workflows.candidacies import ResearchCandidaciesWorkflow
from predictelection.workflows.debates import ResearchDebatesWorkflow
from predictelection.workflows.names import RESEARCH_WORKFLOWS
from predictelection.workflows.structure import ResearchStructureWorkflow


WORKFLOWS: Sequence[Any] = (
    ResearchDebatesWorkflow,
    ResearchStructureWorkflow,
    ResearchCandidaciesWorkflow,
)

# The CLI dispatches by name and the worker by class; a name the worker does not
# register is a run that would sit in the queue forever, so tie them together.
assert {workflow.__name__ for workflow in WORKFLOWS} == set(
    RESEARCH_WORKFLOWS.values()
), "workflows/names.py and the registered workflows disagree"
