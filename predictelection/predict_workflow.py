
from datetime import datetime

from pydantic import BaseModel, Field
from temporalio import workflow


class FindPollsReq(BaseModel):
    candidates: list[str]
    race_hint: str = Field(description="Hint to llm of what race this is, ie state, primary, etc")
    time_range: tuple[datetime, datetime] | None = None

class FindPollsResp(BaseModel):
    pass


@workflow.defn
class FindPolls:

    @workflow.run
    async def run(self, req: FindPollsReq) -> FindPollsResp:
        return FindPollsResp()
