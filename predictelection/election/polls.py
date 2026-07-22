from pydantic import BaseModel

class PollRow(BaseModel):
    candidate_name: str
    topline_percent_favoriability: float



class Poll(BaseModel):
    rows: PollRow
