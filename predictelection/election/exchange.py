


from predictelection.clients.kalshi import Event

class ElectionEvent(Event):
    party: str


async def find_race_market_ticker(candidate_term: str, *, race_hint: str | None = None) -> str | None:
    """Find """

async def find_race_event_ticker(search_term: str) -> str | None:
    """Find election race ticker"""
