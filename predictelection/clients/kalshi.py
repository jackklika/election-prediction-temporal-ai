"""
Wrapper around kalshi client, with our config and models.

We redefine the response models since we want full control of the models,
ability to pre-process and make custom properties, and also validation errors if the kalshi API changes.
"""

import base64
from datetime import datetime, timezone
from decimal import Decimal
from typing import Annotated

from pydantic import (
    AliasChoices,
    BaseModel,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from predictelection.clients._base_config import ConfigBase

import kalshi_python_async as kp


Dollars = Annotated[Decimal, Field(max_digits=28, decimal_places=6)]


class KalshiConfig(ConfigBase):
    api_key_id: str = Field(validation_alias=AliasChoices("kalshi_api_key_id"))
    private_key_base64: str = Field(
        validation_alias=AliasChoices("kalshi_private_key_base64")
    )

    private_key: bytes = b""  # to be populated by validator

    @model_validator(mode="after")
    def _decode_private_key(self):
        self.private_key = base64.b64decode(self.private_key_base64)
        return self


class Balance(BaseModel):
    balance: StrictInt  # balance in cents
    balance_dollars: Dollars  # balance in dollars
    portfolio_value: StrictInt
    updated_at: datetime

    @field_validator("balance_dollars", mode="before")
    @classmethod
    def _parse_dollars(cls, v: str | Decimal) -> Decimal:
        return Decimal(v)

    @classmethod
    def from_balance_resp(cls, resp: kp.GetBalanceResponse) -> "Balance":
        return cls(
            balance=resp.balance,
            balance_dollars=Decimal(resp.balance_dollars),
            portfolio_value=resp.portfolio_value,
            updated_at=datetime.fromtimestamp(resp.updated_ts, tz=timezone.utc),
        )

class Event(BaseModel):
    event_ticker: StrictStr
    series_ticker: StrictStr
    sub_title: StrictStr
    title: StrictStr
    markets: list[Market] | None = None

    settlement_sources: list[SettlementSource] | None = Field(description="The official sources used for the determination of markets within this event. Methodology is defined in the rulebook.")

    collateral_return_type: StrictStr = Field(description="Specifies how collateral is returned when markets settle (e.g., 'binary' for standard yes/no markets).")
    mutually_exclusive: StrictBool = Field(description="If true, only one market in this event can resolve to 'yes'. If false, multiple markets can resolve to 'yes'.")

    @classmethod
    def from_event_response(cls, resp: kp.GetEventResponse) -> "Event":
        _event: kp.EventData = resp.event
        return Event.from_kalshi_model(_event)

    @classmethod
    def from_kalshi_model(cls, event: kp.EventData) -> "Event":
        return cls(
            event_ticker=event.event_ticker,
            series_ticker=event.series_ticker,
            markets=[Market.from_kalshi_market(m) for m in event.markets] if event.markets is not None else None,
            sub_title=event.sub_title,
            title=event.title,
            settlement_sources=[SettlementSource.from_kalshi_model(s) for s in event.settlement_sources] if event.settlement_sources else None,
            collateral_return_type=event.collateral_return_type,
            mutually_exclusive=event.mutually_exclusive,
        )

class SettlementSource(BaseModel):
    name: StrictStr | None = Field(default=None, description="Name of the settlement source")
    url: StrictStr | None = Field(default=None, description="URL to the settlement source")

    @classmethod
    def from_kalshi_model(cls, model: kp.SettlementSource) -> SettlementSource:
        return SettlementSource(
            name=model.name,
            url=model.url,
        )


class Market(BaseModel):
    ticker: StrictStr
    event_ticker: StrictStr
    market_type: StrictStr = Field(description="Identifies the type of market")

    title: StrictStr | None = None
    subtitle: StrictStr | None = None

    result: StrictStr | None = None
    rules_primary: StrictStr = Field(description="A plain language description of the most important market terms")
    rules_secondary: StrictStr = Field(description="A plain language description of secondary market terms")

    last_price_dollars: Dollars = Field(description="US dollar amount as a fixed-point decimal string with up to 6 decimal places of precision. This is the maximum supported precision; valid quote intervals for a given market are constrained by that market's price level structure.")
    yes_bid_dollars: Dollars = Field(description="US dollar amount as a fixed-point decimal string with up to 6 decimal places of precision. This is the maximum supported precision; valid quote intervals for a given market are constrained by that market's price level structure.")
    yes_bid_size_fp: StrictStr = Field(description="Fixed-point contract count string (2 decimals, e.g., \"10.00\"; referred to as \"fp\" in field names). Requests accept 0-2 decimal places (e.g., \"10\", \"10.0\", \"10.00\"); responses always emit 2 decimals. Fractional contract values (e.g., \"2.50\") are supported; the minimum granularity is 0.01 contracts.")
    yes_ask_dollars: Dollars = Field(description="US dollar amount as a fixed-point decimal string with up to 6 decimal places of precision. This is the maximum supported precision; valid quote intervals for a given market are constrained by that market's price level structure.")
    yes_ask_size_fp: StrictStr = Field(description="Fixed-point contract count string (2 decimals, e.g., \"10.00\"; referred to as \"fp\" in field names). Requests accept 0-2 decimal places (e.g., \"10\", \"10.0\", \"10.00\"); responses always emit 2 decimals. Fractional contract values (e.g., \"2.50\") are supported; the minimum granularity is 0.01 contracts.")
    no_bid_dollars: Dollars = Field(description="US dollar amount as a fixed-point decimal string with up to 6 decimal places of precision. This is the maximum supported precision; valid quote intervals for a given market are constrained by that market's price level structure.")
    no_ask_dollars: Dollars = Field(description="US dollar amount as a fixed-point decimal string with up to 6 decimal places of precision. This is the maximum supported precision; valid quote intervals for a given market are constrained by that market's price level structure.")

    volume_fp: StrictStr = Field(description="Fixed-point contract count string (2 decimals, e.g., \"10.00\"; referred to as \"fp\" in field names). Requests accept 0-2 decimal places (e.g., \"10\", \"10.0\", \"10.00\"); responses always emit 2 decimals. Fractional contract values (e.g., \"2.50\") are supported; the minimum granularity is 0.01 contracts.")
    volume_24h_fp: StrictStr = Field(description="Fixed-point contract count string (2 decimals, e.g., \"10.00\"; referred to as \"fp\" in field names). Requests accept 0-2 decimal places (e.g., \"10\", \"10.0\", \"10.00\"); responses always emit 2 decimals. Fractional contract values (e.g., \"2.50\") are supported; the minimum granularity is 0.01 contracts.")
    liquidity_dollars: Dollars = Field(description="US dollar amount as a fixed-point decimal string with up to 6 decimal places of precision. This is the maximum supported precision; valid quote intervals for a given market are constrained by that market's price level structure.")
    open_interest_fp: StrictStr = Field(description="Fixed-point contract count string (2 decimals, e.g., \"10.00\"; referred to as \"fp\" in field names). Requests accept 0-2 decimal places (e.g., \"10\", \"10.0\", \"10.00\"); responses always emit 2 decimals. Fractional contract values (e.g., \"2.50\") are supported; the minimum granularity is 0.01 contracts.")

    occurrence_datetime: datetime | None = Field(default=None, description="The recorded datetime when the underlying event occurred, if available")
    created_time: datetime
    updated_time: datetime = Field(description="Time of the last non-trading metadata update.")
    open_time: datetime
    close_time: datetime

    @classmethod
    def from_kalshi_market(cls, market: kp.Market) -> "Market":
        return cls(
            ticker=market.ticker,
            event_ticker=market.event_ticker,
            market_type=market.market_type,

            title=market.title,
            subtitle=market.subtitle,

            result=market.result,
            rules_primary=market.rules_primary,
            rules_secondary=market.rules_secondary,

            last_price_dollars=Decimal(market.last_price_dollars),
            yes_bid_dollars=Decimal(market.yes_bid_dollars),
            yes_bid_size_fp=market.yes_bid_size_fp,
            yes_ask_dollars=Decimal(market.yes_ask_dollars),
            yes_ask_size_fp=market.yes_ask_size_fp,
            no_bid_dollars=Decimal(market.no_bid_dollars),
            no_ask_dollars=Decimal(market.no_ask_dollars),

            volume_fp=market.volume_fp,
            volume_24h_fp=market.volume_24h_fp,
            liquidity_dollars=Decimal(market.liquidity_dollars),
            open_interest_fp=market.open_interest_fp,

            occurrence_datetime=market.occurrence_datetime,
            created_time=market.created_time,
            updated_time=market.updated_time,
            open_time=market.open_time,
            close_time=market.close_time,
        )

class KalshiClient:
    def __init__(self, *, config: KalshiConfig | None = None):
        self._config: KalshiConfig = KalshiConfig() if not config else config  # ty: ignore[missing-argument]

        _kalshi_config = kp.Configuration(
            host="https://api.elections.kalshi.com/trade-api/v2",
        )
        _kalshi_config.api_key_id = self._config.api_key_id
        _kalshi_config.private_key_pem = self._config.private_key

        self._client: kp.KalshiClient = kp.KalshiClient(configuration=_kalshi_config)

    async def get_balance(self) -> Balance:
        _resp: kp.GetBalanceResponse = await self._client.get_balance()
        return Balance.from_balance_resp(_resp)

    async def get_event(self, event_ticker: str, *, with_nested_markets: bool = True) -> Event:
        _resp: kp.GetEventResponse = await self._client.get_event(event_ticker=event_ticker, with_nested_markets=with_nested_markets)
        return Event.from_event_response(_resp)
