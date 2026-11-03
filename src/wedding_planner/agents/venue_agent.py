import datetime

from firecrawl.types import SearchData
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wedding_planner.clients import FC_APP, MODEL


class VenueContext(BaseModel):
    destination: str = ""
    n_guests: int = 30
    budget: int = 10_000
    currency: str = "EUR"
    booking_dates: list[datetime.date]


class VenueState(AgentState):
    destination: str
    n_guests: int
    budget: int
    currency: str
    booking_dates: list[datetime.date]


class VenueResponse(BaseModel):
    destination: str
    n_guests: int
    budget: int
    currency: str
    booking_dates: list[datetime.date]


SYS_PMPT_VENUE_AGENT = """
This is a venue agent
"""


@tool(description="Getting the location for the venue")
def get_destination(runtime: ToolRuntime[VenueContext]) -> str:
    return runtime.context.destination


@tool(description="Getting the number of guests for venue")
def get_n_guests(runtime: ToolRuntime[VenueContext]) -> int:
    return runtime.context.n_guests


@tool(description="Getting the max budeget for the venue")
def get_budget(runtime: ToolRuntime[VenueContext]) -> int:
    return runtime.context.budget


@tool(description="Getting the currency for the venue")
def get_venue_currency(runtime: ToolRuntime[VenueContext]) -> str:
    return runtime.context.currency


@tool(description="Getting the booking dates for the venue")
def get_booking_dates(runtime: ToolRuntime[VenueContext]) -> list[datetime.date]:
    return runtime.context.booking_dates


@tool
def get_venue(query: str) -> SearchData:
    """search the web for information"""
    return FC_APP.search(query, limit=5)


venue_agent = create_agent(
    model=MODEL,
    system_prompt=SYS_PMPT_VENUE_AGENT,
    tools=[
        get_venue,
        get_destination,
        get_n_guests,
        get_budget,
        get_venue_currency,
        get_booking_dates,
    ],
    context_schema=VenueContext,
    state_schema=VenueState,
    response_format=VenueResponse,
    checkpointer=InMemorySaver(),
)
