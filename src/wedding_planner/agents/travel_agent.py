import datetime

from firecrawl.types import SearchData
from langchain.agents import AgentState, create_agent
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wedding_planner.clients import FC_APP, MODEL


class TravelContext(BaseModel):
    travel_locations: dict = {"origin": "Oslo", "destination": "Prague"}
    n_travellers: int = 4
    max_budget: int = 3000
    currency: str = "EUR"
    total_travel_time: int = 10
    travel_dates: list[datetime.date] = [
        datetime.date(2026, 12, 20),
        datetime.date(2027, 1, 5),
    ]


class TravelState(AgentState):
    travel_locations: dict
    n_travellers: int
    max_budget: int
    currency: str
    total_travel_time: int
    travel_dates: list[datetime.date]


class TravelResponse(BaseModel):
    home: str
    destination: str
    n_travellers: int
    max_budget: int
    currency: str
    total_travel_time: int
    travel_dates: list[datetime.date]


SYS_PMPT_TRAVEL_AGENT = """
This is travel agent
"""


@tool
def get_travel_plan(query: str, destination: str) -> SearchData:
    """search the web for information"""
    return FC_APP.search(
        query=query, sources=["web", "news"], limit=7, location=destination
    )


@tool(description="Getting the home and destination location for the flight")
def get_travel_locations(runtime: ToolRuntime[TravelContext]) -> dict:
    return runtime.context.travel_locations


@tool(description="Getting the number of passengers for the flight")
def get_n_travellers(runtime: ToolRuntime[TravelContext]) -> int:
    return runtime.context.n_travellers


@tool(description="Getting the max budeget for the flight")
def get_max_budget(runtime: ToolRuntime[TravelContext]) -> int:
    return runtime.context.max_budget


@tool(description="Getting the currency for the flight")
def get_travel_currency(runtime: ToolRuntime[TravelContext]) -> str:
    return runtime.context.currency


@tool(description="Getting the total travel time for the flight")
def get_total_travel_time(runtime: ToolRuntime[TravelContext]) -> int:
    return runtime.context.total_travel_time


@tool(description="Getting the departure and return dates for the flight")
def get_travel_dates(runtime: ToolRuntime[TravelContext]) -> list[datetime.date]:
    return runtime.context.travel_dates


travel_agent = create_agent(
    model=MODEL,
    system_prompt=SYS_PMPT_TRAVEL_AGENT,
    tools=[
        get_travel_plan,
        get_travel_dates,
        get_travel_locations,
        get_n_travellers,
        get_max_budget,
        get_travel_currency,
        get_total_travel_time,
    ],
    context_schema=TravelContext,
    state_schema=TravelState,
    response_format=TravelResponse,
    checkpointer=InMemorySaver(),
)
