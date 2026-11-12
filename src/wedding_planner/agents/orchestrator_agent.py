from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wedding_planner.agents.dj_agent import dj_agent
from wedding_planner.agents.travel_agent import (
    TravelContext,
    TravelLocations,
    travel_agent,
)
from wedding_planner.agents.venue_agent import VenueContext, venue_agent
from wedding_planner.clients import MODEL

SYS_PMPT_ORC_AGENT = """
You are an agent
"""


class OrchestratorContext(BaseModel):
    x: str = "10"


class OrchestratorResponse(BaseModel):
    x: str = "10"


class OrchestratorState(AgentState):
    x: str


@tool(description="Calling the travel agent")
async def call_travel_agent(
    request: str,
    travel_locations: TravelLocations,
    n_travellers: int,
    currency: str = "EUR",
):
    response = await travel_agent.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=f"{request}. Find flights from {travel_locations.origin} to {travel_locations.destination} with {n_travellers} in currency {currency}"
                )
            ]
        },
        context=TravelContext(
            travel_locations=travel_locations,
            n_travellers=n_travellers,
            currency=currency,
        ),
    )
    return response["structured_response"]


@tool(description="Calling the venue agent")
async def call_venue_agent(venue_location: str, request: str, currency: str = "EUR"):
    response = await venue_agent.ainvoke(
        {"messages": [HumanMessage(content=f"{request} venue in {venue_location}")]},
        context=VenueContext(venue_location=venue_location, currency=currency),
    )
    return response["structured_response"]


@tool(description="Calling the dj agent")
async def call_dj_agent():
    response = await dj_agent.ainvoke(
        {"messages": [HumanMessage(content=f"Find songs")]}
    )
    return response["messages"][-1].content


orchestrator_agent = create_agent(
    model=MODEL,
    system_prompt=SYS_PMPT_ORC_AGENT,
    tools=[call_dj_agent, call_travel_agent, call_venue_agent],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=MODEL, trigger=("tokens", 100), keep=("messages", 1)
        ),
    ],
)
