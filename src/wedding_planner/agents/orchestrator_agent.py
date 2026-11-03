from langchain.agents import AgentState, create_agent
from langchain.messages import HumanMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from wedding_planner.agents.dj_agent import dj_agent
from wedding_planner.agents.travel_agent import travel_agent
from wedding_planner.agents.venue_agent import venue_agent
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
async def call_travel_agent():
    origin = ""
    destination = ""
    response = await travel_agent.ainvoke(
        {
            "messages": [
                HumanMessage(content=f"Find flights from {origin} to {destination}")
            ]
        }
    )
    return response["messages"][-1].content


@tool(description="Calling the venue agent")
async def call_venue_agent():
    destination = ""
    response = await venue_agent.ainvoke(
        {"messages": [HumanMessage(content=f"Find venue in {destination}")]}
    )
    return response["messages"][-1].content


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
)
