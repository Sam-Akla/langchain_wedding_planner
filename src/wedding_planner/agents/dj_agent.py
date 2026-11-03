from firecrawl.types import SearchData
from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from wedding_planner.clients import FC_APP, MODEL


class DJContext(BaseModel):
    genre: str = "pop"
    theme: str = "wedding"
    n_songs: int = 100


class DJState(AgentState):
    genre: str
    theme: str
    n_songs: int


class DJResponse(BaseModel):
    genre: str
    theme: str
    n_songs: int


SYS_PMPT_DJ_AGENT = """
This is travel agent
"""


@tool(description="Getting the genre of songs for the playlist")
def get_genre(runtime: ToolRuntime[DJContext]) -> str:
    return runtime.context.genre


@tool(description="Getting the theme for the songs")
def get_theme(runtime: ToolRuntime[DJContext]) -> str:
    return runtime.context.theme


@tool(description="Getting the total number of songs for the playlist")
def get_n_songs(runtime: ToolRuntime[DJState]) -> int:
    return runtime.state["n_songs"]


@tool(description="Getting the total number of songs for the playlist")
def update_n_songs(n_songs: int, runtime: ToolRuntime) -> int:
    return Command(
        update={
            "messages": [
                ToolMessage(
                    "Successfully updated the color", tool_call_id=runtime.tool_call_id
                )
            ],
            "n_songs": n_songs,
        }
    )


@tool
def get_songs(query: str, destination: str) -> SearchData:
    """search the web for information"""
    return FC_APP.search(
        query=query, sources=["web", "news"], limit=7, location=destination
    )


dj_agent = create_agent(
    model=MODEL,
    system_prompt=SYS_PMPT_DJ_AGENT,
    tools=[get_songs, get_genre, get_theme, get_n_songs],
    context_schema=DJContext,
    state_schema=DJState,
    response_format=DJResponse,
    checkpointer=InMemorySaver(),
)
