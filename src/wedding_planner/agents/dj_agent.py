# TODO: Complete the system prompt for the dj agent
# TODO: Create a get context tool for the dj agent

from typing import Literal

from firecrawl.types import SearchData
from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from wedding_planner.clients import FC_APP, MODEL


class DJContext(BaseModel):
    """Defaults used when starting a new conversation."""

    genre: str = "pop"
    theme: str = "wedding"
    n_songs: int = Field(default=100, ge=1)


class DJState(AgentState):
    """Current preferences for this conversation."""

    genre: str
    theme: str
    n_songs: int


class SongRecommendation(BaseModel):
    title: str
    artist: str
    reason: str = Field(description="Why this song fits the requested genre and theme.")


class DJResponse(BaseModel):
    """Final recommendations returned to the user."""

    genre: str = Field(description="Genre used for the recommendations.")
    theme: str = Field(description="Theme used for the recommendations.")
    songs: list[SongRecommendation]
    status: Literal["needs_details", "recommendations", "no_results"]
    message: str
    questions: list[str] = Field(default_factory=list)


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
def update_n_songs(n_songs: int, runtime: ToolRuntime) -> Command:
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
