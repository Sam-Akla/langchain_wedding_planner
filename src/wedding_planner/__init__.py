import asyncio

from rich.pretty import pprint

from wedding_planner.agents.orchestrator_agent import (
    OrchestratorContext,
    orchestrator_agent,
)
from wedding_planner.agents.travel_agent import (
    TravelContext,
    TravelLocations,
    travel_agent,
)


async def run() -> None:
    result = await travel_agent.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Help me find flights from Oslo to Prague for 2 people. Dates are 22.12.2026 to 03.01.2027, currency in EURO. Let me know about missing details",
                }
            ]
        },
        context=TravelContext(
            travel_locations=TravelLocations(
                origin="Oslo",
                destination="Prague",
            ),
            n_travellers=4,
            currency="EUR",
        ),
        config={"configurable": {"thread_id": "travel-demo"}},
    )

    pprint(result["structured_response"])


def main() -> None:
    asyncio.run(run())
