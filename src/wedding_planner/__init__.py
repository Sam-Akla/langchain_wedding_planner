from rich.pretty import pprint

from wedding_planner.agents.travel_agent import TravelContext, travel_agent


def main() -> None:
    print("Hello from wedding-planner!")
    result = travel_agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "search for flights from my home to destination.",
                }
            ]
        },
        context=TravelContext(),
        config={"configurable": {"thread_id": "travel-demo"}},
    )
    pprint(result)
