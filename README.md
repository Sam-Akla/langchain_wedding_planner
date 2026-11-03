# Wedding Planner

This is an AI assistant for wedding planning. The project

3 agents:
    - Travel: Finding flights from and to your ideal destination. Consider total travel timer, budget, number of passengers.
    - Venue: Searches the web for wedding venue. Consider parameters such as wedding size, budget, location.
    - DJ: Searches a music database for a playlist or songs matching the right genre.

Main coordinator:
    - Coordinate the different agents together.

## TODO

- [ ] Fixing the response format.
  - The response format is not properly showing the necessary details
- [ ] Choose the balance between context and state.
- [ ] Connect sub agents to main agent

## Resources

### Frontend

    - https://github.com/shadcn-ui/chatbot-template
    - https://www.neobrutalism.dev

# Wedding Planner

An AI wedding planning assistant built with Python, LangChain, and Firecrawl. The project explores coordinating specialist agents to research travel, wedding venues, and music.

> **Status:** Early development. The command-line entry point currently runs a travel-agent demo. Venue, DJ, and orchestrator agents are defined, but their integration is incomplete.

## Overview

The application is organized around three specialist agents and a coordinator:

| Agent | Intended role | Current implementation |
| --- | --- | --- |
| Travel | Research flights using destinations, dates, passenger count, budget, and travel-time preferences | Web search and tools for reading travel preferences |
| Venue | Research venues using location, guest count, budget, and booking dates | Web search and tools for reading venue preferences |
| DJ | Find songs matching a genre, theme, and playlist length | Web search and tools for reading music preferences |
| Orchestrator | Coordinate the specialist agents | Delegation tools exist; request parameters and context forwarding need implementation |

Searches currently use Firecrawl. Dedicated flight, booking, and music database integrations are not implemented.

## Requirements

- Python 3.13 or later
- [uv](https://docs.astral.sh/uv/)
- A Firecrawl API key
- Access to an Azure OpenAI deployment through an OpenAI-compatible endpoint

## Installation

From the project directory, install the dependencies:

```bash
uv sync
```

## Configuration

Create a `.env` file in the project root:

```dotenv
FIRECRAWL_API_KEY=your-firecrawl-api-key

AZURE_OPENAI_MODEL=your-deployment-name
AZURE_OPENAI_API_KEY=your-azure-openai-api-key
AZURE_OPENAI_ENDPOINT=your-openai-compatible-base-url
```

The application loads these variables using `python-dotenv`.

`AZURE_OPENAI_MODEL` is passed to `ChatOpenAI` as the model name, and `AZURE_OPENAI_ENDPOINT` is passed directly as its `base_url`. Supply the compatible base URL for your deployment.

Keep API keys out of version control.

## Running the Travel Demo

```bash
uv run wedding-planner
```

The command invokes the travel agent with its default preferences and prints the resulting agent state.

| Preference | Default |
| --- | --- |
| Origin | Oslo |
| Destination | Prague |
| Travellers | 4 |
| Maximum budget | EUR 3,000 |
| Maximum total travel time | 10; unit not yet specified |
| Departure date | December 20, 2026 |
| Return date | January 5, 2027 |

The defaults are defined in `TravelContext` in `src/wedding_planner/agents/travel_agent.py`.

Running the demo makes external model and search requests, which may incur usage charges.

## Project Structure

```text
wedding_planner/
├── pyproject.toml             # Dependencies and command-line entry point
├── uv.lock                    # Locked dependency versions
├── README.md
├── src/
│   └── wedding_planner/
│       ├── __init__.py        # Travel demo entry point
│       ├── bootstrap.py       # Loads environment variables
│       ├── clients.py         # Initializes model and Firecrawl clients
│       └── agents/
│           ├── orchestrator_agent.py
│           ├── travel_agent.py
│           ├── venue_agent.py
│           └── dj_agent.py
├── tests/
└── ui/
```

## Architecture

Each specialist agent uses:

- **LangChain tools** to search the web and retrieve preferences.
- **Pydantic context models** to describe runtime inputs.
- **Custom agent state** to represent information available during execution.
- **Structured response models** to define output fields.
- **In-memory checkpointing** to retain conversation state within the running process.

The orchestrator exposes tools that invoke the specialist agents. These calls currently use placeholder requests and do not forward the context required by the specialists.

## Current Limitations

- The command-line application runs only the travel demo.
- Agent system prompts are placeholders.
- Structured responses describe planning preferences but do not yet include flight options, venue recommendations, or song lists.
- The boundary between runtime context and mutable state needs refinement.
- Checkpoints are stored in memory and do not persist across application restarts.
- A complete user interface and end-to-end planning workflow are not implemented.

## Roadmap

- [ ] Improve agent prompts and tool descriptions.
- [ ] Define useful response schemas for flights, venues, and playlists.
- [ ] Clarify which information belongs in context and state.
- [ ] Forward user requirements and context through the orchestrator.
- [ ] Validate the complete multi-agent workflow.
- [ ] Add tests for tools, configuration, and delegation.
- [ ] Build a user interface.

## Frontend References

- [shadcn/ui chatbot template](https://github.com/shadcn-ui/chatbot-template)
- [Neobrutalism components](https://www.neobrutalism.dev)
