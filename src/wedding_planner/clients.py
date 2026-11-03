import os

from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_openai import ChatOpenAI

load_dotenv()

FC_API_KEY = os.getenv("FIRECRAWL_API_KEY")

if not FC_API_KEY:
    raise RuntimeError("FIRECRAWL_API_KEY environment variable is not set")

FC_APP = Firecrawl(api_key=FC_API_KEY)


model_name = os.getenv("AZURE_OPENAI_MODEL")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

if not model_name or not api_key or not endpoint:
    raise RuntimeError("One or multiple environmental variables are missing")

MODEL = ChatOpenAI(model=model_name, base_url=endpoint, api_key=api_key)
