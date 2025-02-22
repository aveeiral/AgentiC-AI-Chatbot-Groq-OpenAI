# Step 1. Setup API Keys for Groq and Tavily

import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("groq_api_key")
TAVILY_API_KEY = os.getenv("tavily_api_key")
OPENAI_API_KEY = os.getenv("openai_api_key")


# Step 2. Setup LLM & Tools

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm = ChatOpenAI(model = "gpt-4o-mini")
groq_llm = ChatGroq(model = "llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=2)

# Step 3. Setup AI Agent with Search tool functionality

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt = "Act as an AI Agent who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if provider=="Groq":
        llm = ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm = ChatOpenAI(model = llm_id)

    tools = [TavilySearchResults(max_results=2)] if allow_search else []

    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    #query = "What happened in Afganistan vs South Africa champions trophy match?"
    state = {"messages":query}
    response=agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]

    return (ai_messages[-1])

