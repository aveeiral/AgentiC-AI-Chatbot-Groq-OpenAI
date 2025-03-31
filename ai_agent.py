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


# Function to check relevance & get a score
def check_relevance_and_suggest(role, query, llm):
    prompt = f"""
    Your role: {role}
    
    Question: {query}
    
    Task: 
    - Provide a relevance score (0-100) for how related this question is to the given role.
    - Suggest 3 questions that are related to the given role.

    Response format:
    Relevance Score: <score>
    Suggestions:
    - <Question 1>
    - <Question 2>
    - <Question 3>
    """

    response = llm.invoke(prompt).content  

    # Extract relevance score safely
    score = 0
    suggestions = None

    try:
        score_line = next(line for line in response.split("\n") if "Relevance Score:" in line)
        score = int(score_line.split(":")[-1].strip())

        if "Suggestions:" in response:
            suggestions_section = response.split("Suggestions:\n", 1)[-1]
            suggestions = suggestions_section.strip().split("\n")
    except Exception as e:
        print("Error parsing LLM response:", e)

    return score, suggestions

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider, role_strictness):
    if provider=="Groq":
        llm = ChatGroq(model=llm_id)
    elif provider=="OpenAI":
        llm = ChatOpenAI(model = llm_id)

     # Check score the query is relevant/suggestions
    score, suggestions = check_relevance_and_suggest(system_prompt, query, llm)

    if score < role_strictness:
        return f"Sorry, I can only answer questions related to my assigned role. (Role Relevance Score: {score}).\n\nHere are some example questions you can ask:\n" + "\n".join(suggestions)
    
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


# llm_id = "gpt-4o-mini"
# query = "Tell me about wheels"
# allow_search = False
# system_prompt = "Act as a car mechanic"
# provider = "OpenAI"
# role_strictness = 80

# mess = get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider, role_strictness)
# print(mess)

