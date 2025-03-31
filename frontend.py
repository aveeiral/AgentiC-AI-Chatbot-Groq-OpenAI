# step 1. Setup UI with streamlit (model provider, model, system prompt, web_search, query)
import streamlit as st
import requests  # used in step 2

st.set_page_config(page_title="langGraph Agent Ui", layout="centered")
st.title("AI Chatbot Agents")
st.write("Create and Interact with AI Agents!")

system_prompt=st.text_area("Define Your AI Agent: ", height = 70, placeholder="Type your system prompt here...")
# Create a slider with min=0, max=100, step=5
role_threshold = st.slider("Role Strictness:", min_value=0, max_value=100, step=5)


MODEL_NAMES_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAMES_OPENAI = ["gpt-4o-mini"]

provider=st.radio("Select Provider:", ("Groq", "OpenAI"))
if provider == "Groq":
    selected_model = st.selectbox("Select Groq Mosdel:", MODEL_NAMES_GROQ)
elif provider == "OpenAI":
    selected_model = st.selectbox("Select OpenAI Mosdel:", MODEL_NAMES_OPENAI)

allow_web_search = st.checkbox("Allow Web Search")

user_query = st.text_area("Enter your query: ", height=150, placeholder="Ask Anything")

API_URL = "http://127.0.0.1:9999/chat"

if st.button("Ask Agent!"):
    if user_query.strip():
        # Get response from backend shows here
        #Step 2. Connect with backend via URL

        payload = {
            "model_name": selected_model,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search,
            "rolestrictness": role_threshold
        }

        response = requests.post(API_URL, json=payload)

        if response.status_code == 200:
            response_data = response.json()

            if "error" in response_data:
                st.error(response_data["error"])
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data}")







#Step 2. Connect with backend via URL