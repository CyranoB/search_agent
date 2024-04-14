import datetime
import os

import dotenv
import streamlit as st

from langchain_core.tracers.langchain import LangChainTracer
from langchain.callbacks.base import BaseCallbackHandler
from langsmith.client import Client

import web_rag as wr
import web_crawler as wc

dotenv.load_dotenv()

ls_tracer = LangChainTracer(
    project_name=os.getenv("LANGSMITH_PROJECT_NAME"),
    client=Client()
)


class StreamHandler(BaseCallbackHandler):
    """Stream handler that appends tokens to container."""
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.container.markdown(self.text)

st.title("🔍 Simple Search Agent 💬")

if "providers" not in st.session_state:
    providers = []
    if os.getenv("COHERE_API_KEY"):
        providers.append("cohere")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    if os.getenv("GROQ_API_KEY"):
        providers.append("groq")
    if os.getenv("OLLAMA_API_KEY"):
        providers.append("ollama")
    if os.getenv("FIREWORKS_API_KEY"):
        providers.append("fireworks")        
    if os.getenv("CREDENTIALS_PROFILE_NAME"):
        providers.append("bedrock")
    st.session_state["providers"] = providers

with st.sidebar:
    st.write("Options")
    model_provider = st.selectbox("🧠 Model provider 🧠", st.session_state["providers"])
    temperature = st.slider("🌡️ Model temperature 🌡️", 0.0, 1.0, 0.1, help="The higher the more creative")
    max_pages = st.slider("🔍 Max pages to retrieve 🔍", 1, 20, 15, help="How many web pages to retrive from the internet")
    top_k_documents = st.slider("📄 How many document extracts to consider 📄", 1, 20, 5, help="How many of the top extracts to consider")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
    
for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])
    if message["role"] == "assistant" and 'message_id' in message:
        st.download_button(
            label="Download",
            data=message["content"],
            file_name=f"{message['message_id']}.txt",
            mime="text/plain"
        )

if prompt := st.chat_input("Enter you instructions..." ):   
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chat, embedding_model = wr.get_models(model_provider, temperature=temperature)

    with st.status("Thinking", expanded=True):
        st.write("I first need to do some research")
        
        optimize_search_query = wr.optimize_search_query(chat, query=prompt, callbacks=[ls_tracer])
        st.write(f"I should search the web for: {optimize_search_query}")
        
        sources = wc.get_sources(optimize_search_query, max_pages=max_pages)
       
        st.write(f"I'll now retrieve the {len(sources)} webpages and documents I found")
        contents = wc.get_links_contents(sources)

        st.write( f"Reading through the {len(contents)} sources I managed to retrieve")
        vector_store = wc.vectorize(contents, embedding_model=embedding_model)
        st.write(f"I collected {vector_store.index.ntotal} chunk of data and I can now answer")

    rag_prompt = wr.build_rag_prompt(prompt, optimize_search_query, vector_store, top_k=5, callbacks=[ls_tracer])  
    with st.chat_message("assistant"):
        st_cb = StreamHandler(st.empty())
        result = chat.invoke(rag_prompt, stream=True, config={ "callbacks": [st_cb, ls_tracer]})
        response = result.content.strip()
        message_id = f"{prompt}{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        st.session_state.messages.append({"role": "assistant", "content": response})
        if st.session_state.messages[-1]["role"] == "assistant":
            st.download_button(
                label="Download",
                data=st.session_state.messages[-1]["content"],
                file_name=f"{message_id}.txt",
                mime="text/plain"
            )
