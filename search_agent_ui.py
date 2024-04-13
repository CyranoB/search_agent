import dotenv

import streamlit as st

import web_rag as wr
import web_crawler as wc

from langchain_core.tracers.langchain import LangChainTracer
from langsmith.client import Client

dotenv.load_dotenv()

ls_tracer = LangChainTracer(
    project_name="Search Agent UI",
    client=Client()
)


chat = wr.get_chat_llm(provider="cohere")

st.title("🔍 Simple Search Agent 💬")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for message in st.session_state.messages:
    st.chat_message(message["role"]).write(message["content"])

if prompt := st.chat_input():
    
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    message = "I first need to do some research"
    st.chat_message("assistant").write(message)
    st.session_state.messages.append({"role": "assistant", "content": message})
    
    with st.spinner("Optimizing search query"):
        optimize_search_query = wr.optimize_search_query(chat, query=prompt, callbacks=[ls_tracer])
        
    message = f"I'll search the web for: {optimize_search_query}"
    st.chat_message("assistant").write(message)
    st.session_state.messages.append({"role": "assistant", "content": message})
    
    
    with st.spinner(f"Searching the web for: {optimize_search_query}"):
        sources = wc.get_sources(optimize_search_query, max_pages=20)
        
    with st.spinner(f"I'm now retrieveing the {len(sources)} webpages and documents I found (be patient)"):
        contents = wc.get_links_contents(sources)


    with st.spinner( f"Reading through the {len(contents)} sources I managed to retrieve"):
        vector_store = wc.vectorize(contents)

    with st.spinner( "Ok I have now enough information to answer"):
        response = wr.query_rag(chat, prompt, optimize_search_query, vector_store, callbacks=[ls_tracer])

    st.chat_message("assistant").write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    