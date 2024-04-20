"""
Module for performing retrieval-augmented generation (RAG) using LangChain.
This module provides functions to optimize search queries, retrieve relevant documents,
and generate answers to questions using the retrieved context. It leverages the LangChain
library for building the RAG pipeline.
Functions:
- get_optimized_search_messages(query: str) -> list:
Generate optimized search messages for a given query.
- optimize_search_query(chat_llm, query: str, callbacks: list = []) -> str:
Optimize the search query using the chat language model.
- get_rag_prompt_template() -> ChatPromptTemplate:
Get the prompt template for retrieval-augmented generation (RAG).
- format_docs(docs: list) -> str:
Format the retrieved documents into a JSON string.
- multi_query_rag(chat_llm, question: str, search_query: str, vectorstore, callbacks: list = []) -> str:
Perform RAG using multiple queries to retrieve relevant documents.
- query_rag(chat_llm, question: str, search_query: str, vectorstore, callbacks: list = []) -> str:
Perform RAG using a single query to retrieve relevant documents.
"""
import os
import json
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_aws import ChatBedrock
from langchain_cohere.chat_models import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_fireworks.chat_models import ChatFireworks
#from langchain_groq import ChatGroq
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.chat_models.ollama import ChatOllama

def get_models(provider, model=None, temperature=0.0):
    match provider:
        case 'bedrock':
            credentials_profile_name=os.getenv('CREDENTIALS_PROFILE_NAME')
            if model is None:
                model = "anthropic.claude-3-sonnet-20240229-v1:0"
            chat_llm = ChatBedrock(
                credentials_profile_name=credentials_profile_name,
                model_id=model,
                model_kwargs={"temperature": temperature, "max_tokens":4096 },
            )
            embedding_model = BedrockEmbeddings(
                model_id='cohere.embed-multilingual-v3',
                credentials_profile_name=credentials_profile_name
            )
            embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        case 'openai':
            if model is None:
                model = "gpt-3.5-turbo"
            chat_llm = ChatOpenAI(model_name=model, temperature=temperature)
            embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        case 'groq':
            if model is None:
                model = 'mixtral-8x7b-32768'
            chat_llm = ChatGroq(model_name=model, temperature=temperature)
            embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        case 'ollama':
            if model is None:
                model = 'llama2'
            chat_llm = ChatOllama(model=model, temperature=temperature)
            embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        case 'cohere':
            if model is None:
                model = 'command-r-plus'
            chat_llm = ChatCohere(model=model, temperature=temperature)
            #embedding_model = CohereEmbeddings(model="embed-english-light-v3.0")
            embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        case 'fireworks':
            if model is None:
                #model = 'accounts/fireworks/models/dbrx-instruct'
                model = 'accounts/fireworks/models/llama-v3-70b-instruct'
            chat_llm = ChatFireworks(model_name=model, temperature=temperature, max_tokens=8192)
            embedding_model = OpenAIEmbeddings(model='text-embedding-3-small')
        case _:
            raise ValueError(f"Unknown LLM provider {provider}")
    
    return chat_llm, embedding_model


def get_optimized_search_messages(query):
    """
    Generate optimized search messages for a given query.

    Args:
        query (str): The user's query.

    Returns:
        list: A list containing the system message and human message for optimized search.
    """
    system_message = SystemMessage(
        content="""
            I want you to act as a prompt optimizer for web search. 
            I will provide you with a chat prompt, and your goal is to optimize it into a search string that will yield the most relevant and useful information from a search engine like Google.
            To optimize the prompt:
            - Identify the key information being requested
            - Arrange the keywords into a concise search string
            - Keep it short, around 1 to 5 words total
            - Put the most important keywords first
            
            Some tips and things to be sure to remove:
            - Remove any conversational or instructional phrases
            - Removed style such as "in the style of", "engaging", "short", "long"
            - Remove lenght instruction (example: essay, article, letter, blog, post, blogpost, etc)
            - Remove style instructions (exmaple: "in the style of", engaging, short, long)
            - Remove lenght instruction (example: essay, article, letter, etc)
            
            You should answer only with the optimized search query and add "**" to the end of the search string to indicate the end of the query
            
            Example:
                Question: How do I bake chocolate chip cookies from scratch?
                chocolate chip cookies recipe from scratch**
            Example:
                Question: I would like you to show me a timeline of Marie Curie's life. Show results as a markdown table
                Marie Curie timeline**
            Example:
                Question: I would like you to write a long article on NATO vs Russia. Use known geopolitical frameworks.
                geopolitics nato russia**
            Example:
                Question: Write an engaging LinkedIn post about Andrew Ng
                Andrew Ng**
            Example:
                Question: Write a short article about the solar system in the style of Carl Sagan
                solar system**
            Example:
                Question: Should I use Kubernetes? Answer in the style of Gilfoyle from the TV show Silicon Valley
                Kubernetes decision**
            Example:
                Question: Biography of Napoleon. Include a table with the major events.
                napoleon biography events**
            Example:
                Question: Write a short article on the history of the United States. Include a table with the major events.
                united states history events**
            Example:
                Question: Write a short article about the solar system in the style of donald trump
                solar system**
            Exmaple:
                Question: Write a short linkedin about how the "freakeconomics" book previsions didn't pan out
                freakeconomics book predictions failed**
        """
    )
    human_message = HumanMessage(
        content=f"""                 
            Question: {query}
             
        """
    )
    return [system_message, human_message]


def optimize_search_query(chat_llm, query, callbacks=[]):
    messages = get_optimized_search_messages(query)
    response = chat_llm.invoke(messages, config={"callbacks": callbacks})
    optimized_search_query = response.content
    return optimized_search_query.strip('"').split("**", 1)[0]


def get_rag_prompt_template():
    """
    Get the prompt template for Retrieval-Augmented Generation (RAG).

    Returns:
        ChatPromptTemplate: The prompt template for RAG.
    """
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=[],
            template="""
                You are an expert research assistant.
                You are provided with a Context in JSON format and a Question. 
                Each JSON entry contains: content, title, link

                Use RAG to answer the Question, providing references and links to the Context material you retrieve and use in your answer:
                When generating your answer, follow these steps:
                - Retrieve the most relevant context material from your knowledge base to help answer the question
                - Cite the references you use by including the title, author, publication, and a link to each source
                - Synthesize the retrieved information into a clear, informative answer to the question
                - Format your answer in Markdown, using heading levels 2-3 as needed
                - Include a "References" section at the end with the full citations and link for each source you used
                
                If you cannot answer the question with confidence just say: "I'm not sure about the answer to be honest"
                If the provided context is not relevant to the question, just say: "The context provided is not relevant to the question"
            """
        )
    )
    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context", "query"],
            template="""
                Context: 
                ---------------------
                {context}
                ---------------------
                Question: {query}
                Answer:
            """
        )
    )
    return ChatPromptTemplate(
        input_variables=["context", "query"],
        messages=[system_prompt, human_prompt],
    )

def format_docs(docs):
    formatted_docs = []
    for d in docs:
        content = d.page_content
        title = d.metadata['title']
        source = d.metadata['source']
        doc = {"content": content, "title": title, "link": source}
        formatted_docs.append(doc)
    docs_as_json = json.dumps(formatted_docs, indent=2, ensure_ascii=False)
    return docs_as_json


def multi_query_rag(chat_llm, question, search_query, vectorstore, callbacks = []):
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=chat_llm, include_original=True,
    )
    unique_docs = retriever_from_llm.get_relevant_documents(
        query=search_query, callbacks=callbacks, verbose=True
    )
    context = format_docs(unique_docs)
    prompt = get_rag_prompt_template().format(query=question, context=context)
    response = chat_llm.invoke(prompt, config={"callbacks": callbacks})
    return response.content

def get_context_size(chat_llm):
    if isinstance(chat_llm, ChatOpenAI):
        if chat_llm.model_name.startswith("gpt-4"):
            return 128000
        else:
            return 16385
    if isinstance(chat_llm, ChatFireworks):
        return 8192
    if isinstance(chat_llm, ChatGroq):
        return 37862
    if isinstance(chat_llm, ChatOllama):
        return 8192
    if isinstance(chat_llm, ChatCohere):
        return 128000
    if isinstance(chat_llm, ChatBedrock):
        if chat_llm.model_id.startswith("anthropic.claude-3"):
            return 200000
        if chat_llm.model_id.startswith("anthropic.claude"):
            return 100000
        if chat_llm.model_id.startswith("mistral"):
            if chat_llm.model_id.startswith("mistral.mixtral-8x7b"):
                return 4096
            else:
                return 8192
    return 4096
        
    
def build_rag_prompt(chat_llm, question, search_query, vectorstore, top_k = 10, callbacks = []):
    done = False
    while not done:
        unique_docs = vectorstore.similarity_search(
            search_query, k=top_k, callbacks=callbacks, verbose=True)
        context = format_docs(unique_docs)
        prompt = get_rag_prompt_template().format(query=question, context=context)
        nbr_tokens = chat_llm.get_num_tokens(prompt)
        if  top_k <= 1 or nbr_tokens <= get_context_size(chat_llm) - 768:
            done = True
        else:
            top_k = int(top_k * 0.75)
       
    return prompt

def query_rag(chat_llm, question, search_query, vectorstore, top_k = 10, callbacks = []):
    prompt = build_rag_prompt(chat_llm, question, search_query, vectorstore, top_k=top_k, callbacks = callbacks)
    response = chat_llm.invoke(prompt, config={"callbacks": callbacks})
    return response.content