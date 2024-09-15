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

from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_cohere import ChatCohere
from langchain_fireworks.chat_models import ChatFireworks
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langsmith import traceable


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
            You are a prompt optimizer for web search. Your task is to take a given chat prompt or question and transform it into an optimized search string that will yield the most relevant and useful information from a search engine like Google.
            The goal is to create a search query that will help users find the most accurate and pertinent information related to their original prompt or question. An effective search string should be concise, use relevant keywords, and leverage search engine syntax for better results.
            
            To optimize the prompt:
            - Identify the key information being requested
            - Consider any implicit information or context that might be useful for the search.
            - Arrange the keywords into a concise search string
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
                "Marie Curie" timeline**
            Example:
                Question: I would like you to write a long article on NATO vs Russia. Use known geopolitical frameworks.
                geopolitics nato russia**
            Example:
                Question: Write an engaging LinkedIn post about Andrew Ng
                "Andrew Ng"**
            Example:
                Question: Write a short article about the solar system in the style of Carl Sagan
                solar system**
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



def get_optimized_search_messages2(query):
    """
    Generate optimized search messages for a given query.

    Args:
        query (str): The user's query.

    Returns:
        list: A list containing the system message and human message for optimized search.
    """
    system_message = SystemMessage(
        content="""
            You are a prompt optimizer for web search. Your task is to take a given chat prompt or question and transform it into an optimized search string that will yield the most relevant and useful information from a search engine like Google.

            The goal is to create a search query that will help users find the most accurate and pertinent information related to their original prompt or question. An effective search string should be concise, use relevant keywords, and leverage search engine syntax for better results.

            Here are some key principles for creating effective search queries:
            1. Use specific and relevant keywords
            2. Remove unnecessary words (articles, prepositions, etc.)
            3. Utilize quotation marks for exact phrases
            4. Employ Boolean operators (AND, OR, NOT) when appropriate
            5. Include synonyms or related terms to broaden the search

            I will provide you with a chat prompt or question. Your task is to optimize this into an effective search string.

            Process the input as follows:
            1. Analyze the Question to identify the main topic and key concepts.
            2. Extract the most relevant keywords and phrases.
            3. Consider any implicit information or context that might be useful for the search.

            Then, optimize the search string by:
            1. Removing filler words and unnecessary language
            2. Rearranging keywords in a logical order
            3. Adding quotation marks around exact phrases if applicable
            4. Including relevant synonyms or related terms (in parentheses) to broaden the search
            5. Using Boolean operators if needed to refine the search
            
            You should answer only with the optimized search query and add "**" to the end of the search string to indicate the end of the optimized search query
        """
    )
    human_message = HumanMessage(
        content=f"""                 
            Question: {query}
             
        """
    )
    return [system_message, human_message]


@traceable(run_type="llm", name="optimize_search_query")
def optimize_search_query(chat_llm, query, callbacks=[]):
    messages = get_optimized_search_messages(query)
    response = chat_llm.invoke(messages)
    optimized_search_query = response.content.strip()
    
    # Split by '**' and take the first part, then strip whitespace
    optimized_search_query = optimized_search_query.split("**", 1)[0].strip()
    
    # Remove surrounding quotes if present
    optimized_search_query = optimized_search_query.strip('"')
    
    # If the result is empty, fall back to the original query
    if not optimized_search_query:
        optimized_search_query = query
    
    return optimized_search_query

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
                
                If the provided context is not relevant to the question, say it and answer with your internal knowledge.
                If you cannot answer the question using either the extracts or your internal knowledge, state that you don't have enough information to provide an accurate answer.
                If the information in the provided context is in contradiction with your internal knowledge, answer but warn the user about the contradiction.
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
        32768
    if isinstance(chat_llm, ChatGroq):
        return 32768
    if isinstance(chat_llm, ChatOllama):
        return 120000
    if isinstance(chat_llm, ChatCohere):
        return 120000
    if isinstance(chat_llm, ChatBedrockConverse):
        if chat_llm.model_id.startswith("meta.llama3-1"):
            return 128000
        if chat_llm.model_id.startswith("anthropic.claude-3"):
            return 200000
        if chat_llm.model_id.startswith("anthropic.claude"):
            return 100000
        if chat_llm.model_id.startswith("mistral"):
            if chat_llm.model_id.startswith("mistral.mistral.mistral-large-2407"):
                return 128000
            return 32000
    return 4096
        
@traceable(run_type="retriever")    
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

@traceable(run_type="llm", name="query_rag")
def query_rag(chat_llm, question, search_query, vectorstore, top_k = 10, callbacks = []):
    prompt = build_rag_prompt(chat_llm, question, search_query, vectorstore, top_k=top_k, callbacks = callbacks)
    response = chat_llm.invoke(prompt, config={"callbacks": callbacks})
    return response.content
