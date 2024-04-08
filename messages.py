"""
This module provides functions for generating optimized search messages, RAG prompt templates,
and messages for queries with relevant source documents using the LangChain library.
"""

from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.prompts.prompt import PromptTemplate

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
            I want you to act as a prompt optimizer for web search. I will provide you with a chat prompt, and your goal is to optimize it into a search string that will yield the most relevant and useful information from a search engine like Google.
            To optimize the prompt:
            Identify the key information being requested
            Arrange the keywords into a concise search string
            Keep it short, around 1 to 5 words total
            Put the most important keywords first
            
            Some tips and things to be sure to remove:
            - Remove any conversational or instructional phrases
            - Removed style such as "in the style of", "engaging", "short", "long"
            - Remove lenght instruction (example: essay, article, letter, blog, post, blogpost, etc)
            - Remove style instructions (exmaple: "in the style of", engaging, short, long)
            - Remove lenght instruction (example: essay, article, letter, etc)
            
            Add "**" to the end of the search string to indicate the end of the query
            
            Example:
                Question: How do I bake chocolate chip cookies from scratch?
                Search query: chocolate chip cookies recipe from scratch**
            Example:
                Question: I would like you to show me a timeline of Marie Curie's life. Show results as a markdown table
                Search query: Marie Curie timeline**
            Example:
                Question: I would like you to write a long article on NATO vs Russia. Use known geopolitical frameworks.
                Search query: geopolitics nato russia**
            Example:
                Question: Write an engaging LinkedIn post about Andrew Ng
                Search query: Andrew Ng**
            Example:
                Question: Write a short article about the solar system in the style of Carl Sagan
                Search query: solar system**
            Example:
                Question: Should I use Kubernetes? Answer in the style of Gilfoyle from the TV show Silicon Valley
                Search query: Kubernetes decision**
            Example:
                Question: Biography of Napoleon. Include a table with the major events.
                Search query: napoleon biography events**
            Example:
                Question: Write a short article on the history of the United States. Include a table with the major events.
                Search query: united states history events**
            Example:
                Question: Write a short article about the solar system in the style of donald trump
                Search query: solar system**
        """
    )
    human_message = HumanMessage(
        content=f"""                 
            Question: {query}
            Search query: 
        """
    )
    return [system_message, human_message]

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
