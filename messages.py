import json
from langchain.schema import SystemMessage, HumanMessage

def get_optimized_search_messages(query):
    messages = [
        SystemMessage(
            content="""
                    You are a serach query optimizer specialist.
                    Rewrite the user's question using only the most important keywords. Remove extra words.
                    Tips:
                        Identify the key concepts in the question
                        Remove filler words like "how to", "what is", "I want to"
                        Removed style such as "in the style of", "engaging", "short", "long"
                        Remove lenght instruction (example: essay, article, letter, blog, post, blogpost, etc)
                        Keep it short, around 3-7 words total
                        Put the most important keywords first
                        Remove formatting instructions
                        Remove style instructions (exmaple: in the style of, engaging, short, long)
                        Remove lenght instruction (example: essay, article, letter, etc)
                    Example:
                        Question: How do I bake chocolate chip cookies from scratch?
                        Search query: chocolate chip cookies recipe from scratch
                    Example:
                        Question: I would like you to show me a time line of Marie Curie life. Show results as a markdown table
                        Search query: Marie Curie timeline
                    Example:
                        Question: I would like you to write a long article on nato vs russia. Use know geopolical frameworks.
                        Search query: geopolitics nato russia
                    Example:
                        Question: Write a engaging linkedin post about Andrew Ng
                        Search query: Andrew Ng
                    Example:
                        Question: Write a short artible about the solar system in the style of Carl Sagan
                        Search query: solar system
                    Example:
                        Question: Should I use Kubernetes? Answer in the style of Gilfoyde from the TV show Silicon Valley
                        Search query: Kubernetes decision
                    Example:
                        Question: biography of napoleon. include a table with the major events.
                        Search query: napoleon biography events
                """
        ),
        HumanMessage(
            content=f"""
                Questions: {query}
                Search query:
            """
        ),
    ]
    return messages

def get_query_with_sources_messages(query, relevant_docs):
    messages = [
        SystemMessage(
            content="""
    You are an expert research assistant.
    You are provided with a Context in JSON format and a Question.

    Use RAG to answer the Question, providing references and links to the Context material you retrieve and use in your answer:
    When generating your answer, follow these steps:
    - Retrieve the most relevant context material from your knowledge base to help answer the question
    - Cite the references you use by including the title, author, publication, and a link to each source
    - Synthesize the retrieved information into a clear, informative answer to the question
    - Format your answer in Markdown, using heading levels 2-3 as needed
    - Include a "References" section at the end with the full citations and link for each source you used


    Example of Context JSON entry:
    {
        "page_content": "This provides access to material related to ...",
        "metadata": {
            "title": "Introduction - Marie Curie: Topics in Chronicling America",
            "link": "https://guides.loc.gov/chronicling-america-marie-curie"
        }
    }

    """
        ),
        HumanMessage(
            content= f"""
        Context information is below.
        Context: 
        ---------------------
        {json.dumps(relevant_docs, indent=2, ensure_ascii=False)}
        ---------------------
        Question: {query}
        Answer:
    """
        ),
    ]
    return messages