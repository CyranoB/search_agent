"""search_agent.py

Usage:
    search_agent.py 
        [--domain=domain]
        [--provider=provider]
        [--temperature=temp]
        [--max_pages=num]
        [--output=text]
        SEARCH_QUERY
    search_agent.py --version

Options:
    -h --help                           Show this screen.
    --version                           Show version.
    -d domain --domain=domain           Limit search to a specific domain
    -t temp --temperature=temp          Set the temperature of the LLM [default: 0.0]
    -p provider --provider=provider     Use a specific LLM (choices: bedrock,openai,groq) [default: openai]
    -m num --max_pages=num              Max number of pages to retrieve [default: 10]
    -o text --output=text               Output format (choices: text, markdown) [default: markdown]

"""

import json
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from bs4 import BeautifulSoup
from docopt import docopt
import dotenv

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks import LangChainTracer
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.chat_models.bedrock import BedrockChat
from langsmith import Client

import requests

from rich.console import Console
from rich.rule import Rule
from rich.markdown import Markdown


def get_chat_llm(provider, temperature=0.0):
    console.log(f"Using provider {provider} with temperature {temperature}")
    match provider:
        case 'bedrock':
            chat_llm = BedrockChat(
                credentials_profile_name=os.getenv('CREDENTIALS_PROFILE_NAME'),
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                model_kwargs={"temperature": temperature },
            )
        case 'openai':
            chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=temperature)
        case 'groq':
            chat_llm = ChatGroq(model_name = 'mixtral-8x7b-32768', temperature=temperature)
        case _:
            raise ValueError(f"Unknown LLM provider {provider}")
    return chat_llm

def optimize_search_query(query):
    from messages import get_optimized_search_messages
    messages = get_optimized_search_messages(query)    
    response = chat.invoke(messages, config={"callbacks": callbacks})
    return response.content


def get_sources(query, max_pages=10, domain=None):       
    search_query = query
    if domain:
        search_query += f" site:{domain}"

    url = f"https://api.search.brave.com/res/v1/web/search?q={quote(search_query)}&count={max_pages}"
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.getenv("BRAVE_SEARCH_API_KEY")
    }

    try:
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            raise Exception(f"HTTP error! status: {response.status_code}")

        json_response = response.json()

        if 'web' not in json_response or 'results' not in json_response['web']:
            raise Exception('Invalid API response format')

        final_results = [{
            'title': result['title'],
            'link': result['url'],
            'snippet': result['description'],
            'favicon': result.get('profile', {}).get('img', '')
        } for result in json_response['web']['results']]

        return final_results

    except Exception as error:
        #console.log('Error fetching search results:', error)
        raise



def fetch_with_timeout(url, timeout=8):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        #console.log(f"Skipping {url}! Error: {error}")
        return None

def extract_main_content(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
        for element in soup(["script", "style", "head", "nav", "footer", "iframe", "img"]):
            element.extract()
        main_content = ' '.join(soup.body.get_text().split())
        return main_content
    except Exception as error:
        #console.log(f"Error extracting main content: {error}")
        return None

def process_source(source):
    response = fetch_with_timeout(source['link'], 8)
    if response:
        html = response.text
        main_content = extract_main_content(html)
        return {**source, 'html': main_content}
    return None

def get_links_contents(sources):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_source, sources))

    # Filter out None results
    return [result for result in results if result is not None]

def process_and_vectorize_content(
    contents, 
    query,
    text_chunk_size=1000,
    text_chunk_overlap=200,
    number_of_similarity_results=5
):
    """
    Process and vectorize content using Langchain.
    
    Args:
        contents (list): List of dictionaries containing 'title', 'link', and 'html' keys.
        query (str): Query string for similarity search.
        text_chunk_size (int): Size of each text chunk.
        text_chunk_overlap (int): Overlap between text chunks.
        number_of_similarity_results (int): Number of most similar results to return.
        
    Returns:
        list: List of most similar documents.
    """
    documents = []
    
    for content in contents:
        if content['html']:
            try:
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=text_chunk_size,
                    chunk_overlap=text_chunk_overlap
                )
                texts = text_splitter.split_text(content['html'])
                                
                # Create metadata for each text chunk
                metadatas = [{'title': content['title'], 'link': content['link']} for _ in range(len(texts))]
                                
                # Create vector store
                embeddings = OpenAIEmbeddings()
                docsearch = FAISS.from_texts(texts, embedding=embeddings, metadatas=metadatas)
                
                # Perform similarity search
                docs = docsearch.similarity_search(query, k=number_of_similarity_results)
                doc_dicts = [{'page_content': doc.page_content, 'metadata': doc.metadata} for doc in docs]
                documents.extend(doc_dicts)
                
            except Exception as e:
                console.log(f"[gray]Error processing content for {content['link']}: {e}")

                
    return documents


def answer_query_with_sources(query, relevant_docs):
    from messages import get_query_with_sources_messages
    messages = get_query_with_sources_messages(query, relevant_docs)
    response = chat.invoke(messages, config={"callbacks": callbacks})
    return response

console = Console()
dotenv.load_dotenv()

callbacks = []
if(os.getenv("LANGCHAIN_API_KEY")): 
    callbacks.append(
        LangChainTracer(
            project_name="search agent",
            client=Client(
            api_url="https://api.smith.langchain.com",
            )
        )
    )

if __name__ == '__main__':   
    arguments = docopt(__doc__, version='Search Agent 0.1')

    provider = arguments["--provider"]
    temperature = float(arguments["--temperature"])
    domain=arguments["--domain"] 
    max_pages=arguments["--max_pages"]
    output=arguments["--output"]
    query = arguments["SEARCH_QUERY"]
    
    chat = get_chat_llm(provider, temperature)
    
    with console.status(f"[bold green]Optimizing query for search: {query}"):
        optimize_search_query = optimize_search_query(query)
    console.log(f"Optimized search query: [bold blue]{optimize_search_query}")  
    
    with console.status(f"[bold green]Searching sources using the optimized query: {optimize_search_query}"):
        sources = get_sources(optimize_search_query, max_pages=max_pages, domain=domain)
    console.log(f"Found {len(sources)} sources {'on ' + domain if domain else ''}")

    with console.status(f"[bold green]Fetching content for {len(sources)} sources", spinner="growVertical"):
        contents = get_links_contents(sources)
    console.log(f"Managed to extract content from {len(contents)} sources")

    with console.status(
            f"[bold green]Processing {len(contents)} contents and finding relevant extracts",
            spinner="dots8Bit"
        ):
        relevant_docs = process_and_vectorize_content(contents, query)
    console.log(f"Filtered {len(relevant_docs)} relevant content extracts")

    with console.status(f"[bold green]Querying LLM with {len(relevant_docs)} relevant extracts", spinner='dots8Bit'):
        respomse = answer_query_with_sources(query, relevant_docs)

    console.rule(f"[bold green]Response from {provider}")
    if output == "text":
        console.print(respomse.content)
    else:
        console.print(Markdown(respomse.content))
    console.rule("[bold green]")
