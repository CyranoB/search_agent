"""search_agent.py

Usage:
    search_agent.py 
        [--domain=domain]
        [--provider=provider]
        [--model=model]
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
    -p provider --provider=provider     Use a specific LLM (choices: bedrock,openai,groq,ollama) [default: openai]
    -m model --model=model              Use a specific model
    -n num --max_pages=num              Max number of pages to retrieve [default: 10]
    -o text --output=text               Output format (choices: text, markdown) [default: markdown]

"""

import json
import os
import io
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

from bs4 import BeautifulSoup
from docopt import docopt
import dotenv
import pdfplumber

from langchain_core.documents.base import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks import LangChainTracer
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models.bedrock import BedrockChat
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.vectorstores.faiss import FAISS

from langsmith import Client

import requests

from rich.console import Console
from rich.markdown import Markdown

from messages import get_rag_prompt_template, get_optimized_search_messages


def get_chat_llm(provider, model=None, temperature=0.0):
    match provider:
        case 'bedrock':
            if model is None:
                model = "anthropic.claude-3-sonnet-20240229-v1:0"
            chat_llm = BedrockChat(
                credentials_profile_name=os.getenv('CREDENTIALS_PROFILE_NAME'),
                model_id=model,
                model_kwargs={"temperature": temperature },
            )
        case 'openai':
            if model is None:
                model = "gpt-3.5-turbo"
            chat_llm = ChatOpenAI(model_name=model, temperature=temperature)
        case 'groq':
            if model is None:
                model = 'mixtral-8x7b-32768'
            chat_llm = ChatGroq(model_name=model, temperature=temperature)
        case 'ollama':
            if model is None:
                model = 'llama2'
            chat_llm = ChatOllama(model=model, temperature=temperature)
        case _:
            raise ValueError(f"Unknown LLM provider {provider}")

    console.log(f"Using {model} on {provider} with temperature {temperature}")
    return chat_llm

def optimize_search_query(chat_llm, query):
    messages = get_optimized_search_messages(query)
    response = chat_llm.invoke(messages, config={"callbacks": callbacks})
    optimized_search_query = response.content
    return optimized_search_query.strip('"').split("**", 1)[0]


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
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            return []

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
        main_content = soup.get_text(separator='\n', strip=True)
        return main_content
    except Exception:
        return None

def process_source(source):
    response = fetch_with_timeout(source['link'], 8)
    console.log(f"Processing {source['link']}")
    if response:
        content_type = response.headers.get('Content-Type')
        if content_type == 'application/pdf':
            # The response is a PDF file
            pdf_content = response.content
            # Create a file-like object from the bytes
            pdf_file = io.BytesIO(pdf_content)
            # Extract text from PDF using pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
            return {**source, 'pdf_content': text}
        elif content_type.startswith('text/html'):
            # The response is an HTML file
            html = response.text
            main_content = extract_main_content(html)
            return {**source, 'html': main_content}
        else:
            console.log(f"Skipping {source['link']}! Unsupported content type: {content_type}")
            return None
    return None

def get_links_contents(sources):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_source, sources))

    # Filter out None results
    return [result for result in results if result is not None]

def vectorize(contents, text_chunk_size=1000,text_chunk_overlap=200,):
    documents = []
    for content in contents:
        page_content = content['snippet']
        if 'htlm' in content:
            page_content = content['html']
        if 'pdf_content' in content:
            page_content = content['pdf_content']        
        try:
            metadata = {'title': content['title'], 'source': content['link']}
            doc = Document(page_content=page_content, metadata=metadata)
            documents.append(doc)
        except Exception as e:
            console.log(f"[gray]Error processing content for {content['link']}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=text_chunk_size,
        chunk_overlap=text_chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)
    return store

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


def query_rag(chat_llm, question, search_query, vectorstore):
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




console = Console()
dotenv.load_dotenv()

callbacks = []
if os.getenv("LANGCHAIN_API_KEY"):
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
    model = arguments["--model"]
    temperature = float(arguments["--temperature"])
    domain=arguments["--domain"]
    max_pages=arguments["--max_pages"]
    output=arguments["--output"]
    query = arguments["SEARCH_QUERY"]

    chat = get_chat_llm(provider, model, temperature)

    with console.status(f"[bold green]Optimizing query for search: {query}"):
        optimize_search_query = optimize_search_query(chat, query)
    console.log(f"Optimized search query: [bold blue]{optimize_search_query}")

    with console.status(
            f"[bold green]Searching sources using the optimized query: {optimize_search_query}"
        ):
        sources = get_sources(optimize_search_query, max_pages=max_pages, domain=domain)
    console.log(f"Found {len(sources)} sources {'on ' + domain if domain else ''}")

    with console.status(
        f"[bold green]Fetching content for {len(sources)} sources", spinner="growVertical"
    ):
        contents = get_links_contents(sources)
    console.log(f"Managed to extract content from {len(contents)} sources")

    with console.status(f"[bold green]Embeddubg {len(contents)} sources for content", spinner="growVertical"):
        vector_store = vectorize(contents)

    with console.status("[bold green]Querying LLM relevant context", spinner='dots8Bit'):
        respomse = query_rag(chat, query, optimize_search_query, vector_store)

    console.rule(f"[bold green]Response from {provider}")
    if output == "text":
        console.print(respomse)
    else:
        console.print(Markdown(respomse))
    console.rule("[bold green]")
