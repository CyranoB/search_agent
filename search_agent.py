"""search_agent.py

Usage:
    search_agent.py 
        [--domain=domain]
        [--provider=provider]
        [--model=model]
        [--embedding_model=model]
        [--temperature=temp]
        [--copywrite]
        [--max_pages=num]
        [--max_extracts=num]
        [--use_browser]
        [--output=text]
        [--verbose]
        SEARCH_QUERY
    search_agent.py --version

Options:
    -h --help                           Show this screen.
    --version                           Show version.
    -c --copywrite                      First produce a draft, review it and rewrite for a final text
    -d domain --domain=domain           Limit search to a specific domain
    -t temp --temperature=temp          Set the temperature of the LLM [default: 0.0]
    -m model --model=model              Use a specific model [default: openai/gpt-4o-mini]
    -e model --embedding_model=model    Use an embedding model
    -n num --max_pages=num              Max number of pages to retrieve [default: 10]
    -x num --max_extracts=num           Max number of page extract to consider [default: 7]
    -b --use_browser                    Use browser to fetch content from the web [default: False]
    -o text --output=text               Output format (choices: text, markdown) [default: markdown]
    -v --verbose                        Print verbose output [default: False]

"""

import os

from docopt import docopt
#from schema import Schema, Use, SchemaError
import dotenv

from langchain.callbacks import LangChainTracer

from langsmith import Client, traceable

from rich.console import Console
from rich.markdown import Markdown

import web_rag as wr
import web_crawler as wc
import copywriter as cw
import models as md
import nlp_rag as nr

console = Console()
dotenv.load_dotenv()

def get_selenium_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import WebDriverException

    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    chrome_options.add_argument("--window-size=1920,1080")

    try:
        driver = webdriver.Chrome(options=chrome_options)
        return driver
    except WebDriverException as e:
        print(f"Error creating Selenium WebDriver: {e}")
        return None

callbacks = []
if os.getenv("LANGCHAIN_API_KEY"):
    callbacks.append(
        LangChainTracer(client=Client())
    )
@traceable(run_type="tool", name="search_agent")
def main(arguments):
    verbose = arguments["--verbose"]
    copywrite_mode = arguments["--copywrite"]
    model = arguments["--model"]
    embedding_model = arguments["--embedding_model"]
    temperature = float(arguments["--temperature"])
    domain=arguments["--domain"]
    max_pages=int(arguments["--max_pages"])
    max_extract=int(arguments["--max_extracts"])
    output=arguments["--output"]
    use_selenium=arguments["--use_browser"]
    query = arguments["SEARCH_QUERY"]

    chat = md.get_model(model, temperature)
    if embedding_model is None:
        use_nlp = True
        nlp = nr.get_nlp_model()
    else:
        embedding_model = md.get_embedding_model(embedding_model)
        use_nlp = False

    if verbose:
        model_name = getattr(chat, 'model_name', None) or getattr(chat, 'model', None) or getattr(chat, 'model_id', None) or str(chat)
        console.log(f"Using embedding model: {embedding_model_name}")
        if not use_nlp:
            embedding_model_name = getattr(embedding_model, 'model_name', None) or getattr(embedding_model, 'model', None) or getattr(embedding_model, 'model_id', None) or str(embedding_model)
            console.log(f"Using model: {embedding_model_name}")
        

    with console.status(f"[bold green]Optimizing query for search: {query}"):
        optimized_search_query = wr.optimize_search_query(chat, query)
        if len(optimized_search_query) < 3:
            optimized_search_query = query
    console.log(f"Optimized search query: [bold blue]{optimized_search_query}")

    with console.status(
            f"[bold green]Searching sources using the optimized query: {optimized_search_query}"
        ):
        sources = wc.get_sources(optimized_search_query, max_pages=max_pages, domain=domain)
    console.log(f"Found {len(sources)} sources {'on ' + domain if domain else ''}")

    with console.status(
        f"[bold green]Fetching content for {len(sources)} sources", spinner="growVertical"
    ):
        contents = wc.get_links_contents(sources, get_selenium_driver, use_selenium=use_selenium)
    console.log(f"Managed to extract content from {len(contents)} sources")

    if use_nlp:
        with console.status(f"[bold green]Splitting {len(contents)} sources for content", spinner="growVertical"):            
            chunks = nr.recursive_split_documents(contents)
            #chunks = nr.chunk_contents(nlp, contents)
            console.log(f"Split {len(contents)} sources into {len(chunks)} chunks")
        with console.status(f"[bold green]Searching relevant chunks", spinner="growVertical"):
            import time

            start_time = time.time()
            relevant_results = nr.semantic_search(optimized_search_query, chunks, nlp, top_n=max_extract)
            end_time = time.time()
            execution_time = end_time - start_time
            console.log(f"Semantic search took {execution_time:.2f} seconds")
            console.log(f"Found {len(relevant_results)} relevant chunks")
        with console.status(f"[bold green]Writing content", spinner="growVertical"):
            draft = nr.query_rag(chat, query, relevant_results)
    else:
        with console.status(f"[bold green]Embedding {len(contents)} sources for content", spinner="growVertical"):
            vector_store = wc.vectorize(contents, embedding_model)
        with console.status("[bold green]Writing content", spinner='dots8Bit'):
            draft = wr.query_rag(chat, query, optimized_search_query, vector_store, top_k = max_extract)


    console.rule(f"[bold green]Response")
    if output == "text":
        console.print(draft)
    else:
        console.print(Markdown(draft))
    console.rule("[bold green]")
    
    if(copywrite_mode):
        with console.status("[bold green]Getting comments from the reviewer", spinner="dots8Bit"):
            comments = cw.generate_comments(chat, query, draft)

        console.rule("[bold green]Response from reviewer")
        if output == "text":
            console.print(comments)
        else:
            console.print(Markdown(comments))
        console.rule("[bold green]")

        with console.status("[bold green]Writing the final text", spinner="dots8Bit"):
            final_text = cw.generate_final_text(chat, query, draft, comments)

        console.rule("[bold green]Final text")
        if output == "text":
            console.print(final_text)
        else:
            console.print(Markdown(final_text))
        console.rule("[bold green]")

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Search Agent 0.1')
    main(arguments)

