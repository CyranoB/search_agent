"""search_agent.py

Usage:
    search_agent.py 
        [--domain=domain]
        [--provider=provider]
        [--model=model]
        [--temperature=temp]
        [--copywrite]
        [--max_pages=num]
        [--max_extracts=num]
        [--output=text]
        SEARCH_QUERY
    search_agent.py --version

Options:
    -h --help                           Show this screen.
    --version                           Show version.
    -c --copywrite                      First produce a draft, review it and rewrite for a final text
    -d domain --domain=domain           Limit search to a specific domain
    -t temp --temperature=temp          Set the temperature of the LLM [default: 0.0]
    -p provider --provider=provider     Use a specific LLM (choices: bedrock,openai,groq,ollama,cohere,fireworks) [default: openai]
    -m model --model=model              Use a specific model
    -n num --max_pages=num              Max number of pages to retrieve [default: 10]
    -e num --max_extracts=num           Max number of page extract to consider [default: 5]
    -o text --output=text               Output format (choices: text, markdown) [default: markdown]

"""

import os

from docopt import docopt
#from schema import Schema, Use, SchemaError
import dotenv

from langchain.callbacks import LangChainTracer

from langsmith import Client

from rich.console import Console
from rich.markdown import Markdown

import web_rag as wr
import web_crawler as wc
import copywriter as cw

console = Console()
dotenv.load_dotenv()

def get_selenium_driver():
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException

    chrome_options = Options()
    chrome_options.add_argument("headless")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--remote-debugging-port=9222")
    chrome_options.add_argument('--blink-settings=imagesEnabled=false')
    chrome_options.add_argument("--window-size=1920,1080")

    driver = webdriver.Chrome(options=chrome_options)
    return driver

callbacks = []
if os.getenv("LANGCHAIN_API_KEY"):
    callbacks.append(
        LangChainTracer(client=Client())
    )

if __name__ == '__main__':
    arguments = docopt(__doc__, version='Search Agent 0.1')
    
    #schema = Schema({
    #    '--max_pages': Use(int, error='--max_pages must be an integer'),
    #    '--temperature': Use(float, error='--temperature must be an float'),
    #})

    #try:
    #    arguments = schema.validate(arguments)
    #except SchemaError as e:
    #    exit(e)

    copywrite_mode = arguments["--copywrite"]
    provider = arguments["--provider"]
    model = arguments["--model"]
    temperature = float(arguments["--temperature"])
    domain=arguments["--domain"]
    max_pages=arguments["--max_pages"]
    max_extract=int(arguments["--max_extracts"])
    output=arguments["--output"]
    query = arguments["SEARCH_QUERY"]

    chat, embedding_model = wr.get_models(provider, model, temperature)

    with console.status(f"[bold green]Optimizing query for search: {query}"):
        optimize_search_query = wr.optimize_search_query(chat, query, callbacks=callbacks)
        if len(optimize_search_query) < 3:
            optimize_search_query = query
    console.log(f"Optimized search query: [bold blue]{optimize_search_query}")

    with console.status(
            f"[bold green]Searching sources using the optimized query: {optimize_search_query}"
        ):
        sources = wc.get_sources(optimize_search_query, max_pages=max_pages, domain=domain)
    console.log(f"Found {len(sources)} sources {'on ' + domain if domain else ''}")

    with console.status(
        f"[bold green]Fetching content for {len(sources)} sources", spinner="growVertical"
    ):
        contents = wc.get_links_contents(sources, get_selenium_driver)
    console.log(f"Managed to extract content from {len(contents)} sources")

    with console.status(f"[bold green]Embedding {len(contents)} sources for content", spinner="growVertical"):
        vector_store = wc.vectorize(contents, embedding_model)

    with console.status("[bold green]Writing content", spinner='dots8Bit'):
        draft = wr.query_rag(chat, query, optimize_search_query, vector_store, top_k = max_extract, callbacks=callbacks)

    console.rule(f"[bold green]Response from {provider}")
    if output == "text":
        console.print(draft)
    else:
        console.print(Markdown(draft))
    console.rule("[bold green]")
    
    if(copywrite_mode):
        with console.status("[bold green]Getting comments from the reviewer", spinner="dots8Bit"):
            comments = cw.generate_comments(chat, query, draft, callbacks=callbacks)

        console.rule("[bold green]Response from reviewer")
        if output == "text":
            console.print(comments)
        else:
            console.print(Markdown(comments))
        console.rule("[bold green]")

        with console.status("[bold green]Writing the final text", spinner="dots8Bit"):
            final_text = cw.generate_final_text(chat, query, draft, comments, callbacks=callbacks)

        console.rule("[bold green]Final text")
        if output == "text":
            console.print(final_text)
        else:
            console.print(Markdown(final_text))
        console.rule("[bold green]")
