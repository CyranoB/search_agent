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
    -p provider --provider=provider     Use a specific LLM (choices: bedrock,openai,groq,ollama,cohere) [default: openai]
    -m model --model=model              Use a specific model
    -n num --max_pages=num              Max number of pages to retrieve [default: 10]
    -o text --output=text               Output format (choices: text, markdown) [default: markdown]

"""

import os

from docopt import docopt
import dotenv

from langchain.callbacks import LangChainTracer

from langsmith import Client

from rich.console import Console
from rich.markdown import Markdown

import web_rag as wr
import web_crawler as wc

console = Console()
dotenv.load_dotenv()

callbacks = []
if os.getenv("LANGCHAIN_API_KEY"):
    callbacks.append(
        LangChainTracer(client=Client())
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

    chat = wr.get_chat_llm(provider, model, temperature)
    console.log(f"Using {chat.get_name} on {provider} with temperature {temperature}")

    with console.status(f"[bold green]Optimizing query for search: {query}"):
        optimize_search_query = wr.optimize_search_query(chat, query, callbacks=callbacks)
    console.log(f"Optimized search query: [bold blue]{optimize_search_query}")

    with console.status(
            f"[bold green]Searching sources using the optimized query: {optimize_search_query}"
        ):
        sources = wc.get_sources(optimize_search_query, max_pages=max_pages, domain=domain)
    console.log(f"Found {len(sources)} sources {'on ' + domain if domain else ''}")

    with console.status(
        f"[bold green]Fetching content for {len(sources)} sources", spinner="growVertical"
    ):
        contents = wc.get_links_contents(sources)
    console.log(f"Managed to extract content from {len(contents)} sources")

    with console.status(f"[bold green]Embeddubg {len(contents)} sources for content", spinner="growVertical"):
        vector_store = wc.vectorize(contents)

    with console.status("[bold green]Querying LLM relevant context", spinner='dots8Bit'):
        respomse = wr.query_rag(chat, query, optimize_search_query, vector_store, callbacks=callbacks)

    console.rule(f"[bold green]Response from {provider}")
    if output == "text":
        console.print(respomse)
    else:
        console.print(Markdown(respomse))
    console.rule("[bold green]")
