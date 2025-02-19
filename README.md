---
title: Search Agent
emoji: 🔍
colorFrom: gray
colorTo: yellow
python_version: 3.11
sdk: streamlit
sdk_version: 1.38.0
app_file: search_agent_ui.py
pinned: false
license: apache-2.0
---

⚠️ **This project is a demonstration / proof-of-concept and is not intended for use in production environments. It is provided as-is, without warranty or guarantee of any kind. The code and any accompanying materials are for educational, testing, or evaluation purposes only.** ⚠️

# Simple Search Agent

This Python project provides a search agent that can perform web searches, optimize search queries, fetch and process web content, and generate responses using a language model and the retrieved information. It does a bit of what [Perplexity AI](https://www.perplexity.ai/) does.

The Streamlit GUI hosted on 🤗 Spaces is [available to test](https://huggingface.co/spaces/CyranoB/search_agent)

This Python script and Streamlit GUI are a basic search agent that utilizes the LangChain library to perform optimized web searches, retrieve relevant content, and generate informative answers to user queries. The script supports multiple language models and providers, including OpenAI, Anthropic, and Groq.

The main functionality of the script can be summarized as follows:

1. **Query Optimization**: The user's input query is optimized for web search by identifying the key information requested and transforming it into a concise search string using the language model's capabilities.
2. **Web Search**: The optimized search query is used to fetch search results from the Brave Search API. The script allows limiting the search to a specific domain and setting the maximum number of pages to retrieve.
3. **Content Extraction**: The script fetches the content of the retrieved search results, handling both HTML and PDF documents. It uses the `trafilatura` library to extract the main text content from web pages and `pdfplumber` to extract text from PDF files. For complex web pages, Selenium can be optionally used to ensure accurate content retrieval. The extracted content is then prepared for further processing.
4. **Vectorization**: The extracted content is split into smaller text chunks using a RecursiveCharacterTextSplitter. These chunks are then vectorized using the specified embedding model if provided. If no embedding model is specified, spaCy NLP is used for semantic search. The vectorized data is stored in a FAISS vector store for efficient retrieval.
5. **Query Answering**: The user's original query is answered using a Retrieval-Augmented Generation (RAG) approach. This involves retrieving the most relevant text chunks from the vector store. The language model then generates an informative answer by synthesizing the retrieved information, citing the sources used, and formatting the response in Markdown.

The script supports various options for customization, such as specifying the language model provider (OpenAI, Anthropic, Groq, or Ollama), temperature for language model generation, and output format (text or Markdown).

Additionally, the script integrates with the LangChain Tracing V2 feature, allowing users to monitor and analyze the execution of their LangChain applications using the LangChain Studio.

To run the script, users need to provide their API keys for the desired language model provider and the Brave Search API in a `.env` file. The script can be executed from the command line, passing the desired options and the search query as arguments.

## Features

- Supports multiple language model providers (HuggingFace, Bedrock, OpenAI, Groq, Cohere, and Ollama)
- Optimizes search queries using a language model
- Fetches web pages and extracts main content (HTML and PDF)
- Vectorizes the content for efficient retrieval
- Queries the vectorized content using a Retrieval-Augmented Generation (RAG) approach
- Generates markdown-formatted responses with references to the used sources

## Setup and Installation

1. Clone this repo
2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up API keys:

   - create a `.env` file and add your API keys. Use `dotenv.sample` to create this file.
   - Get an API key from the following sources: https://brave.com/search/api/
   - Optionally you can add API keys from other LLM providers.

## Usage

You can run the search agent from the command line using the following syntax:

```bash
python search_agent.py [OPTIONS] SEARCH_QUERY
```

### Options:
| Option | Description |
|--------|-------------|
| `-h --help` | Show this screen |
| `--version` | Show version |
| `-c --copywrite` | First produce a draft, review it and rewrite for a final text |
| `-d domain --domain=domain` | Limit search to a specific domain |
| `-t temp --temperature=temp` | Set the temperature of the LLM [default: 0.0] |
| `-m model --model=model` | Use a specific model [default: hf:Qwen/Qwen2.5-72B-Instruct] |
| `-e model --embedding_model=model` | Use an embedding model |
| `-n num --max_pages=num` | Max number of pages to retrieve [default: 10] |
| `-x num --max_extracts=num` | Max number of page extract to consider [default: 7] |
| `-b --use_browser` | Use browser to fetch content from the web [default: False] |
| `-o text --output=text` | Output format (choices: text, markdown) [default: markdown] |
| `-v --verbose` | Print verbose output [default: False] |

The model can be a language model provider and a model name separated by a colon. e.g. `openai:gpt-4o-mini`
If a embedding model is not specified, spaCy will be used for semantic search.


### Examples

```bash
python search_agent.py 'What is the radioactive anomaly in the Pacific Ocean?'
```

```bash
python search_agent.py "Write a linked post about the current state of M&A for startups. Write in the style of Russ from Silicon Valley TV show." -m openai:gpt-4o-mini
```

```bash
 python search_agent.py "Write a linked post about the state of M&A for startups in 2025. Write in the style of Russ from TV show Silicon Valley" -b -m groq:llama-3.1-8b-instant -e cohere:embed-multilingual-v3.0 -t 0.7 -n 20 -x 15 
```

```bash
 python search_agent.py "Write an engaging long linked post about the state of M&A for startups in 2025" -m bedrock:claude-3-5-haiku-20241022-v1:0 -e bedrock:amazon.titan-embed-text-v2:0
```

## License

This project is licensed under the Apache License Version 2.0. See the `LICENSE` file for details.

Let me know if you have any other questions! The key components are using a web search API to find relevant information, extracting the key snippets from the search results, passing that as context to a large language model, and having the LLM generate a natural language answer based on the web search context.

## Project Structure

The project consists of several key components:

- `search_agent.py`: The main script that handles the core search agent functionality
- `search_agent_ui.py`: Streamlit-based user interface for the search agent
- `web_crawler.py`: Handles web content fetching and processing
- `web_rag.py`: Implements the Retrieval-Augmented Generation (RAG) functionality
- `nlp_rag.py`: Natural language processing utilities for RAG
- `models.py`: Contains model definitions and configurations
- `copywriter.py`: Implements content rewriting and optimization features

## Additional Tools

The project includes several development and configuration files:

- `requirements.txt`: Lists all Python dependencies
- `.env`: Configuration file for API keys and settings (use `dotenv.sample` as a template)
- `.gitignore`: Specifies which files Git should ignore
- `LICENSE`: Apache License Version 2.0
- `.devcontainer/`: Contains development container configuration for consistent development environments