---
title: Search Agent
emoji: 🔍
colorFrom: gray
colorTo: yellow
sdk: streamlit
sdk_version: 1.33.0
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
3. **Content Extraction**: The script fetches the content of the retrieved search results, handling both HTML and PDF documents. It extracts the main text content from web pages and text from PDF files.
4. **Vectorization**: The extracted content is split into smaller text chunks using a RecursiveCharacterTextSplitter and vectorized using the specified embedding model. The vectorized data is stored in a FAISS vector store for efficient retrieval.
5. **Query Answering**: The user's original query is answered by retrieving the most relevant text chunks from the vector store. The language model generates an informative answer by synthesizing the retrieved information, citing the sources used, and formatting the response in Markdown.

The script supports various options for customization, such as specifying the language model provider (OpenAI, Anthropic, Groq, or Ollama), temperature for language model generation, and output format (text or Markdown).

Additionally, the script integrates with the LangChain Tracing V2 feature, allowing users to monitor and analyze the execution of their LangChain applications using the LangChain Studio.

To run the script, users need to provide their API keys for the desired language model provider and the Brave Search API in a `.env` file. The script can be executed from the command line, passing the desired options and the search query as arguments.

## Features

- Supports multiple language model providers (Bedrock, OpenAI, Groq, Cohere, and Ollama)
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

   - You will need API keys for the Brave Search API and LLM API.
   - Add your API keys to the `.env` file. Use `dotenv.sample` to create this file.

## Usage

You can run the search agent from the command line using the following syntax:

```bash
python search_agent.py [OPTIONS] SEARCH_QUERY
```

### Options:

- `-h`, `--help`: Show this help message and exit.
- `--version`: Show the program's version number and exit.
- `-c`, `--copywrite`: First produce a draft, review it, and rewrite for a final text.
- `-d DOMAIN`, `--domain=DOMAIN`: Limit search to a specific domain.
- `-t TEMP`, `--temperature=TEMP`: Set the temperature of the LLM [default: 0.0].
- `-m MODEL`, `--model=MODEL`: Use a specific model [default: openai/gpt-4o-mini].
- `-e MODEL`, `--embedding_model=MODEL`: Use a specific embedding model [default: same provider as model].
- `-n NUM`, `--max_pages=NUM`: Max number of pages to retrieve [default: 10].
- `-x NUM`, `--max_extracts=NUM`: Max number of page extracts to consider [default: 7].
- `-s`, `--use_selenium`: Use selenium to fetch content from the web [default: False].
- `-o TEXT`, `--output=TEXT`: Output format (choices: text, markdown) [default: markdown].

### Examples

```bash
python search_agent.py -m openai/gpt-4o-mini "Write a linked post about the current state of M&A for startups. Write in the style of Russ from Silicon Valley TV show."
```

```bash
 python search_agent.py -m openai -e ollama -t 0.7 -n 20 -x 15  "Write a linked post about the state of M&A for startups in 2024. Write in the style of Russ from TV show Silicon Valley" -s   
```

## License

This project is licensed under the Apache License Version 2.0. See the `LICENSE` file for details.

Let me know if you have any other questions! The key components are using a web search API to find relevant information, extracting the key snippets from the search results, passing that as context to a large language model, and having the LLM generate a natural language answer based on the web search context.