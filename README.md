# Simple Search Agent

This Python project provides a search agent that can perform web searches, optimize search queries, fetch and process web content, and generate responses using a language model and the retrieved information.
Does a bit what [Perplexity AI](https://www.perplexity.ai/) does.


This Python script is a search agent that utilizes the LangChain library to perform optimized web searches, retrieve relevant content, and generate informative answers to user queries. The script supports multiple language models and providers, including OpenAI, Anthropic, and Groq.

The main functionality of the script can be summarized as follows:

1. **Query Optimization**: The user's input query is optimized for web search by identifying the key information requested and transforming it into a concise search string using the language model's capabilities.
2. **Web Search**: The optimized search query is used to fetch search results from the Brave Search API. The script allows limiting the search to a specific domain and setting the maximum number of pages to retrieve.
3. **Content Extraction**: The script fetches the content of the retrieved search results, handling both HTML and PDF documents. It extracts the main text content from web pages and text from PDF files.
4. **Vectorization**: The extracted content is split into smaller text chunks and vectorized using OpenAI's text embeddings. The vectorized data is stored in a FAISS vector store for efficient retrieval.
5. **Query Answering**: The user's original query is answered by retrieving the most relevant text chunks from the vector store using a Multi-Query Retriever. The language model generates an informative answer by synthesizing the retrieved information, citing the sources used, and formatting the response in Markdown.

The script supports various options for customization, such as specifying the language model provider (OpenAI, Anthropic, Groq, or OllaMa), temperature for language model generation, and output format (text or Markdown).

Additionally, the script integrates with the LangChain Tracing V2 feature, allowing users to monitor and analyze the execution of their LangChain applications using the LangChain Studio.

To run the script, users need to provide their API keys for the desired language model provider and the Brave Search API in a `.env` file. The script can be executed from the command line, passing the desired options and the search query as arguments.

## Features

- Supports multiple language model providers (Bedrock, OpenAI, Groq, and Ollama)
- Optimizes search queries using a language model
- Fetches web pages and extracts main content (HTML and PDF)
- Vectorizes the content for efficient retrieval
- Queries the vectorized content using a Retrieval-Augmented Generation (RAG) approach
- Generates markdown-formatted responses with references to the used sources

## Setup and Installation

1. Clone this repo
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up API keys:
   - You will need API keys for the web search API and LLM API. 
   - Add your API keys to the `.env` file. Use `dotenv.sample` to create this file.

## Usage

```
python search_agent.py --query "your search query" --provider "provider_name" --model "model_name" --temperature 0.0
```

Replace `"your search query"` with your desired search query, `"provider_name"` with the language model provider (e.g., `bedrock`, `openai`, `groq`, `ollama`), `"model_name"` with the specific model name (optional), and `temperature` with the desired temperature value for the language model (optional).

Example:
```
➜ python ./search_agent.py  --provider groq -o text "Write a linkedin post on how Sequoia Capital AI Ascent 2024 is interesting"
[21:44:05] Using mixtral-8x7b-32768 on groq with temperature 0.0             search_agent.py:78
[21:44:06] Optimized search query: Sequoia Capital AI Ascent 2024 interest  search_agent.py:248
           Found 10 sources                                                 search_agent.py:252
[21:44:08] Managed to extract content from 7 sources                        search_agent.py:256
[21:44:12] Filtered 21 relevant content extracts                            search_agent.py:263
───────────────────────────────────── Response from groq ──────────────────────────────────────
🚀 Sequoia Capital's AI Ascent 2024 conference brought together some of the brightest minds in
AI, including founders, researchers, and industry leaders. The event was a unique opportunity
to discuss the state of AI and its future, focusing on the promise of generative AI to
revolutionize industries and provide amazing productivity gains.

🌟 Highlights of the conference included talks by Sam Altman of OpenAI, Dylan Field of Figma,
Alfred Mensch of Mistral, Daniela Amodei of Anthropic, Andrew Ng of AI Fund, CJ Desai of
ServiceNow, and independent researcher Andrej Karpathy. Sessions covered a wide range of
topics, from the merits of large and small models to the rise of reasoning agents, the future
of compute, and the evolving AI ecosystem.

💡 One key takeaway from the event is the recognition that we are in a 'primordial soup' phase
of AI development. This is a crucial moment for the technology to transition from being an idea
to solving real-world problems efficiently. Factors like cheap compute power, fast networks,
ubiquitous supercomputers, and readily available data are enabling AI as the next significant
technology wave.

🔜 As we move forward, we can expect AI to become an even more significant part of our lives,
revolutionizing various sectors and offering unprecedented value creation potential. Stay tuned
for the upcoming advancements in AI, and let's continue to explore and harness its vast
capabilities!

_For more information, check out the [Sequoia Capital AI Ascent 2024 conference
recap](https://www.sequoiacap.com/article/ai-ascent-2024/)._

#AI #ArtificialIntelligence #GenerativeAI #SequoiaCapital #AIascent2024
──────────────────────────────────────────────  ───────────────────────────────────────────────

```

## License

This project is licensed under the Apache License Version 2.0. See the `LICENSE` file for details.

Let me know if you have any other questions! The key components are using a web search API to find relevant information, extracting the key snippets from the search results, passing that as context to a large language model, and having the LLM generate a natural language answer based on the web search context.