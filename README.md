# Simple Search Agent

This is a simple search agent that (kind of) does what [Perplexity AI](https://www.perplexity.ai/) does.

## How It Works

1. The user asks the agent a question.
2. The agent performs a web search using the question as the query.
3. The agent extracts the most relevant snippets and information from the top search results. 
4. The extracted web results are passed as context to a large language model.
5. The LLM uses the web search context to generate a final answer to the original question.
6. The agent returns the generated answer to the user.

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