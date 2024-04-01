# Simple Search Agent

This is a simple search agent that accepts a question as input, searches the web for relevant information, and then uses the search results to generate an answer using a large language model (LLM).

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
   - Add your API keys to the `config.py` file.
4. Run the agent:
   ```
   python agent.py
   ```

## Usage

To use the agent, simply run the `agent.py` script and enter your question when prompted. The agent will return a generated answer based on web search results.

Example:
```
$ python agent.py
Enter a question: What is the capital of France?

Based on the information from web searches, the capital of France is Paris. Paris is the largest city in France and has been the country's capital since the 12th century. It is located in north-central France along the Seine River. As the capital, Paris is the seat of France's national government. Key landmarks include the Eiffel Tower, the Louvre museum, and the Notre-Dame cathedral. Paris is also a major global city known for its art, fashion, cuisine and culture.
```

## Configuration

- `config.py`: Contains configuration settings for the agent, including API keys.
- `agent.py`: The main agent script that accepts user input, performs web searches, and generates answers.
- `search.py`: Handles making web search requests and extracting relevant snippets from the results.
- `llm.py`: Interfaces with the language model API to generate answer text based on the web search information.

## Dependencies 

- `requests`: For making HTTP requests to web search and LLM APIs
- `openai`: For interfacing with the OpenAI language model API
- `googlesearch-python`: For performing Google web searches

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

Let me know if you have any other questions! The key components are using a web search API to find relevant information, extracting the key snippets from the search results, passing that as context to a large language model, and having the LLM generate a natural language answer based on the web search context.