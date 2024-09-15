import os
import json
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts.chat import (
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    ChatPromptTemplate
)
from langchain.prompts.prompt import PromptTemplate
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models.bedrock_converse import ChatBedrockConverse
from langchain_cohere import ChatCohere
from langchain_fireworks.chat_models import ChatFireworks
from langchain_fireworks.embeddings import FireworksEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_cohere.chat_models import ChatCohere
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_community.chat_models import ChatPerplexity
from langchain_together import ChatTogether
from langchain_together.embeddings import TogetherEmbeddings



def get_model(provider_model, temperature=0.0):
    provider, model = (provider_model.rstrip('/').split('/') + [None])[:2]
    match provider:
        case 'bedrock':
            if model is None:
                model = "anthropic.claude-3-sonnet-20240229-v1:0"
            chat_llm = ChatBedrockConverse(model=model, temperature=temperature)
        case 'cohere':
            if model is None:
                model = 'command-r-plus'
            chat_llm = ChatCohere(model=model, temperature=temperature)
        case 'fireworks':
            if model is None:
                model = 'accounts/fireworks/models/llama-v3p1-8b-instruct'
            chat_llm = ChatFireworks(model_name=model, temperature=temperature, max_tokens=120000)
        case 'googlegenerativeai':
            if model is None:
                model = "gemini-1.5-flash"
            chat_llm = ChatGoogleGenerativeAI(model=model, temperature=temperature, 
                                              max_tokens=None, timeout=None, max_retries=2,)
        case 'groq':
            if model is None:
                model = 'llama-3.1-8b-instant'
            chat_llm = ChatGroq(model_name=model, temperature=temperature)
        case 'ollama':
            if model is None:
                model = 'llama3.1'
            chat_llm = ChatOllama(model=model, temperature=temperature)
        case 'openai':
            if model is None:
                model = "gpt-4o-mini"
            chat_llm = ChatOpenAI(model_name=model, temperature=temperature)
        case 'perplexity':
            if model is None:
                model = 'llama-3.1-sonar-small-128k-online'
            chat_llm = ChatPerplexity(model=model, temperature=temperature)
        case 'together':
            if model is None:
                model = 'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo'
            chat_llm = ChatTogether(model=model, temperature=temperature)
        case _:
            raise ValueError(f"Unknown LLM provider {provider}")
    
    return chat_llm


def get_embedding_model(provider_embedding_model):
    provider, model = (provider_embedding_model.rstrip('/').split('/') + [None])[:2]
    match provider:
        case 'bedrock':
            if model is None:
                model = "amazon.titan-embed-text-v2:0"
            embedding_model = BedrockEmbeddings(model_id=model)
        case 'cohere':
            if model is None:
                model = "embed-english-light-v3.0"
            embedding_model = CohereEmbeddings(model=model)
        case 'fireworks':
            if model is None:
                model = 'nomic-ai/nomic-embed-text-v1.5'
            embedding_model = FireworksEmbeddings(model=model)
        case 'ollama':
            if model is None:
                model = 'nomic-embed-text:latest'
            embedding_model = OllamaEmbeddings(model=model)
        case 'openai':
            if model is None:
                model = "text-embedding-3-small"
            embedding_model = OpenAIEmbeddings(model=model)
        case 'googlegenerativeai':
            if model is None:
                model = "models/embedding-001"
            embedding_model = GoogleGenerativeAIEmbeddings(model=model)
        case 'groq':
            embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
        case 'perplexity':
            raise ValueError(f"Cannot use Perplexity for embedding model")
        case 'together':
            if model is None:
                model = 'togethercomputer/m2-bert-80M-2k-retrieval'
            embedding_model = TogetherEmbeddings(model=model)
        case _:
            raise ValueError(f"Unknown LLM provider {provider}")

    return embedding_model


import unittest
from unittest.mock import patch
from models import get_embedding_model  # Make sure this import is correct

class TestGetEmbeddingModel(unittest.TestCase):

    @patch('models.BedrockEmbeddings')
    def test_bedrock_embedding(self, mock_bedrock):
        result = get_embedding_model('bedrock')
        mock_bedrock.assert_called_once_with(model_id='cohere.embed-multilingual-v3')
        self.assertEqual(result, mock_bedrock.return_value)

    @patch('models.CohereEmbeddings')
    def test_cohere_embedding(self, mock_cohere):
        result = get_embedding_model('cohere')
        mock_cohere.assert_called_once_with(model='embed-english-light-v3.0')
        self.assertEqual(result, mock_cohere.return_value)

    @patch('models.FireworksEmbeddings')
    def test_fireworks_embedding(self, mock_fireworks):
        result = get_embedding_model('fireworks')
        mock_fireworks.assert_called_once_with(model='nomic-ai/nomic-embed-text-v1.5')
        self.assertEqual(result, mock_fireworks.return_value)

    @patch('models.OllamaEmbeddings')
    def test_ollama_embedding(self, mock_ollama):
        result = get_embedding_model('ollama')
        mock_ollama.assert_called_once_with(model='nomic-embed-text:latest')
        self.assertEqual(result, mock_ollama.return_value)

    @patch('models.OpenAIEmbeddings')
    def test_openai_embedding(self, mock_openai):
        result = get_embedding_model('openai')
        mock_openai.assert_called_once_with(model='text-embedding-3-small')
        self.assertEqual(result, mock_openai.return_value)

    @patch('models.GoogleGenerativeAIEmbeddings')
    def test_google_embedding(self, mock_google):
        result = get_embedding_model('googlegenerativeai')
        mock_google.assert_called_once_with(model='models/embedding-001')
        self.assertEqual(result, mock_google.return_value)

    @patch('models.TogetherEmbeddings')
    def test_together_embedding(self, mock_together):
        result = get_embedding_model('together')
        mock_together.assert_called_once_with(model='BAAI/bge-base-en-v1.5')
        self.assertEqual(result, mock_together.return_value)

    def test_invalid_provider(self):
        with self.assertRaises(ValueError):
            get_embedding_model('invalid_provider')

    def test_groq_provider(self):
        with self.assertRaises(ValueError):
            get_embedding_model('groq')

    def test_perplexity_provider(self):
        with self.assertRaises(ValueError):
            get_embedding_model('perplexity')


import unittest
from unittest.mock import patch
from models import get_model  # Make sure this import is correct

class TestGetModel(unittest.TestCase):

    @patch('models.ChatBedrockConverse')
    def test_bedrock_model(self, mock_bedrock):
        result = get_model('bedrock')
        mock_bedrock.assert_called_once_with(
            model="anthropic.claude-3-sonnet-20240229-v1:0",
            temperature=0.0
        )
        self.assertEqual(result, mock_bedrock.return_value)

    @patch('models.ChatCohere')
    def test_cohere_model(self, mock_cohere):
        result = get_model('cohere')
        mock_cohere.assert_called_once_with(model='command-r-plus', temperature=0.0)
        self.assertEqual(result, mock_cohere.return_value)

    @patch('models.ChatFireworks')
    def test_fireworks_model(self, mock_fireworks):
        result = get_model('fireworks')
        mock_fireworks.assert_called_once_with(
            model_name='accounts/fireworks/models/llama-v3p1-8b-instruct',
            temperature=0.0,
            max_tokens=120000
        )
        self.assertEqual(result, mock_fireworks.return_value)

    @patch('models.ChatGoogleGenerativeAI')
    def test_google_model(self, mock_google):
        result = get_model('googlegenerativeai')
        mock_google.assert_called_once_with(
            model="gemini-1.5-pro",
            temperature=0.0,
            max_tokens=None,
            timeout=None,
            max_retries=2
        )
        self.assertEqual(result, mock_google.return_value)

    @patch('models.ChatGroq')
    def test_groq_model(self, mock_groq):
        result = get_model('groq')
        mock_groq.assert_called_once_with(model_name='llama2-70b-4096', temperature=0.0)
        self.assertEqual(result, mock_groq.return_value)

    @patch('models.ChatOllama')
    def test_ollama_model(self, mock_ollama):
        result = get_model('ollama')
        mock_ollama.assert_called_once_with(model='llama3.1', temperature=0.0)
        self.assertEqual(result, mock_ollama.return_value)

    @patch('models.ChatOpenAI')
    def test_openai_model(self, mock_openai):
        result = get_model('openai')
        mock_openai.assert_called_once_with(model_name='gpt-4o-mini', temperature=0.0)
        self.assertEqual(result, mock_openai.return_value)

    @patch('models.ChatPerplexity')
    def test_perplexity_model(self, mock_perplexity):
        result = get_model('perplexity')
        mock_perplexity.assert_called_once_with(model='llama-3.1-sonar-small-128k-online', temperature=0.0)
        self.assertEqual(result, mock_perplexity.return_value)

    @patch('models.ChatTogether')
    def test_together_model(self, mock_together):
        result = get_model('together')
        mock_together.assert_called_once_with(model='meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo', temperature=0.0)
        self.assertEqual(result, mock_together.return_value)

    def test_invalid_provider(self):
        with self.assertRaises(ValueError):
            get_model('invalid_provider')

    def test_custom_temperature(self):
        with patch('models.ChatOpenAI') as mock_openai:
            result = get_model('openai', temperature=0.5)
            mock_openai.assert_called_once_with(model_name='gpt-4o-mini', temperature=0.5)
            self.assertEqual(result, mock_openai.return_value)

    def test_custom_model(self):
        with patch('models.ChatOpenAI') as mock_openai:
            result = get_model('openai/gpt-4')
            mock_openai.assert_called_once_with(model_name='gpt-4', temperature=0.0)
            self.assertEqual(result, mock_openai.return_value)

if __name__ == '__main__':
    unittest.main()
