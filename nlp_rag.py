import spacy
from itertools import groupby
from operator import itemgetter
from langsmith import traceable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def get_nlp_model():
    """
    Load and return the spaCy NLP model. Downloads the model if not already installed.
    
    Returns:
        nlp: The loaded spaCy NLP model.
    """
    if not spacy.util.is_package("en_core_web_md"):
        print("Downloading en_core_web_md model...")
        spacy.cli.download("en_core_web_md")
        print("Model downloaded successfully!")
    nlp = spacy.load("en_core_web_md")
    return nlp


def recursive_split_documents(contents, max_chunk_size=1000, overlap=100):
    """
    Split documents into smaller chunks using a recursive character text splitter.

    Args:
        contents (list): List of content dictionaries with 'page_content', 'title', and 'link'.
        max_chunk_size (int): Maximum size of each chunk.
        overlap (int): Overlap between chunks.

    Returns:
        list: List of chunks with text and metadata.
    """
    from langchain_core.documents.base import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter

    documents = []
    for content in contents:
        try:
            page_content = content['page_content']
            if page_content:
                metadata = {'title': content['title'], 'source': content['link']}
                doc = Document(page_content=content['page_content'], metadata=metadata)
                documents.append(doc)
        except Exception as e:
            print(f"Error processing content for {content['link']}: {e}")

    # Initialize recursive text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_chunk_size, chunk_overlap=overlap)

    # Split documents
    split_documents = text_splitter.split_documents(documents)

    # Convert split documents to the same format as recursive_split
    chunks = []
    for doc in split_documents:
        chunk = {
            'text': doc.page_content,
            'metadata': {
                'title': doc.metadata.get('title', ''),
                'source': doc.metadata.get('source', '')
            }
        }
        chunks.append(chunk)

    return chunks


def semantic_search(query, chunks, nlp, similarity_threshold=0.5, top_n=10):
    """
    Perform semantic search to find relevant chunks based on similarity to the query.

    Args:
        query (str): The search query.
        chunks (list): List of text chunks with vectors.
        nlp: The spaCy NLP model.
        similarity_threshold (float): Minimum similarity score to consider a chunk relevant.
        top_n (int): Number of top relevant chunks to return.

    Returns:
        list: List of relevant chunks and their similarity scores.
    """
    # Precompute query vector and its norm
    query_vector = nlp(query).vector
    query_norm = np.linalg.norm(query_vector) + 1e-8  # Add epsilon to avoid division by zero

    # Check if chunks have precomputed vectors; if not, compute them
    if 'vector' not in chunks[0]:
        texts = [chunk['text'] for chunk in chunks]

        # Process texts in batches using nlp.pipe()
        batch_size = 1000  # Adjust based on available memory
        with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'tok2vec']):
            docs = nlp.pipe(texts, batch_size=batch_size)

        # Add vectors to chunks
        for chunk, doc in zip(chunks, docs):
            chunk['vector'] = doc.vector

    # Prepare chunk vectors and norms
    chunk_vectors = np.array([chunk['vector'] for chunk in chunks])
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1) + 1e-8  # Add epsilon to avoid division by zero

    # Compute similarities
    similarities = np.dot(chunk_vectors, query_vector) / (chunk_norms * query_norm)

    # Filter and sort results
    relevant_chunks = [
        (chunk, sim) for chunk, sim in zip(chunks, similarities) if sim > similarity_threshold
    ]
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)

    return relevant_chunks[:top_n]


@traceable(run_type="llm", name="nlp_rag")
def query_rag(chat_llm, query, relevant_results, callbacks = []):
    """
    Generate a response using retrieval-augmented generation (RAG) based on relevant results.

    Args:
        chat_llm: The chat language model to use.
        query (str): The user's query.
        relevant_results (list): List of relevant chunks and their similarity scores.

    Returns:
        str: The generated response.
    """
    prompt = build_rag_prompt(query, relevant_results)
    response = chat_llm.invoke(prompt).content
    return response


def build_rag_prompt(query, relevant_results):
    import web_rag as wr
    formatted_chunks = format_docs(relevant_results)
    prompt = wr.get_rag_prompt_template().format(query=query, context=formatted_chunks)  
    return prompt

def format_docs(relevant_results):
    """
    Convert relevant search results into a JSON-formatted string.

    Args:
        relevant_results (list): List of relevant chunks with metadata.

    Returns:
        str: JSON-formatted string of document chunks.
    """
    import json

    formatted_chunks = []
    for chunk, _ in relevant_results:  # Unpack the tuple, ignore similarity score
        formatted_chunk = {
            "content": chunk['text'],
            "link": chunk['metadata'].get('source', ''),
            "title": chunk['metadata'].get('title', ''),
        }
        formatted_chunks.append(formatted_chunk)

    return json.dumps(formatted_chunks, indent=2)