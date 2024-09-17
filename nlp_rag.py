import spacy
from itertools import groupby
from operator import itemgetter
from langsmith import traceable
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

def get_nlp_model():
    if not spacy.util.is_package("en_core_web_md"):
        print("Downloading en_core_web_md model...")
        spacy.cli.download("en_core_web_md")
        print("Model downloaded successfully!")
    nlp = spacy.load("en_core_web_md")
    return nlp


def recursive_split_documents(contents, max_chunk_size=1000, overlap=100):
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


# Perform semantic search using spaCy
def semantic_search(query, chunks, nlp, similarity_threshold=0.5, top_n=10):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    
    # Precompute query vector and its norm with epsilon to prevent division by zero
    with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'tok2vec']):
        query_vector = nlp(query).vector
    query_norm = np.linalg.norm(query_vector) + 1e-8  # Add epsilon
    
    # Prepare texts from chunks
    texts = [chunk['text'] for chunk in chunks]
    
    # Function to process each text and compute its vector
    def compute_vector(text):
        with nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != 'tok2vec']):
            doc = nlp(text)
            vector = doc.vector
        return vector
    
    # Process texts in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        chunk_vectors = list(executor.map(compute_vector, texts))
    
    chunk_vectors = np.array(chunk_vectors)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1) + 1e-8  # Add epsilon
    
    # Compute similarities using vectorized operations
    similarities = np.dot(chunk_vectors, query_vector) / (chunk_norms * query_norm)
    
    # Filter and sort results
    relevant_chunks = [
        (chunk, sim) for chunk, sim in zip(chunks, similarities) if sim > similarity_threshold
    ]
    relevant_chunks.sort(key=lambda x: x[1], reverse=True)
    
    return relevant_chunks[:top_n]


@traceable(run_type="llm", name="nlp_rag")
def query_rag(chat_llm, query, relevant_results):
    import web_rag as wr

    formatted_chunks = ""
    for chunk, similarity in relevant_results:
        formatted_chunk = f"""
        <source>
        <url>{chunk['metadata']['source']}</url>
        <title>{chunk['metadata']['title']}</title>
        <text>{chunk['text']}</text>
        </source>
        """
        formatted_chunks += formatted_chunk

    prompt = wr.get_rag_prompt_template().format(query=query, context=formatted_chunks)  

    draft = chat_llm.invoke(prompt).content
    return draft