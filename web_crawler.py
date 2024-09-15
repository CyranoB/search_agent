from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import os
import io

from trafilatura import extract
from selenium.common.exceptions import TimeoutException
from langchain_core.documents.base import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langsmith import traceable
import requests
import pdfplumber

@traceable(run_type="tool", name="get_sources")
def get_sources(query, max_pages=10, domain=None):      
    search_query = query
    if domain:
        search_query += f" site:{domain}"

    url = f"https://api.search.brave.com/res/v1/web/search?q={quote(search_query)}&count={max_pages}"
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': os.getenv("BRAVE_SEARCH_API_KEY")
    }

    try:
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code != 200:
            return []

        json_response = response.json()

        if 'web' not in json_response or 'results' not in json_response['web']:
            print(response.text)
            raise Exception('Invalid API response format')

        final_results = [{
            'title': result['title'],
            'link': result['url'],
            'snippet': extract(result['description'], output_format='txt', include_tables=False, include_images=False, include_formatting=True),
            'favicon': result.get('profile', {}).get('img', '')
        } for result in json_response['web']['results']]

        return final_results

    except Exception as error:
        print('Error fetching search results:', error)
        raise



def fetch_with_selenium(url, driver, timeout=8,):
    try:
        driver.set_page_load_timeout(timeout)
        driver.get(url)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        html = driver.page_source
    except TimeoutException:
        print(f"Page load timed out after {timeout} seconds.")
        html = None
    finally:
        driver.quit()
    
    return html

def fetch_with_timeout(url, timeout=8):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        return None


def process_source(source):
    url = source['link']
    response = fetch_with_timeout(url, 2)
    if response:
        content_type = response.headers.get('Content-Type')
        if content_type:
            if content_type.startswith('application/pdf'):
                # The response is a PDF file
                pdf_content = response.content
                # Create a file-like object from the bytes
                pdf_file = io.BytesIO(pdf_content)
                # Extract text from PDF using pdfplumber
                with pdfplumber.open(pdf_file) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text()
                return {**source, 'page_content': text}
            elif content_type.startswith('text/html'):
                # The response is an HTML file
                html = response.text
                main_content = extract(html, output_format='txt', include_links=True)
                return {**source, 'page_content': main_content}
            else:
                print(f"Skipping {url}! Unsupported content type: {content_type}")
                return {**source, 'page_content': source['snippet']}
        else:
            print(f"Skipping {url}! No content type")
            return {**source, 'page_content': source['snippet']}
    return {**source, 'page_content': None}

@traceable(run_type="tool", name="get_links_contents")
def get_links_contents(sources, get_driver_func=None, use_selenium=False):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_source, sources))

    if get_driver_func is None or not use_selenium:
        return [result for result in results if result is not None and result['page_content']]

    for result in results:
        if result['page_content'] is None:
            url = result['link']
            print(f"Fetching with selenium {url}")
            driver = get_driver_func()
            html = fetch_with_selenium(url, driver)
            main_content = extract(html, output_format='txt', include_links=True)
            if main_content:
                result['page_content'] = main_content
    return results

@traceable(run_type="embedding")
def vectorize(contents, embedding_model):
    documents = []
    total_content_length = 0
    for content in contents:
        try:
            page_content = content['page_content']
            if page_content:
                metadata = {'title': content['title'], 'source': content['link']}
                doc = Document(page_content=content['page_content'], metadata=metadata)
                documents.append(doc)
                total_content_length += len(page_content)
        except Exception as e:
            print(f"[gray]Error processing content for {content['link']}: {e}")

    # Define a threshold for when to use pre-splitting (e.g., 1 million characters)
    pre_split_threshold = 1_000_000

    if total_content_length > pre_split_threshold:
        # Use pre-splitting for large datasets
        pre_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len,
        )
        documents = pre_splitter.split_documents(documents)

    semantic_chunker = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")
    
    vector_store = None
    batch_size = 200  # Adjust this value if needed

    for i in range(0, len(documents), batch_size):
        batch = documents[i:i+batch_size]
        
        # Split each document in the batch using SemanticChunker
        chunked_docs = []
        for doc in batch:
            chunked_docs.extend(semantic_chunker.split_documents([doc]))
        
        if vector_store is None:
            vector_store = FAISS.from_documents(chunked_docs, embedding_model)
        else:
            vector_store.add_documents(chunked_docs)

    return vector_store