from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import os
import io

from trafilatura import extract
from selenium.common.exceptions import TimeoutException
from langchain_core.documents.base import Document
from langchain_community.vectorstores.faiss import FAISS
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langsmith import traceable
import requests
import pdfplumber

@traceable(run_type="tool", name="get_sources")
def get_sources(query, max_pages=10, domain=None):
    """
    Fetch search results from the Brave Search API based on the given query.

    Args:
        query (str): The search query.
        max_pages (int): Maximum number of pages to retrieve.
        domain (str, optional): Limit search to a specific domain.

    Returns:
        list: A list of search results with title, link, snippet, and favicon.
    """
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

def fetch_with_selenium(url, driver, timeout=8):
    """
    Fetch the HTML content of a webpage using Selenium.

    Args:
        url (str): The URL of the webpage.
        driver: Selenium WebDriver instance.
        timeout (int): Page load timeout in seconds.

    Returns:
        str: The HTML content of the page.
    """
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
    """
    Fetch a webpage with a specified timeout.

    Args:
        url (str): The URL of the webpage.
        timeout (int): Request timeout in seconds.

    Returns:
        Response: The HTTP response object, or None if an error occurred.
    """
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response
    except requests.RequestException as error:
        return None

def process_source(source):
    """
    Process a single source to extract its content.

    Args:
        source (dict): A dictionary containing the source's link and other metadata.

    Returns:
        dict: The source with its extracted page content.
    """
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
    """
    Retrieve and process the content of multiple sources.

    Args:
        sources (list): A list of source dictionaries.
        get_driver_func (callable, optional): Function to get a Selenium WebDriver.
        use_selenium (bool): Whether to use Selenium for fetching content.

    Returns:
        list: A list of processed sources with their page content.
    """
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
    """
    Vectorize the contents using the specified embedding model.

    Args:
        contents (list): A list of content dictionaries.
        embedding_model: The embedding model to use.

    Returns:
        FAISS: A FAISS vector store containing the vectorized documents.
    """
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Split documents
    split_documents = text_splitter.split_documents(documents)

    # Create vector store
    vector_store = None
    batch_size = 250  # Slightly less than 256 to be safe

    for i in range(0, len(split_documents), batch_size):
        batch = split_documents[i:i+batch_size]

        if vector_store is None:
            vector_store = FAISS.from_documents(batch, embedding_model)
        else:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            embeddings = embedding_model.embed_documents(texts)
            vector_store.add_embeddings(
                list(zip(texts, embeddings)),
                metadatas
            )

    return vector_store
