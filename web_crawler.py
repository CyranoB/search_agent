from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote

import os
import io

from trafilatura import extract
from selenium.common.exceptions import TimeoutException
from langchain_core.documents.base import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS

import requests
import pdfplumber

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
            raise Exception('Invalid API response format')

        final_results = [{
            'title': result['title'],
            'link': result['url'],
            'snippet': result['description'],
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
    #console.log(f"Processing {url}")
    response = fetch_with_timeout(url, 8)
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

def get_links_contents(sources, get_driver_func=None):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_source, sources))

    if get_driver_func is None:
        return [result for result in results if result is not None]

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

def vectorize(contents):
    documents = []
    for content in contents:
        try:
            page_content = content['page_content']
            if page_content: # Sometimes Selenium is not fetching properly
                metadata = {'title': content['title'], 'source': content['link']}
                doc = Document(page_content=content['page_content'], metadata=metadata)
                documents.append(doc)
        except Exception as e:
            print(f"[gray]Error processing content for {content['link']}: {e}")
    semantic_chunker = SemanticChunker(OpenAIEmbeddings(model="text-embedding-3-large"), breakpoint_threshold_type="percentile")
    docs = semantic_chunker.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    store = FAISS.from_documents(docs, embeddings)
    return store