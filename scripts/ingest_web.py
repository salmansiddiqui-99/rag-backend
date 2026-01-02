"""
Web content ingestion script for indexing content from sitemap URLs.

Fetches HTML pages, extracts text, chunks content, generates embeddings, and stores in Qdrant + Neon.
"""
import sys
import os
import hashlib
import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Any
import uuid
import requests
from bs4 import BeautifulSoup
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import config
from src.utils.chunking import chunk_text
from src.services.embedding_service import embedding_service
from src.services.vector_service import vector_service
from src.services.db_service import db_service


def fetch_sitemap_urls(sitemap_url: str) -> List[str]:
    """
    Fetch and parse sitemap XML to extract all URLs.

    Args:
        sitemap_url: URL to sitemap.xml file

    Returns:
        List of URLs from sitemap
    """
    print(f"Fetching sitemap from: {sitemap_url}")
    response = requests.get(sitemap_url, timeout=30)
    response.raise_for_status()

    # Parse XML
    root = ET.fromstring(response.content)

    # Handle XML namespace
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    # Extract all <loc> elements
    urls = []
    for url_element in root.findall('.//ns:loc', namespace):
        urls.append(url_element.text)

    # If no namespace found, try without namespace
    if not urls:
        for url_element in root.findall('.//loc'):
            urls.append(url_element.text)

    print(f"Found {len(urls)} URLs in sitemap")
    return urls


def extract_main_content(html: str, url: str) -> Dict[str, str]:
    """
    Extract main content from HTML page.

    Args:
        html: Raw HTML content
        url: Page URL for title extraction

    Returns:
        Dictionary with 'title' and 'text' keys
    """
    soup = BeautifulSoup(html, 'html.parser')

    # Remove script, style, nav, footer, header elements
    for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'iframe']):
        element.decompose()

    # Try to extract title
    title = None

    # First try page title from <title> tag
    title_tag = soup.find('title')
    if title_tag:
        title = title_tag.get_text().strip()
        # Clean up common patterns like "Page Title | Site Name"
        if '|' in title:
            title = title.split('|')[0].strip()

    # Try to find main content area (common Docusaurus selectors)
    main_content = None

    # Docusaurus specific selectors
    selectors = [
        'article',
        'main',
        '.markdown',
        '[class*="docPage"]',
        '[class*="docItemContainer"]',
        '.main-wrapper'
    ]

    for selector in selectors:
        main_content = soup.select_one(selector)
        if main_content:
            break

    # Fallback to body if no main content found
    if not main_content:
        main_content = soup.find('body')

    if not main_content:
        return {'title': title or 'Unknown', 'text': ''}

    # Try to extract H1 for better title
    h1 = main_content.find('h1')
    if h1 and not title:
        title = h1.get_text().strip()

    # Extract text content
    text = main_content.get_text(separator='\n', strip=True)

    # Clean up excessive whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = text.strip()

    # Use URL path as fallback title
    if not title:
        path = url.rstrip('/').split('/')[-1]
        title = path.replace('-', ' ').replace('_', ' ').title()

    return {
        'title': title,
        'text': text
    }


def fetch_and_process_page(url: str) -> List[Dict[str, Any]]:
    """
    Fetch a web page and process it into chunks with embeddings.

    Args:
        url: Page URL to fetch

    Returns:
        List of chunk dictionaries ready for Qdrant upsert
    """
    print(f"  Processing: {url}")

    try:
        # Fetch page
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Extract content
        content = extract_main_content(response.text, url)
        title = content['title']
        text = content['text']

        if not text or len(text) < 50:
            print(f"    WARNING: Minimal content found, skipping")
            return []

        # Chunk text
        chunks = chunk_text(
            text,
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP
        )

        print(f"    Created {len(chunks)} chunks from {len(text)} characters")

        # Generate embeddings
        print(f"    Generating embeddings...")
        embeddings = embedding_service.embed_batch(chunks)

        # Create chunk objects
        chunk_objects = []
        for idx, (chunk_text_str, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_obj = {
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text_str,
                "embedding": embedding,
                "chapter_title": title,
                "section_heading": None,
                "chunk_index": idx,
                "source_file_path": url
            }
            chunk_objects.append(chunk_obj)

        return chunk_objects

    except requests.RequestException as e:
        print(f"    ERROR fetching URL: {e}")
        return []
    except Exception as e:
        print(f"    ERROR processing page: {e}")
        return []


def ingest_from_sitemap(sitemap_url: str) -> Dict[str, Any]:
    """
    Ingest all pages from a sitemap URL.

    Args:
        sitemap_url: URL to sitemap.xml file

    Returns:
        Dictionary with ingestion statistics
    """
    start_time = time.time()

    # Fetch URLs from sitemap
    urls = fetch_sitemap_urls(sitemap_url)

    if not urls:
        raise ValueError(f"No URLs found in sitemap: {sitemap_url}")

    # Filter out search and markdown-page URLs (not content pages)
    content_urls = [
        url for url in urls
        if '/search' not in url and '/markdown-page' not in url
    ]

    print(f"\nProcessing {len(content_urls)} content pages (filtered from {len(urls)} total URLs)")

    # Ensure Qdrant collection exists
    print("\nInitializing Qdrant collection...")
    vector_service.create_collection()

    # Process each URL
    all_chunks = []
    processed_urls = []

    for url in content_urls:
        try:
            chunks = fetch_and_process_page(url)

            if chunks:
                all_chunks.extend(chunks)

                # Store metadata
                content_hash = hashlib.sha256(url.encode()).hexdigest()
                db_service.upsert_document_metadata(
                    file_path=url,
                    content_hash=content_hash,
                    chunk_count=len(chunks)
                )

                processed_urls.append(url)

            # Small delay to be respectful
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR processing {url}: {e}")
            continue

    # Upsert all chunks to Qdrant
    if all_chunks:
        print(f"\nUploading {len(all_chunks)} chunks to Qdrant...")
        vector_service.upsert_chunks(all_chunks)
        print("  Upload complete!")

    duration = time.time() - start_time

    return {
        "indexed_pages": len(processed_urls),
        "total_chunks": len(all_chunks),
        "duration_seconds": round(duration, 2),
        "pages": processed_urls
    }


def main():
    """Main ingestion script entry point."""
    parser = argparse.ArgumentParser(description="Ingest web content from sitemap into vector database")
    parser.add_argument(
        "--sitemap-url",
        type=str,
        required=True,
        help="URL to sitemap.xml file"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RAG CHATBOT WEB CONTENT INGESTION")
    print("=" * 60)
    print(f"Sitemap URL: {args.sitemap_url}\n")

    try:
        result = ingest_from_sitemap(args.sitemap_url)

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Pages indexed: {result['indexed_pages']}")
        print(f"Total chunks: {result['total_chunks']}")
        print(f"Duration: {result['duration_seconds']}s")
        print(f"Average: {result['total_chunks'] / result['indexed_pages']:.1f} chunks/page")
        print("\nIndexed pages:")
        for url in result['pages']:
            print(f"  - {url}")

        return 0

    except Exception as e:
        print(f"\nERROR: Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
