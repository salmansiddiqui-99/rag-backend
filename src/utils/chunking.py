"""
Text chunking utilities for splitting content into token-sized chunks.
"""
import tiktoken
from typing import List


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: Input text
        encoding_name: Encoding to use (default: cl100k_base for GPT-3.5/4)

    Returns:
        Number of tokens
    """
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))


def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
    encoding_name: str = "cl100k_base"
) -> List[str]:
    """
    Split text into chunks with token-based sizing and overlap.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens (default: 512)
        overlap: Token overlap between chunks (default: 50)
        encoding_name: Tiktoken encoding to use

    Returns:
        List of text chunks
    """
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)

    if len(tokens) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]

        # Decode tokens back to text
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)

        # Move start position with overlap
        start = end - overlap

    return chunks
