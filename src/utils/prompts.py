"""
System prompt templates for AI generation with grounding.
"""

RAG_SYSTEM_PROMPT = """You are an expert assistant for the "Physical AI & Humanoid Robotics" book.

Your role is to answer questions based ONLY on the provided context from the book.

Guidelines:
1. Use ONLY information from the provided context chunks
2. If the context doesn't contain the answer, respond: "I couldn't find information about that in the book."
3. Cite specific sections when possible (e.g., "According to Chapter X...")
4. Do not invent or assume information not present in the context
5. Synthesize information from multiple chunks when relevant
6. Provide clear, concise, technical answers

Context from book:
{context}

Previous conversation:
{conversation_history}

User question: {query}

Response:"""


SELECTED_TEXT_SYSTEM_PROMPT = """You are an expert assistant for the "Physical AI & Humanoid Robotics" book.

Your role is to answer the user's question based ONLY on the selected text they provided.

Guidelines:
1. Use ONLY information from the selected text below
2. Do not reference external knowledge or other parts of the book
3. If the selected text doesn't contain enough information, say so
4. Provide clear, helpful explanations based on the selection
5. For "explain this" or "summarize" requests, rephrase the content clearly

Selected text:
{selected_text}

Previous conversation:
{conversation_history}

User question: {query}

Response:"""


def format_rag_prompt(query: str, context_chunks: list[str], conversation_history: str = "") -> str:
    """
    Format RAG mode system prompt with context and query.

    Args:
        query: User's question
        context_chunks: Retrieved text chunks from vector search
        conversation_history: Formatted conversation history

    Returns:
        Formatted prompt string
    """
    context = "\n\n---\n\n".join(context_chunks)
    return RAG_SYSTEM_PROMPT.format(
        context=context,
        conversation_history=conversation_history or "(No previous conversation)",
        query=query
    )


def format_selected_text_prompt(query: str, selected_text: str, conversation_history: str = "") -> str:
    """
    Format selected-text mode system prompt.

    Args:
        query: User's question
        selected_text: User-selected text
        conversation_history: Formatted conversation history

    Returns:
        Formatted prompt string
    """
    return SELECTED_TEXT_SYSTEM_PROMPT.format(
        selected_text=selected_text,
        conversation_history=conversation_history or "(No previous conversation)",
        query=query
    )


def format_conversation_history(messages: list[dict]) -> str:
    """
    Format conversation history for inclusion in prompts.

    Args:
        messages: List of conversation messages with 'role' and 'content'

    Returns:
        Formatted conversation history string
    """
    if not messages:
        return ""

    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        formatted.append(f"{role}: {content}")

    return "\n".join(formatted)
