"""
Chat endpoint for RAG queries and selected-text questions.
"""
from fastapi import APIRouter, HTTPException
from src.models.query import ChatQuery, ChatResponse, SourceChunk, QueryMode
from src.services.embedding_service import embedding_service
from src.services.vector_service import vector_service
from src.services.generation_service import generation_service
from src.utils.prompts import (
    format_rag_prompt,
    format_selected_text_prompt,
    format_conversation_history
)
from src.utils.chunking import count_tokens
from src.config import config
from src.agent import query_agent
from datetime import datetime

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(query: ChatQuery):
    """
    Process chatbot query in RAG or selected-text mode.

    **RAG Mode** (no selected_text):
    - Embeds query
    - Searches Qdrant for top-K relevant chunks
    - Generates response with citations

    **Selected-Text Mode** (selected_text provided):
    - Uses selected_text as context
    - Skips vector search
    - Generates response based only on selection

    Args:
        query: ChatQuery with user question and optional context

    Returns:
        ChatResponse with AI-generated answer and citations

    Raises:
        HTTPException 400: Invalid query (too long, empty selected_text)
        HTTPException 429: Rate limit exceeded
        HTTPException 500: Service error
    """
    try:
        # Validate query token count
        query_tokens = count_tokens(query.query)
        if query_tokens > config.MAX_QUERY_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"Query exceeds {config.MAX_QUERY_TOKENS} tokens (got {query_tokens})"
            )

        # Format conversation history
        conversation_history_str = ""
        if query.conversation_history:
            messages = [{"role": msg.role, "content": msg.content} for msg in query.conversation_history]
            conversation_history_str = format_conversation_history(messages)

        # Determine mode: RAG or selected-text
        if query.selected_text:
            # SELECTED-TEXT MODE
            return await _handle_selected_text_mode(
                query.query,
                query.selected_text,
                conversation_history_str
            )
        else:
            # RAG MODE
            return await _handle_rag_mode(
                query.query,
                conversation_history_str
            )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again in a moment."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Chat processing failed: {error_msg}"
            )


async def _handle_rag_mode(query_text: str, conversation_history: str) -> ChatResponse:
    """
    Handle RAG mode: full-book retrieval and generation.

    Args:
        query_text: User's question
        conversation_history: Formatted conversation history

    Returns:
        ChatResponse with citations
    """
    # Step 1: Embed query
    query_vector = embedding_service.embed_query(query_text)

    # Step 2: Search Qdrant for relevant chunks
    search_results = vector_service.search(
        query_vector=query_vector,
        limit=config.TOP_K_CHUNKS
    )

    if not search_results:
        # No relevant content found
        return ChatResponse(
            response="I couldn't find information about that in the book. Please try rephrasing your question or ask about a different topic covered in the Physical AI & Humanoid Robotics book.",
            source_chunks=[],
            mode=QueryMode.RAG,
            timestamp=datetime.utcnow()
        )

    # Step 3: Build context from retrieved chunks
    context_chunks = [result['text'] for result in search_results]

    # Step 4: Format prompt
    prompt = format_rag_prompt(
        query=query_text,
        context_chunks=context_chunks,
        conversation_history=conversation_history
    )

    # Step 5: Generate response
    response_text = generation_service.generate_response(prompt)

    # Step 6: Format source citations
    source_chunks = []
    for result in search_results:
        source_chunks.append(SourceChunk(
            chapter=result['chapter_title'],
            section=result.get('section_heading'),
            snippet=result['text'][:200] + "..."  # First 200 chars as snippet
        ))

    return ChatResponse(
        response=response_text,
        source_chunks=source_chunks,
        mode=QueryMode.RAG,
        timestamp=datetime.utcnow()
    )


async def _handle_selected_text_mode(
    query_text: str,
    selected_text: str,
    conversation_history: str
) -> ChatResponse:
    """
    Handle selected-text mode: context from user selection only.

    Args:
        query_text: User's question
        selected_text: User-selected text
        conversation_history: Formatted conversation history

    Returns:
        ChatResponse without citations (mode=selected_text)
    """
    # Validate selected text
    if len(selected_text) > config.MAX_SELECTED_TEXT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Selected text exceeds {config.MAX_SELECTED_TEXT_CHARS} characters"
        )

    # Format prompt
    prompt = format_selected_text_prompt(
        query=query_text,
        selected_text=selected_text,
        conversation_history=conversation_history
    )

    # Generate response
    response_text = generation_service.generate_response(prompt)

    return ChatResponse(
        response=response_text,
        source_chunks=[],  # No retrieval in selected-text mode
        mode=QueryMode.SELECTED_TEXT,
        timestamp=datetime.utcnow()
    )


@router.post("/chat-agent", response_model=ChatResponse)
async def chat_with_agent(query: ChatQuery):
    """
    Chat endpoint using OpenAI Agent with function calling.

    This endpoint uses the RAGAgent which:
    - Dynamically retrieves context via function calling
    - Ensures responses are grounded in retrieved content
    - Provides transparent reasoning traces
    - Includes confidence scores based on retrieval quality

    Args:
        query: ChatQuery with user question and optional conversation history

    Returns:
        ChatResponse with AI-generated answer, sources, and metadata

    Raises:
        HTTPException 400: Invalid query (too long)
        HTTPException 429: Rate limit exceeded
        HTTPException 500: Service error
    """
    try:
        # Validate query token count
        query_tokens = count_tokens(query.query)
        if query_tokens > config.MAX_QUERY_TOKENS:
            raise HTTPException(
                status_code=400,
                detail=f"Query exceeds {config.MAX_QUERY_TOKENS} tokens (got {query_tokens})"
            )

        # Format conversation history for agent
        conversation_history = None
        if query.conversation_history:
            conversation_history = [
                {"role": msg.role, "content": msg.content}
                for msg in query.conversation_history
            ]

        # Query the agent
        agent_response = await query_agent(
            user_query=query.query,
            conversation_history=conversation_history
        )

        # Map AgentResponse to ChatResponse format
        source_chunks = [
            SourceChunk(
                chapter=source.chapter,
                section=source.section,
                snippet=source.text[:200] + "..." if len(source.text) > 200 else source.text
            )
            for source in agent_response.sources
        ]

        return ChatResponse(
            response=agent_response.answer,
            source_chunks=source_chunks,
            mode=QueryMode.RAG,
            timestamp=datetime.utcnow(),
            metadata={
                "confidence": agent_response.confidence,
                "reasoning_steps": len(agent_response.reasoning_steps),
                "tool_calls": agent_response.metadata.get("tool_calls_count", 0),
                "total_tokens": agent_response.metadata.get("total_tokens", 0),
                "agent_mode": "function_calling"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "rate limit" in error_msg.lower() or "429" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again in a moment."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Agent chat processing failed: {error_msg}"
            )
