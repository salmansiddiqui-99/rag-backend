"""
OpenAI Agent with RAG integration using the official OpenAI Agents SDK.

This module implements an intelligent agent that:
- Uses OpenAI Agents SDK with function calling to dynamically retrieve context
- Grounds all responses in retrieved book content
- Provides transparent reasoning traces
- Includes confidence scores based on retrieval quality
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field

# Required SDK imports for RAGAgent
try:
    from agents import Agent, Runner, function_tool, RunContextWrapper, SQLiteSession
    HAS_AGENTS_SDK = True
except ImportError:
    HAS_AGENTS_SDK = False
    # Fallback to dummy implementation if SDK is not available
    def function_tool(func):
        return func

    # Define dummy classes for when the SDK is not available
    class Agent:
        pass

    class Runner:
        @staticmethod
        def run(*args, **kwargs):
            raise NotImplementedError("OpenAI Agents SDK is not installed")

    class RunContextWrapper:
        pass

    class SQLiteSession:
        def __init__(self, *args, **kwargs):
            pass

import asyncio

from src.config import config
from src.services.embedding_service import embedding_service
from src.services.vector_service import vector_service


# ============================================================================
# Response Models
# ============================================================================

class ReasoningStep(BaseModel):
    """A step in the agent's reasoning process."""
    action: str = Field(..., description="Type of action: retrieve_context, reasoning, synthesis")
    query: Optional[str] = Field(None, description="Search query if action is retrieve_context")
    num_chunks: Optional[int] = Field(None, description="Number of chunks retrieved")
    details: Optional[str] = Field(None, description="Additional details about this step")


class SourceChunk(BaseModel):
    """A retrieved source chunk with metadata."""
    text: str = Field(..., description="Chunk text content")
    chapter: str = Field(..., description="Chapter/module title")
    section: Optional[str] = Field(None, description="Section heading")
    relevance_score: float = Field(..., description="Cosine similarity score (0-1)")
    source: str = Field(..., description="Source file path or URL")


class AgentResponse(BaseModel):
    """Complete agent response with answer, sources, and reasoning."""
    answer: str = Field(..., description="Final answer to user query")
    sources: List[SourceChunk] = Field(default_factory=list, description="Retrieved source chunks")
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list, description="Agent's reasoning trace")
    confidence: Literal["high", "medium", "low"] = Field(..., description="Confidence in answer")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata (tokens, timing, etc.)")


# ============================================================================
# System Prompt
# ============================================================================

AGENT_SYSTEM_PROMPT = """You are an AI tutor for Physical AI and Humanoid Robotics.

Your role is to answer questions using ONLY information from the course book.

WORKFLOW:
1. When a user asks a question, use the retrieve_context tool to search the book
2. You may call retrieve_context multiple times with different queries to gather comprehensive information
3. Synthesize your answer ONLY from the retrieved context
4. Always cite the chapter and section where information came from
5. If the retrieved context doesn't contain enough information, say so explicitly

GROUNDING RULES:
- NEVER use knowledge outside the retrieved context
- ALWAYS cite sources (chapter, section)
- If uncertain, state your confidence level
- If context is insufficient, acknowledge gaps

CITATION FORMAT:
"According to Module 1: ROS 2 Architecture, ..."
"As explained in the Digital Twin Simulation section, ..."

IMPORTANT: You must use the retrieve_context tool before answering. Do not answer from memory."""


# ============================================================================
# Function Tools
# ============================================================================

@function_tool
def retrieve_context(search_query: str, num_chunks: int = 5) -> dict:
    """
    Retrieve relevant context from the Physical AI book using semantic search.

    Use this tool to find information before answering user questions.
    You can call this multiple times with different queries to gather comprehensive information.

    Args:
        search_query: Query to search for in the book. Be specific and focused.
        num_chunks: Number of chunks to retrieve (1-10). Use more chunks for complex topics. Default is 5.

    Returns:
        Dictionary containing retrieved chunks with chapter, section, relevance score, and text.
    """
    # Validate num_chunks
    num_chunks = max(1, min(10, num_chunks))

    # 1. Embed query
    query_vector = embedding_service.embed_query(search_query)

    # 2. Search Qdrant
    results = vector_service.search(query_vector, limit=num_chunks)

    # 3. Format for agent
    return {
        "chunks": [
            {
                "text": r["text"],
                "chapter": r["chapter_title"],
                "section": r.get("section_heading") or None,
                "relevance_score": r["score"],
                "source": r["source_file_path"]
            }
            for r in results
        ],
        "total_retrieved": len(results),
        "search_query": search_query
    }


# ============================================================================
# RAG Agent with OpenAI Agents SDK
# ============================================================================

class RAGAgent:
    """OpenAI Agent with RAG capabilities using the official Agents SDK."""

    def __init__(self):
        """Initialize the RAG agent with OpenAI Agents SDK."""
        if not HAS_AGENTS_SDK:
            raise ImportError("The OpenAI Agents SDK is required for this module. Please install it using: pip install openai-agents")

        self.embedding_service = embedding_service
        self.vector_service = vector_service

        # Create the main agent with the RAG system prompt and tools
        self.agent = Agent(
            name="RAG Agent",
            instructions=AGENT_SYSTEM_PROMPT,
            tools=[retrieve_context],
            model=config.OPENROUTER_MODEL  # Using OpenRouter model
        )

    async def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AgentResponse:
        """
        Process user query using the OpenAI Agents SDK.

        This method uses the SDK's built-in agent loop to handle tool calls
        and response generation while tracking sources and reasoning.
        """
        # Prepare the session to maintain conversation history if provided
        session = None
        if conversation_history:
            # Create a temporary session to maintain context
            session = SQLiteSession("temp_session", ":memory:")
            # Add conversation history to the session
            formatted_history = []
            for msg in conversation_history:
                formatted_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            await session.add_items(formatted_history)

        # Run the agent with the user query
        result = await Runner.run(
            self.agent,
            user_query,
            session=session
        )

        # Extract the final response
        final_output = result.final_output

        # The agent should have called retrieve_context tool during execution
        # We need to extract the reasoning steps and sources from the session or tool calls
        reasoning_steps = []
        all_sources = []

        # If session exists, we can get the conversation items to extract reasoning
        if session:
            items = await session.get_items()
            for item in items:
                if item.get("role") == "tool" and "retrieve_context" in str(item.get("content", "")):
                    # Extract tool call information to build reasoning steps
                    # This would require parsing the session items to extract tool call results
                    reasoning_steps.append(
                        ReasoningStep(
                            action="retrieve_context",
                            query="Extracted from tool call",
                            num_chunks=5,  # Placeholder - would need to extract from actual call
                            details="Context retrieved via RAG system"
                        )
                    )

        # For now, we'll create a basic response structure
        # In a real implementation, we'd need to track the tool calls more carefully
        unique_sources = self._extract_sources_from_output(final_output)

        source_chunks = [
            SourceChunk(
                text=s["text"],
                chapter=s["chapter"],
                section=s["section"],
                relevance_score=s["relevance_score"],
                source=s["source"]
            )
            for s in unique_sources
        ]

        confidence = self._determine_confidence(unique_sources)

        metadata = {
            "total_tokens": 0,  # SDK doesn't provide token usage in the same way
            "tool_calls_count": len(reasoning_steps),
            "iterations": 1,  # SDK handles iterations internally
            "finish_reason": "completed"
        }

        return AgentResponse(
            answer=final_output,
            sources=source_chunks,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            metadata=metadata
        )

    def _extract_sources_from_output(self, output: str) -> List[dict]:
        """
        Extract source information from agent output.
        This is a simplified approach - in practice, you'd want to track sources
        more systematically during tool calls.
        """
        # This is a placeholder implementation
        # In a real implementation, we'd track sources during tool execution
        return []

    def _determine_confidence(self, sources: List[dict]) -> Literal["high", "medium", "low"]:
        """Determine confidence based on retrieval quality."""
        if not sources:
            return "low"

        avg_score = sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0

        if avg_score > 0.6 and len(sources) >= 3:
            return "high"
        elif avg_score > 0.4:
            return "medium"
        else:
            return "low"


# ============================================================================
# Improved RAG Agent with Better Source Tracking
# ============================================================================

class RAGAgent:
    """OpenAI Agent with RAG capabilities using the official Agents SDK."""

    def __init__(self):
        """Initialize the RAG agent with OpenAI Agents SDK."""
        if not HAS_AGENTS_SDK:
            raise ImportError("The OpenAI Agents SDK is required for this module. Please install it using: pip install openai-agents")

        self.embedding_service = embedding_service
        self.vector_service = vector_service

        # Create the main agent with the RAG system prompt and tools
        self.agent = Agent(
            name="RAG Agent",
            instructions=AGENT_SYSTEM_PROMPT,
            tools=[retrieve_context],
            model=config.OPENROUTER_MODEL  # Using OpenRouter model
        )

    async def query(
        self,
        user_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> AgentResponse:
        """
        Process user query using the OpenAI Agents SDK with improved source tracking.
        """
        # Prepare the session to maintain conversation history if provided
        session = None
        if conversation_history:
            # Create a temporary session to maintain context
            session = SQLiteSession("temp_session", ":memory:")
            # Add conversation history to the session
            formatted_history = []
            for msg in conversation_history:
                formatted_history.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            await session.add_items(formatted_history)

        # Run the agent with the user query
        result = await Runner.run(
            self.agent,
            user_query,
            session=session
        )

        # Extract the final response
        final_output = result.final_output

        # We need to track sources and reasoning more systematically
        # For this, we'll create a custom agent that captures tool call results
        reasoning_steps, all_sources = await self._extract_tool_calls_and_sources(session)

        unique_sources = self._deduplicate_sources(all_sources)

        source_chunks = [
            SourceChunk(
                text=s["text"],
                chapter=s["chapter"],
                section=s["section"],
                relevance_score=s["relevance_score"],
                source=s["source"]
            )
            for s in unique_sources
        ]

        confidence = self._determine_confidence(unique_sources)

        metadata = {
            "total_tokens": 0,  # SDK doesn't provide token usage in the same way
            "tool_calls_count": len(reasoning_steps),
            "iterations": 1,  # SDK handles iterations internally
            "finish_reason": "completed"
        }

        return AgentResponse(
            answer=final_output,
            sources=source_chunks,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
            metadata=metadata
        )

    async def _extract_tool_calls_and_sources(self, session):
        """Extract tool calls and sources from the session."""
        reasoning_steps = []
        all_sources = []

        if session:
            items = await session.get_items()
            for item in items:
                if item.get("role") == "tool":
                    # Parse the tool call result
                    content = item.get("content", "")
                    if "retrieve_context" in content:
                        try:
                            import json
                            tool_result = json.loads(content)
                            if "chunks" in tool_result:
                                all_sources.extend(tool_result["chunks"])
                                reasoning_steps.append(
                                    ReasoningStep(
                                        action="retrieve_context",
                                        query=tool_result.get("search_query", "Unknown query"),
                                        num_chunks=tool_result.get("total_retrieved", 0),
                                        details=f"Retrieved {tool_result.get('total_retrieved', 0)} chunks"
                                    )
                                )
                        except json.JSONDecodeError:
                            # If JSON parsing fails, skip this item
                            continue

        return reasoning_steps, all_sources

    def _determine_confidence(self, sources: List[dict]) -> Literal["high", "medium", "low"]:
        """Determine confidence based on retrieval quality."""
        if not sources:
            return "low"

        avg_score = sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0

        if avg_score > 0.6 and len(sources) >= 3:
            return "high"
        elif avg_score > 0.4:
            return "medium"
        else:
            return "low"

    def _deduplicate_sources(self, sources: List[dict]) -> List[dict]:
        """Remove duplicate chunks based on text content."""
        seen = set()
        unique = []

        for source in sources:
            text_hash = hash(source["text"][:100] if source["text"] else "")

            if text_hash not in seen:
                seen.add(text_hash)
                unique.append(source)

        return unique


# ============================================================================
# Singleton Pattern & Convenience Functions
# ============================================================================

# Use the SDK-based implementation
_agent_instance: Optional[RAGAgent] = None


def get_agent() -> RAGAgent:
    """
    Get or create singleton RAGAgent instance.

    Returns:
        RAGAgent singleton instance using OpenAI Agents SDK
    """
    global _agent_instance
    if not HAS_AGENTS_SDK:
        raise ImportError("The OpenAI Agents SDK is required for this module. Please install it using: pip install openai-agents")

    if _agent_instance is None:
        _agent_instance = RAGAgent()
    return _agent_instance


async def query_agent(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> AgentResponse:
    """
    Convenience function to query the RAG agent.

    Args:
        user_query: User's question
        conversation_history: Optional conversation history

    Returns:
        AgentResponse with answer, sources, and reasoning

    Example:
        >>> response = await query_agent("What is ROS 2?")
        >>> print(response.answer)
        >>> print(f"Confidence: {response.confidence}")
        >>> print(f"Sources: {len(response.sources)}")
    """
    agent = get_agent()
    return await agent.query(user_query, conversation_history)


# ============================================================================
# Sync Wrapper (for non-async contexts)
# ============================================================================

def query_agent_sync(
    user_query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> AgentResponse:
    """
    Synchronous wrapper for query_agent.

    Args:
        user_query: User's question
        conversation_history: Optional conversation history

    Returns:
        AgentResponse with answer, sources, and reasoning
    """
    import asyncio

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run async function
    return loop.run_until_complete(query_agent(user_query, conversation_history))
