"""
Simple validation script for the OpenAI RAG Agent.

Tests:
1. Simple factual query
2. Multi-step retrieval (complex question)
3. Grounding validation (out-of-scope query)
4. Citation validation (specific topic)
"""
import sys
import asyncio
from pathlib import Path

# Add src to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

# Windows UTF-8 encoding fix
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from src.agent import query_agent


# ============================================================================
# Test Cases
# ============================================================================

TEST_CASES = [
    {
        "name": "Simple Factual Query",
        "query": "What is ROS 2?",
        "expected_criteria": [
            "Should mention ROS 2",
            "Should cite chapter/module",
            "Should have high/medium confidence",
            "Should have at least 1 source"
        ]
    },
    {
        "name": "Multi-Step Retrieval",
        "query": "How do ROS 2 nodes communicate with each other?",
        "expected_criteria": [
            "Should mention communication mechanisms (topics, services, actions)",
            "Should cite specific sections",
            "Should have multiple sources",
            "Should mention DDS or middleware"
        ]
    },
    {
        "name": "Grounding Validation (Out-of-Scope)",
        "query": "What is TensorFlow?",
        "expected_criteria": [
            "Should acknowledge information not found",
            "Should NOT hallucinate TensorFlow details",
            "Should have low confidence",
            "Should suggest it's outside book scope"
        ]
    },
    {
        "name": "Citation Validation",
        "query": "Explain DDS in ROS 2",
        "expected_criteria": [
            "Should cite Module 1 or ROS 2 Architecture",
            "Should explain DDS (Data Distribution Service)",
            "Should have specific chapter/section citations",
            "Should have high confidence if found"
        ]
    }
]


# ============================================================================
# Validation Functions
# ============================================================================

def print_separator(char="=", length=80):
    """Print a separator line."""
    print(char * length)


def print_test_header(test_num: int, test_name: str, query: str):
    """Print test header."""
    print(f"\n{'='*80}")
    print(f"TEST {test_num}: {test_name}")
    print(f"{'='*80}")
    print(f"Query: \"{query}\"")
    print()


def print_agent_response(response):
    """Print agent response with formatting."""
    print(f"ANSWER:")
    print("-" * 80)
    print(response.answer)
    print()

    print(f"CONFIDENCE: {response.confidence.upper()}")
    print()

    print(f"SOURCES ({len(response.sources)}):")
    print("-" * 80)
    for i, source in enumerate(response.sources, 1):
        print(f"{i}. Chapter: {source.chapter}")
        print(f"   Section: {source.section}")
        print(f"   Relevance: {source.relevance_score:.3f}")
        print(f"   Text: {source.text[:100]}...")
        print()

    print(f"REASONING STEPS ({len(response.reasoning_steps)}):")
    print("-" * 80)
    for i, step in enumerate(response.reasoning_steps, 1):
        print(f"{i}. Action: {step.action}")
        if step.query:
            print(f"   Query: {step.query}")
        if step.num_chunks:
            print(f"   Retrieved: {step.num_chunks} chunks")
        if step.details:
            print(f"   Details: {step.details}")
        print()

    print(f"METADATA:")
    print("-" * 80)
    print(f"Total Tokens: {response.metadata.get('total_tokens', 'N/A')}")
    print(f"Tool Calls: {response.metadata.get('tool_calls_count', 0)}")
    print(f"Iterations: {response.metadata.get('iterations', 'N/A')}")
    print(f"Finish Reason: {response.metadata.get('finish_reason', 'N/A')}")
    print()


def print_expected_criteria(criteria: list):
    """Print expected validation criteria."""
    print("EXPECTED CRITERIA:")
    print("-" * 80)
    for criterion in criteria:
        print(f"  • {criterion}")
    print()


async def run_test(test_num: int, test_case: dict):
    """Run a single test case."""
    print_test_header(test_num, test_case["name"], test_case["query"])
    print_expected_criteria(test_case["expected_criteria"])

    try:
        # Query the agent
        response = await query_agent(test_case["query"])

        # Print response
        print_agent_response(response)

        print("✓ Test completed successfully")

    except Exception as e:
        print(f"✗ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


async def main():
    """Run all validation tests."""
    print("=" * 80)
    print("OPENAI RAG AGENT VALIDATION")
    print("=" * 80)
    print(f"Running {len(TEST_CASES)} test cases...")
    print()

    for i, test_case in enumerate(TEST_CASES, 1):
        await run_test(i, test_case)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print("\nManual Review Required:")
    print("  1. Verify answers are grounded in retrieved content")
    print("  2. Check citation quality (chapter/section accuracy)")
    print("  3. Confirm out-of-scope query handling")
    print("  4. Validate confidence scores match retrieval quality")


if __name__ == "__main__":
    asyncio.run(main())
