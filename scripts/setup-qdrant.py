"""Script to initialize Qdrant collection for chapter embeddings"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import settings

def setup_qdrant_collection() -> None:
    """Create Qdrant collection for chapter embeddings if it doesn't exist"""
    try:
        # Initialize Qdrant client
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

        # Check if collection exists
        try:
            collection_info = client.get_collection(settings.QDRANT_COLLECTION)
            print(f"[OK] Collection '{settings.QDRANT_COLLECTION}' already exists")
            return
        except Exception:
            pass

        # Create collection
        client.recreate_collection(
            collection_name=settings.QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=settings.QDRANT_VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )

        print(f"[OK] Created Qdrant collection '{settings.QDRANT_COLLECTION}'")
        print(f"  - Vector size: {settings.QDRANT_VECTOR_SIZE}")
        print(f"  - Distance metric: {settings.QDRANT_DISTANCE_METRIC}")

        # Verify collection was created
        collection_info = client.get_collection(settings.QDRANT_COLLECTION)
        print(f"[OK] Collection verified: {collection_info.points_count} points")

    except Exception as e:
        print(f"[ERROR] Error setting up Qdrant collection: {e}")
        raise


if __name__ == "__main__":
    try:
        print("Setting up Qdrant collection...")
        setup_qdrant_collection()
        print("\n[OK] Qdrant setup complete!")
    except Exception as e:
        print(f"\n[ERROR] Setup failed: {e}")
        sys.exit(1)
