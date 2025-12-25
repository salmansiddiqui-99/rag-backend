"""Unit tests for RAGService retrieval and context management (T059)"""
import pytest
from uuid import UUID, uuid4
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from src.services.rag_service import RAGService
from src.models.rag import RetrievalMode, ResponseStatus, RetrievedChunkData


@pytest.fixture
def mock_db():
    """Create a mock database session"""
    return Mock()


@pytest.fixture
def mock_qdrant_client():
    """Create a mock Qdrant client"""
    return Mock()


@pytest.fixture
def rag_service(mock_db):
    """Create RAGService instance with mocked clients"""
    with patch('src.services.rag_service.QdrantClient'):
        with patch('src.services.rag_service.openai.Client'):
            service = RAGService(mock_db)
            service.qdrant_client = Mock()
            service.openai_client = Mock()
            return service


class TestEmbedQuery:
    """Tests for embed_query method (embedding and caching)"""

    def test_embed_query_uses_cache_on_repeat(self, rag_service):
        """T052: embed_query should return cached embedding for identical queries"""
        query = "test query"
        embedding = [0.1, 0.2, 0.3]

        # Mock OpenAI response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=embedding)]
        rag_service.openai_client.embeddings.create.return_value = mock_response

        # First call - should hit API
        result1 = rag_service.embed_query(query)
        assert result1 == embedding
        assert rag_service.openai_client.embeddings.create.call_count == 1

        # Second call - should use cache
        result2 = rag_service.embed_query(query)
        assert result2 == embedding
        assert rag_service.openai_client.embeddings.create.call_count == 1  # No additional call

    def test_embed_query_case_insensitive_cache(self, rag_service):
        """T052: Cache key should be case-insensitive"""
        embedding = [0.1, 0.2, 0.3]
        mock_response = Mock()
        mock_response.data = [Mock(embedding=embedding)]
        rag_service.openai_client.embeddings.create.return_value = mock_response

        # Different case should hit cache
        result1 = rag_service.embed_query("Test Query")
        result2 = rag_service.embed_query("test query")
        assert result1 == result2
        assert rag_service.openai_client.embeddings.create.call_count == 1

    def test_embed_query_lru_eviction(self, rag_service):
        """T052: Cache should evict oldest entry when size exceeded"""
        rag_service.cache_max_size = 2
        embedding = [0.1, 0.2]
        mock_response = Mock()
        mock_response.data = [Mock(embedding=embedding)]
        rag_service.openai_client.embeddings.create.return_value = mock_response

        # Add 3 items to exceed cache size of 2
        result1 = rag_service.embed_query("query1")
        result2 = rag_service.embed_query("query2")
        result3 = rag_service.embed_query("query3")

        # Cache should only contain last 2
        assert len(rag_service.embedding_cache) == 2
        assert "query1" not in rag_service.embedding_cache

    def test_embed_query_api_error(self, rag_service):
        """embed_query should raise ValueError on API error"""
        rag_service.openai_client.embeddings.create.side_effect = Exception("API Error")

        with pytest.raises(ValueError, match="Embedding failed"):
            rag_service.embed_query("test query")


class TestRetrieveChunks:
    """Tests for retrieve_chunks method (retrieval pipeline)"""

    def test_retrieve_chunks_performance_logging(self, rag_service, mocker):
        """T054: retrieve_chunks should log performance metrics"""
        rag_service._search_vectors = Mock(return_value=[])
        logger_mock = mocker.patch('src.services.rag_service.logger')
        rag_service.embed_query = Mock(return_value=[0.1, 0.2])

        result = rag_service.retrieve_chunks(
            query_text="test",
            retrieval_mode=RetrievalMode.GLOBAL
        )

        # Should log with timing info
        assert logger_mock.info.called
        call_args = logger_mock.info.call_args[0][0]
        assert "Retrieval pipeline completed" in call_args
        assert "embed=" in call_args
        assert "ms" in call_args

    def test_retrieve_chunks_text_selection_mode(self, rag_service, mocker):
        """Text-selection mode should use keyword matching"""
        rag_service._retrieve_from_selected_text = Mock(return_value=[])

        result = rag_service.retrieve_chunks(
            query_text="test",
            retrieval_mode=RetrievalMode.TEXT_SELECTION,
            selected_text="some selected text"
        )

        rag_service._retrieve_from_selected_text.assert_called_once()

    def test_retrieve_chunks_chapter_filter(self, rag_service):
        """Chapter-specific mode should pass chapter_id to search"""
        chapter_id = uuid4()
        rag_service._search_vectors = Mock(return_value=[])
        rag_service.embed_query = Mock(return_value=[0.1, 0.2])

        rag_service.retrieve_chunks(
            query_text="test",
            retrieval_mode=RetrievalMode.CHAPTER_SPECIFIC,
            chapter_id=chapter_id
        )

        rag_service._search_vectors.assert_called_once()
        call_args = rag_service._search_vectors.call_args
        assert call_args[1]["chapter_id"] == chapter_id


class TestSearchVectors:
    """Tests for _search_vectors method (vector similarity search)"""

    def test_search_vectors_batch_fetching(self, rag_service, mocker):
        """T053: _search_vectors should batch fetch chunks"""
        chunk_id = uuid4()
        mock_result = Mock()
        mock_result.score = 0.9
        mock_result.payload = {"chunk_id": str(chunk_id)}

        rag_service.qdrant_client.search.return_value = [mock_result]
        rag_service._batch_fetch_chunks = Mock(return_value={
            chunk_id: Mock(id=chunk_id, chapter_id=uuid4(), section_title="Test", text="Test content", token_count=100)
        })

        result = rag_service._search_vectors(query_embedding=[0.1, 0.2])

        # Should call batch fetch
        rag_service._batch_fetch_chunks.assert_called_once()

    def test_search_vectors_similarity_threshold(self, rag_service, mocker):
        """T051: _search_vectors should filter by similarity threshold"""
        chunk_id = uuid4()
        mock_result = Mock()
        mock_result.score = 0.3  # Below threshold (0.75 default)
        mock_result.payload = {"chunk_id": str(chunk_id)}

        rag_service.qdrant_client.search.return_value = [mock_result]
        logger_mock = mocker.patch('src.services.rag_service.logger')

        result = rag_service._search_vectors(
            query_embedding=[0.1, 0.2],
            top_k=5
        )

        # Should skip low-similarity chunk
        assert len(result) == 0
        assert logger_mock.debug.called
        call_args = logger_mock.debug.call_args[0][0]
        assert "Skipping chunk" in call_args

    def test_search_vectors_performance_metrics(self, rag_service, mocker):
        """T054: _search_vectors should log search timing"""
        chunk_id = uuid4()
        mock_result = Mock()
        mock_result.score = 0.9
        mock_result.payload = {"chunk_id": str(chunk_id)}

        rag_service.qdrant_client.search.return_value = [mock_result]
        rag_service._batch_fetch_chunks = Mock(return_value={
            chunk_id: Mock(id=chunk_id, chapter_id=uuid4(), section_title="Test", text="Test content")
        })
        logger_mock = mocker.patch('src.services.rag_service.logger')

        result = rag_service._search_vectors(query_embedding=[0.1, 0.2])

        # Should log timing metrics
        assert logger_mock.debug.called
        call_args = logger_mock.debug.call_args[0][0]
        assert "search=" in call_args
        assert "fetch=" in call_args
        assert "ms" in call_args

    def test_search_vectors_with_chapter_filter(self, rag_service):
        """_search_vectors should apply chapter filter"""
        chapter_id = uuid4()

        result = rag_service._search_vectors(
            query_embedding=[0.1, 0.2],
            chapter_id=chapter_id
        )

        # Check that Qdrant was called with filter
        rag_service.qdrant_client.search.assert_called_once()
        call_args = rag_service.qdrant_client.search.call_args
        assert call_args[1]["query_filter"] is not None


class TestBatchFetchChunks:
    """Tests for _batch_fetch_chunks method (T053)"""

    def test_batch_fetch_chunks_empty_list(self, rag_service):
        """_batch_fetch_chunks should handle empty list"""
        result = rag_service._batch_fetch_chunks([])
        assert result == {}

    def test_batch_fetch_chunks_multiple(self, rag_service):
        """_batch_fetch_chunks should fetch multiple chunks efficiently"""
        chunk_ids = [uuid4(), uuid4(), uuid4()]

        mock_chunks = [
            Mock(id=chunk_ids[0]),
            Mock(id=chunk_ids[1]),
            Mock(id=chunk_ids[2])
        ]
        rag_service.db.query.return_value.filter.return_value.all.return_value = mock_chunks

        result = rag_service._batch_fetch_chunks(chunk_ids)

        # Should return dict mapping chunk_id to chunk
        assert len(result) == 3
        assert all(cid in result for cid in chunk_ids)

        # Should use single query with IN clause
        rag_service.db.query.assert_called_once()


class TestLogRagQuery:
    """Tests for log_rag_query method (audit trail)"""

    def test_log_rag_query_creates_records(self, rag_service):
        """log_rag_query should create RAGQuery and RetrievedChunk records"""
        query_id = uuid4()
        rag_service.db.add = Mock()
        rag_service.db.flush = Mock()
        rag_service.db.commit = Mock()

        # Mock the RAGQuery model
        with patch('src.services.rag_service.RAGQuery') as MockRAGQuery:
            mock_query = Mock(id=query_id)
            MockRAGQuery.return_value = mock_query

            chunks = [
                RetrievedChunkData(
                    chunk_id=uuid4(),
                    chapter_id=uuid4(),
                    section_title="Test",
                    text="content",
                    similarity_score=0.9,
                    rank=1
                )
            ]

            result = rag_service.log_rag_query(
                query_text="test",
                retrieval_mode=RetrievalMode.GLOBAL,
                retrieved_chunks=chunks,
                response_status=ResponseStatus.SUCCESS
            )

            # Should create records
            assert rag_service.db.add.called
            assert rag_service.db.commit.called


class TestGetRetrievalStats:
    """Tests for get_retrieval_stats method"""

    def test_get_retrieval_stats_success(self, rag_service, mocker):
        """get_retrieval_stats should return system statistics"""
        # Mock database queries
        rag_service.db.query.return_value.count.return_value = 100
        rag_service.db.query.return_value.distinct.return_value.count.return_value = 12
        rag_service.db.query.return_value.all.return_value = [
            Mock(token_count=200) for _ in range(10)
        ]

        # Mock Qdrant collection info
        mock_collection = Mock(points_count=100)
        rag_service.qdrant_client.get_collection.return_value = mock_collection

        result = rag_service.get_retrieval_stats()

        assert result["total_chunks_indexed"] == 100
        assert result["total_chapters"] == 12
        assert "avg_chunk_tokens" in result
        assert "embedding_model" in result
        assert "vectors_indexed" in result
