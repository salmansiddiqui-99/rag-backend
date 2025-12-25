"""Initial schema creation for RAG Chatbot

Revision ID: 001
Revises:
Create Date: 2025-12-10 22:00:00.000000

Creates tables:
- modules: Textbook modules (4 total)
- chapters: Individual chapters (12 total)
- content_chunks: Text segments with embeddings
- chat_sessions: Anonymous user conversation threads
- rag_queries: Individual queries and responses
- retrieved_chunks: Junction table for query-chunk relationships

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create modules table
    op.create_table(
        'modules',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.String(length=500), nullable=False),
        sa.Column('order', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('"order" >= 1 AND "order" <= 4', name='ck_module_order'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('order', name='uq_module_order'),
        sa.Index('ix_module_order', 'order'),
        sa.Index('ix_module_created_at', 'created_at'),
    )

    # Create chapters table
    op.create_table(
        'chapters',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('module_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('number', sa.Integer(), nullable=False),
        sa.Column('title', sa.String(length=150), nullable=False),
        sa.Column('content_markdown', sa.Text(), nullable=False),
        sa.Column('learning_objectives', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('references', postgresql.ARRAY(sa.String()), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='draft', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("number >= 1 AND number <= 12", name='ck_chapter_number'),
        sa.CheckConstraint("status IN ('draft', 'published', 'archived')", name='ck_chapter_status'),
        sa.ForeignKeyConstraint(['module_id'], ['modules.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('module_id', 'number', name='uq_chapter_module_number'),
        sa.Index('ix_chapter_module_id', 'module_id'),
        sa.Index('ix_chapter_created_at', 'created_at'),
    )

    # Create content_chunks table with Qdrant reference
    op.create_table(
        'content_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('qdrant_id', sa.String(length=64), nullable=False, unique=True),
        sa.Column('chapter_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('section_title', sa.String(length=100), nullable=False),
        sa.Column('text', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=False),
        sa.Column('embedding_model', sa.String(length=50), server_default='text-embedding-3-small', nullable=False),
        sa.Column('embedding_dimensions', sa.Integer(), server_default='384', nullable=False),
        sa.Column('embedding_created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('token_count >= 100 AND token_count <= 500', name='ck_chunk_token_bounds'),
        sa.ForeignKeyConstraint(['chapter_id'], ['chapters.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('qdrant_id', name='uq_chunk_qdrant_id'),
        sa.UniqueConstraint('chapter_id', 'chunk_index', name='uq_chunk_position'),
        sa.Index('ix_chunk_chapter_id', 'chapter_id'),
        sa.Index('ix_chunk_qdrant_id', 'qdrant_id'),
        sa.Index('ix_chunk_embedding_created', 'embedding_created_at'),
        sa.Index('ix_chunk_created_at', 'created_at'),
    )

    # Create chat_sessions table (anonymous sessions with 24-hour TTL)
    op.create_table(
        'chat_sessions',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_token', sa.String(length=128), nullable=False, unique=True),
        sa.Column('ip_hash', sa.String(length=64), nullable=False),
        sa.Column('browser_type', sa.String(length=100), nullable=True),
        sa.Column('initial_url', sa.String(length=1024), nullable=True),
        sa.Column('mode', sa.String(length=20), server_default='global', nullable=False),
        sa.Column('message_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_tokens_used', sa.Integer(), server_default='0', nullable=False),
        sa.Column('total_cost_usd', sa.Float(), server_default='0.0', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('last_activity', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.Column('expires_at', sa.DateTime(), nullable=False),
        sa.CheckConstraint('message_count >= 0', name='ck_session_message_count'),
        sa.CheckConstraint('total_cost_usd >= 0', name='ck_session_cost_positive'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_token', name='uq_session_token'),
        sa.Index('ix_session_token', 'session_token'),
        sa.Index('ix_session_created_at', 'created_at'),
        sa.Index('ix_session_expires_at', 'expires_at'),
    )

    # Create rag_queries table (individual queries with comprehensive metrics)
    op.create_table(
        'rag_queries',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('session_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chapter_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('query_text', sa.Text(), nullable=False),
        sa.Column('query_tokens', sa.Integer(), nullable=False),
        sa.Column('query_embedding_model', sa.String(length=50), server_default='text-embedding-3-small', nullable=True),
        sa.Column('retrieval_mode', sa.String(length=20), nullable=False),
        sa.Column('selected_text', sa.Text(), nullable=True),
        sa.Column('retrieved_chunk_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('retrieval_latency_ms', sa.Integer(), nullable=False),
        sa.Column('similarity_threshold_used', sa.Float(), server_default='0.5', nullable=False),
        sa.Column('max_similarity_score', sa.Float(), nullable=True),
        sa.Column('response_text', sa.Text(), nullable=False),
        sa.Column('response_tokens', sa.Integer(), nullable=False),
        sa.Column('llm_model', sa.String(length=50), server_default='gpt-4o', nullable=False),
        sa.Column('llm_latency_ms', sa.Integer(), nullable=False),
        sa.Column('temperature_used', sa.Float(), server_default='0.7', nullable=False),
        sa.Column('grounded_in_context', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('response_status', sa.String(length=20), nullable=False),
        sa.Column('user_feedback', sa.String(length=20), nullable=True),
        sa.Column('feedback_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("retrieval_mode IN ('global', 'selected_text')", name='ck_query_retrieval_mode'),
        sa.CheckConstraint("response_status IN ('success', 'no_context', 'error')", name='ck_query_response_status'),
        sa.CheckConstraint('retrieved_chunk_count >= 0 AND retrieved_chunk_count <= 10', name='ck_query_chunk_count'),
        sa.CheckConstraint('retrieval_latency_ms > 0 AND retrieval_latency_ms < 2000', name='ck_query_retrieval_latency'),
        sa.CheckConstraint('llm_latency_ms > 0 AND llm_latency_ms < 10000', name='ck_query_llm_latency'),
        sa.CheckConstraint('max_similarity_score IS NULL OR (max_similarity_score >= 0 AND max_similarity_score <= 1)', name='ck_query_similarity_bounds'),
        sa.ForeignKeyConstraint(['session_id'], ['chat_sessions.id'], ),
        sa.ForeignKeyConstraint(['chapter_id'], ['chapters.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('session_id', 'created_at', name='uq_query_session_timestamp'),
        sa.Index('ix_query_session_id', 'session_id'),
        sa.Index('ix_query_chapter_id', 'chapter_id'),
        sa.Index('ix_query_created_at', 'created_at'),
        sa.Index('ix_query_grounded', 'grounded_in_context'),
        sa.Index('ix_query_status', 'response_status'),
    )

    # Create retrieved_chunks table (junction: queries -> chunks)
    op.create_table(
        'retrieved_chunks',
        sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('query_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rank', sa.Integer(), nullable=False),
        sa.Column('similarity_score', sa.Float(), nullable=False),
        sa.Column('used_in_response', sa.Boolean(), server_default='true', nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint('rank >= 1 AND rank <= 10', name='ck_retrieved_rank_bounds'),
        sa.CheckConstraint('similarity_score >= 0 AND similarity_score <= 1', name='ck_retrieved_similarity_bounds'),
        sa.ForeignKeyConstraint(['query_id'], ['rag_queries.id'], ),
        sa.ForeignKeyConstraint(['chunk_id'], ['content_chunks.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('query_id', 'chunk_id', name='uq_retrieved_query_chunk'),
        sa.UniqueConstraint('query_id', 'rank', name='uq_retrieved_query_rank'),
        sa.Index('ix_retrieved_query_id', 'query_id'),
        sa.Index('ix_retrieved_chunk_id', 'chunk_id'),
        sa.Index('ix_retrieved_rank', 'rank'),
        sa.Index('ix_retrieved_similarity', 'similarity_score'),
    )


def downgrade() -> None:
    # Drop tables in reverse order (dependencies first)
    op.drop_table('retrieved_chunks')
    op.drop_table('rag_queries')
    op.drop_table('chat_sessions')
    op.drop_table('content_chunks')
    op.drop_table('chapters')
    op.drop_table('modules')
