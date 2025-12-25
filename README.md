# RAG Chatbot Backend API

Backend service for the Integrated RAG Chatbot for the Physical AI & Humanoid Robotics Course textbook.

## Setup Instructions

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### 1. Create Virtual Environment

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

**Required credentials:**
- `OPENAI_API_KEY` - OpenAI API key (https://platform.openai.com/api-keys)
- `QDRANT_URL` - Qdrant Cloud instance URL (https://cloud.qdrant.io/)
- `QDRANT_API_KEY` - Qdrant Cloud API key
- `DATABASE_URL` - Neon Postgres connection string (https://neon.tech/)

See `.env.setup-guide.md` for detailed setup instructions.

### 4. Initialize Database

```bash
# Run database migrations
alembic upgrade head
```

### 5. Run the Server

```bash
# Development mode with auto-reload
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

# Production mode
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

View API documentation at `http://localhost:8000/docs`

## Project Structure

```
backend/
├── src/
│   ├── api/              # FastAPI route handlers
│   │   ├── chapters.py   # Chapter endpoints
│   │   ├── rag.py        # RAG retrieval endpoints
│   │   ├── chatbot.py    # Chatbot streaming endpoints
│   │   └── health.py     # Health check endpoints
│   ├── models/           # Data models and schemas
│   │   ├── database.py   # SQLAlchemy ORM models
│   │   ├── chapter.py    # Chapter entities
│   │   ├── module.py     # Module entities
│   │   └── rag.py        # RAG request/response schemas
│   ├── services/         # Business logic
│   │   ├── rag_service.py        # RAG retrieval orchestration
│   │   ├── chatbot_service.py    # Chatbot orchestration
│   │   ├── embedding.py          # Text embedding service
│   │   ├── chunking.py           # Text chunking service
│   │   └── response_verifier.py  # Hallucination detection
│   ├── config.py         # Environment configuration
│   ├── errors.py         # Custom exception classes
│   └── main.py           # FastAPI app entry point
├── tests/                # Unit and integration tests
├── scripts/              # Utility scripts
├── alembic/              # Database migration configs
├── .env.example          # Environment variables template
├── .env.setup-guide.md   # Detailed setup guide
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## API Endpoints

### Health Checks
- `GET /health` - Service health status
- `GET /ready` - Readiness probe for load balancers
- `GET /live` - Liveness probe

### RAG Retrieval
- `POST /api/query` - Retrieve relevant chunks for a query
  - Input: `QueryRequest { query: str, top_k: int }`
  - Output: `QueryResponse { chunks: List[TextChunk], total_found: int }`

### LLM Chat (Streaming)
- `POST /api/chatbot/query` - Get streamed response from LLM
  - Input: `ChatRequest { query: str, mode: "global" | "selected_text", selected_text?: str }`
  - Output: Server-Sent Events stream of ChatResponse messages

### Text Embedding
- `POST /api/embed` - Get vector embedding for text
  - Input: `EmbedRequest { text: str }`
  - Output: `EmbedResponse { embedding: List[float], dimensions: int }`

### Chapters
- `GET /api/chapters` - List all chapters
- `GET /api/chapters/{chapter_id}` - Get chapter details
- `POST /api/chapters/ingest` - Ingest chapter content into RAG

## Key Services

### RAGService
Handles vector search and chunk retrieval:
- `retrieve_chunks(query, top_k, threshold)` - Get most relevant chunks
- Performance target: <800ms for retrieval

### ChatbotService
Orchestrates retrieval and LLM response generation:
- `stream_chat(query, mode, selected_text)` - Stream AI response
- Performance target: <2s end-to-end latency
- Implements hallucination detection: rejects responses when no relevant context

### EmbeddingService
Text vectorization:
- Uses OpenAI text-embedding-3-small (384 dimensions)
- Cached for performance optimization

### ChunkingService
Text fragmentation for vector storage:
- Chunks 200-400 tokens using tiktoken
- Overlap between chunks for context preservation

## Environment Variables

See `.env.setup-guide.md` for complete configuration reference.

Key variables:
```
# API Configuration
API_HOST=127.0.0.1
API_PORT=8000
DEBUG=true

# Database
DATABASE_URL=postgresql://user:password@host/dbname

# Vector Database
QDRANT_URL=https://your-instance.qdrant.io
QDRANT_API_KEY=your_api_key
QDRANT_COLLECTION_NAME=aibook

# LLM (OpenAI)
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# RAG Configuration
RAG_SIMILARITY_THRESHOLD=0.5
RAG_TOP_K_CHUNKS=5
RAG_MAX_RESPONSE_TOKENS=1000
```

## Testing

Run unit tests:
```bash
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Development

### Code Style
- Follow PEP 8 style guide
- Use type hints in all functions
- Format with Black: `black src/ tests/`
- Lint with flake8: `flake8 src/ tests/`

### Making Changes
1. Create a feature branch
2. Write tests first (TDD approach)
3. Implement functionality
4. Ensure all tests pass
5. Create pull request

## Troubleshooting

### ImportError: No module named 'src'
Make sure to run commands from the `backend/` directory, not from a parent directory.

### Qdrant Connection Error
Check that:
- `QDRANT_URL` is correct and accessible
- `QDRANT_API_KEY` has proper permissions
- Network allows connection to Qdrant Cloud

### OpenAI API Error
Check that:
- `OPENAI_API_KEY` is valid (format: `sk-proj-...`)
- API key has sufficient quota
- Model name in `OPENAI_MODEL` is correct

### Database Connection Error
Check that:
- `DATABASE_URL` format is correct: `postgresql://user:password@host:port/database`
- Neon Postgres is accessible from your network
- Credentials are correct

## Deployment

### Docker
```bash
# Build
docker build -t rag-chatbot-backend .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-proj-... \
  -e QDRANT_URL=https://... \
  -e DATABASE_URL=postgresql://... \
  rag-chatbot-backend
```

### Production Checklist
- [ ] Set `DEBUG=false`
- [ ] Set `API_DEBUG=false`
- [ ] Set `LOG_LEVEL=WARNING`
- [ ] Use production database
- [ ] Use production vector database
- [ ] Set up monitoring/alerting
- [ ] Configure CORS for production domain
- [ ] Generate new `SECRET_KEY`: `openssl rand -hex 32`
- [ ] Enable HTTPS: `ENABLE_HTTPS=true`
- [ ] Set appropriate rate limits

## Performance Targets

These are the success criteria from the specification:

- Retrieval latency: <800ms (T030 critical path)
- End-to-end chat latency: <2s (T038-T039 critical path)
- Text selection detection: <500ms (T048 critical path)
- Chunk accuracy: >85% relevant results
- Hallucination prevention: >95% correct responses
- Concurrent users: Support 100+ simultaneous connections

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review `.env.setup-guide.md` for configuration help
3. Check FastAPI docs: http://localhost:8000/docs
4. Review logs for error details

---

**Last Updated**: 2025-12-10
**Status**: Phase 1 Setup Complete
