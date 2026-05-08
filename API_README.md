# Travel RAG System - API Documentation

## Overview

This is a production-ready FastAPI implementation of the Travel RAG System with clean MVC architecture, vector-based retrieval, and LLM-based answer generation.

## Project Architecture

```
api/
├── main.py                 # FastAPI application entry point
├── config/
│   ├── settings.py        # Configuration management
│   └── __init__.py
├── models/
│   ├── schemas.py         # Pydantic models for request/response
│   └── __init__.py
├── services/
│   ├── vector_store.py    # Vector store management with FAISS
│   ├── rag_service.py     # RAG logic and orchestration
│   ├── llm_service.py     # LLM integration for answer generation
│   └── __init__.py
└── routes/
    ├── api_routes.py      # API endpoints
    └── __init__.py
```

## Key Features

✅ **MVC Pattern**: Clean separation of concerns with Models, Views (Routes), and Services
✅ **Vector Store**: FAISS-based semantic search with sentence transformers
✅ **LLM Integration**: Qwen LLM for context-aware answer generation
✅ **RESTful API**: Comprehensive endpoints for retrieval and RAG queries
✅ **Error Handling**: Proper exception handling with detailed error responses
✅ **Logging**: Structured logging throughout the application
✅ **Docker Support**: Multi-stage Dockerfile and docker-compose for easy deployment
✅ **Configuration Management**: Environment-based settings with pydantic

## Installation

### Local Setup

1. **Clone the repository:**

```bash
cd Travel-RAG-System
```

2. **Create a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**

```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the API:**

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Docker Setup

1. **Build the Docker image:**

```bash
docker build -t travel-rag-api .
```

2. **Run the container:**

```bash
docker run -p 8000:8000 -v $(pwd)/data:/app/data travel-rag-api
```

### Docker Compose Setup

1. **Start the services:**

```bash
docker-compose up -d
```

2. **Check logs:**

```bash
docker-compose logs -f api
```

3. **Stop the services:**

```bash
docker-compose down
```

## API Endpoints

### Health & Status

- **GET** `/api/health` - Check API health status
- **GET** `/api/stats` - Get vector store statistics
- **GET** `/api/info` - Get API information

### Initialization

- **POST** `/api/initialize` - Initialize vector store from dataset
  ```json
  {
    "data_path": "/path/to/dataset.json",
    "rebuild": false
  }
  ```

### Retrieval

- **POST** `/api/retrieve` - Retrieve similar documents

  ```json
  {
    "query": "Best places to visit in Egypt",
    "top_k": 5
  }
  ```

- **GET** `/api/retrieve?query=Best+places&top_k=5` - Retrieve (GET version)

### RAG Query

- **POST** `/api/rag` - Full RAG pipeline (retrieval + generation)

  ```json
  {
    "query": "What are the main attractions in Luxor?",
    "top_k": 5
  }
  ```

- **GET** `/api/rag?query=Luxor+attractions&top_k=5` - RAG (GET version)

### Search

- **GET** `/api/search?q=Cairo&limit=5` - Search for destinations

## Usage Examples

### Initialize Vector Store

```bash
curl -X POST http://localhost:8000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{
    "data_path": "data/egypt_places.json",
    "rebuild": false
  }'
```

### Query RAG Pipeline

```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the best tourist attractions in Cairo?",
    "top_k": 5
  }'
```

### Retrieve Documents

```bash
curl -X GET "http://localhost:8000/api/retrieve?query=pyramids&top_k=3"
```

### Check Health

```bash
curl http://localhost:8000/api/health
```

## Response Format

### RAG Response

```json
{
  "query": "What are the main attractions in Luxor?",
  "context": [
    {
      "place": "Luxor",
      "city": "Luxor",
      "type": "tourism",
      "chunk_id": 0,
      "text": "Luxor is home to the Valley of the Kings...",
      "source": "https://en.wikipedia.org/wiki/Luxor",
      "similarity_score": 0.95
    }
  ],
  "answer": "Luxor's main attractions include the Valley of the Kings, Karnak Temple Complex, and Luxor Temple...",
  "sources": ["https://en.wikipedia.org/wiki/Luxor"]
}
```

### Retrieval Response

```json
{
  "query": "Cairo monuments",
  "results": [
    {
      "place": "Giza",
      "city": "Cairo",
      "type": "tourism",
      "chunk_id": 0,
      "text": "The Great Pyramids of Giza are...",
      "source": "https://en.wikipedia.org/wiki/Giza",
      "similarity_score": 0.92
    }
  ],
  "num_results": 1
}
```

## Configuration

Edit `.env` file to configure:

```env
# API Settings
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Embedding Model
EMBEDDING_MODEL=all-MiniLM-L6-v2

# LLM Settings
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=256
USE_GPU=True

# Chunking
CHUNK_SIZE=100
CHUNK_OVERLAP=20
```

## Interactive API Documentation

Once running, access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development

### Add New Endpoints

1. Create handler in `api/routes/api_routes.py`
2. Define request/response models in `api/models/schemas.py`
3. Implement logic in appropriate service

### Add New Services

1. Create service in `api/services/`
2. Export in `api/services/__init__.py`
3. Inject into main.py lifespan

### Running Tests

```bash
pytest tests/ -v
```

## Performance Tips

1. **GPU Support**: Set `USE_GPU=True` for faster inference
2. **Batch Processing**: Prepare embeddings in batches
3. **Caching**: Enable `ENABLE_CACHE=True` to cache results
4. **Model Size**: Use smaller models for faster response times

## Troubleshooting

### Vector Store Not Initializing

```bash
# Check if data file exists
ls -la data/egypt_places.json

# Initialize manually
curl -X POST http://localhost:8000/api/initialize
```

### LLM Not Loading

```bash
# Check if torch/transformers are installed
pip install torch transformers

# Enable fallback to basic generation
# The system will use basic answer generation if LLM fails
```

### Out of Memory

```bash
# Use CPU instead of GPU
USE_GPU=False python -m uvicorn api.main:app

# Or use a smaller LLM model
LLM_MODEL=distilgpt2
```

## Production Deployment

### Using Docker

```bash
docker-compose -f docker-compose.yml up -d
```

### Health Check

```bash
docker-compose ps
```

### View Logs

```bash
docker-compose logs -f api
```

### Update Configuration

Edit `.env` and restart:

```bash
docker-compose restart api
```

## API Rate Limiting

For production, add rate limiting using:

```bash
pip install slowapi
```

## Contributing

See main README.md for contribution guidelines.

## License

MIT License - See LICENSE file for details
