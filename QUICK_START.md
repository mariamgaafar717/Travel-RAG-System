# Travel RAG System - Quick Start Guide

## 📋 Prerequisites

- Python 3.9+ (for CPU/pip setup)
- Miniconda/Conda (for GPU setup - recommended)
- Docker & Docker Compose (optional, for containerized deployment)
- 4GB+ RAM (8GB+ recommended for LLM)
- GPU with CUDA support (optional but recommended)

## ⚡ Quick Start (5 minutes)

### Option A: GPU Setup (Recommended)

```bash
# Prerequisites: Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

cd Travel-RAG-System
conda env create -f environment-gpu.yml
conda activate travel-rag-gpu
python init_vector_store.py
python run_api.py
```

### Option B: CPU Setup (Pip)

```bash
cd Travel-RAG-System
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python init_vector_store.py
python run_api.py
```

Visit: http://localhost:8000/docs for interactive documentation

## 🐳 Docker Quick Start

```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

API will be at: http://localhost:8000

## 📡 API Usage Examples

### Test Health

```bash
curl http://localhost:8000/api/health
```

### Initialize Vector Store

```bash
curl -X POST http://localhost:8000/api/initialize \
  -H "Content-Type: application/json" \
  -d '{"data_path": "data/egypt_places.json", "rebuild": false}'
```

### Search Documents

```bash
curl -X GET "http://localhost:8000/api/retrieve?query=Cairo+pyramids&top_k=3"
```

### Full RAG Query

```bash
curl -X POST http://localhost:8000/api/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main attractions in Luxor?",
    "top_k": 5
  }'
```

## 🔧 Development Workflow

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run with auto-reload
make dev-reload

# Code formatting
make format

# Linting
make lint

# Run tests
make test
```

## 📁 Project Structure

```
api/                 # Main FastAPI application
├── config/          # Configuration management
├── models/          # Request/response schemas
├── services/        # Business logic (Vector Store, RAG, LLM)
├── routes/          # API endpoints
└── main.py          # App entry point

data/               # Data storage directory
├── egypt_places.json        # Dataset
├── chunks.pkl               # Prepared chunks
└── faiss_index.bin         # FAISS index

Dockerfile          # Container definition
docker-compose.yml  # Service orchestration
requirements.txt    # Python dependencies
```

## 🎯 Configuration

Edit `.env` file to customize:

```env
# API
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Models
EMBEDDING_MODEL=all-MiniLM-L6-v2
LLM_MODEL=Qwen/Qwen2.5-1.5B-Instruct

# Parameters
CHUNK_SIZE=100
CHUNK_OVERLAP=20
LLM_MAX_TOKENS=256
USE_GPU=True
```

## 🧪 Testing Your Setup

1. **Check API is running:**

   ```bash
   curl http://localhost:8000/api/health
   ```

2. **View API docs:**
   Open http://localhost:8000/docs in browser

3. **Initialize vector store:**

   ```bash
   python init_vector_store.py
   ```

4. **Make a query:**
   ```bash
   curl -X GET "http://localhost:8000/api/retrieve?query=Egypt&top_k=3"
   ```

## 📊 Available Endpoints

| Method   | Endpoint          | Purpose                 |
| -------- | ----------------- | ----------------------- |
| GET      | `/api/health`     | Health check            |
| GET      | `/api/stats`      | Vector store stats      |
| POST     | `/api/initialize` | Initialize vector store |
| GET/POST | `/api/retrieve`   | Retrieve similar docs   |
| GET/POST | `/api/rag`        | Full RAG pipeline       |
| GET      | `/api/search`     | Quick search            |
| GET      | `/docs`           | Swagger UI              |
| GET      | `/redoc`          | ReDoc documentation     |

## 🚀 Production Deployment

```bash
# Using Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f api

# Scale if needed
docker-compose up -d --scale api=3
```

## 🐛 Troubleshooting

### Vector Store Not Initializing

```bash
# Verify data file exists
ls -la data/egypt_places.json

# Rebuild from scratch
python init_vector_store.py --rebuild
```

### LLM Not Loading

```bash
# Check torch installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

# For GPU setup: Ensure you're using the conda environment
conda activate travel-rag-gpu

# For pip setup: You have CPU-only PyTorch (FAISS will also be CPU-only)
```

### FAISS Still CPU

```bash
# FAISS GPU is only available via conda with faiss-gpu package
# Use Option A (GPU Setup) from Quick Start:
conda env create -f environment-gpu.yml
conda activate travel-rag-gpu
```

### Out of Memory

```bash
# Reduce model or use CPU
export USE_GPU=False
python run_api.py

# Or switch to smaller embedding model in .env:
EMBEDDING_MODEL=all-MiniLM-L6-v2  # Already the smallest
```

## 📚 Documentation

- `API_README.md` - Full API documentation
- `IMPLEMENTATION_SUMMARY.md` - Architecture overview
- Inline code comments throughout

## 🆘 Getting Help

1. Check `API_README.md` for detailed documentation
2. Review code comments and docstrings
3. Check Docker logs: `docker-compose logs -f api`
4. Check application logs in console

## ⏭️ Next Steps

1. **Customize your data**: Replace `egypt_places.json` with your dataset
2. **Tune parameters**: Adjust chunk size, embedding model, etc. in `.env`
3. **Add authentication**: Implement API key validation in routes
4. **Deploy**: Use provided Docker setup for production deployment
5. **Monitor**: Set up logging and monitoring for production

## 📝 Notes

- The system gracefully degrades if LLM fails to load
- Vector store is persisted to disk for quick loading
- All settings can be configured via environment variables
- API documentation is auto-generated at `/docs`
