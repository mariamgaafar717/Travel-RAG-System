.PHONY: help install dev prod lint format test clean docker-build docker-up docker-down docs init-vector-store

help:
	@echo "Travel RAG System - Development Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install dependencies"
	@echo "  make init-vector-store - Initialize vector store"
	@echo ""
	@echo "Development:"
	@echo "  make dev            - Run API in development mode"
	@echo "  make dev-reload     - Run API with auto-reload"
	@echo ""
	@echo "Production:"
	@echo "  make prod           - Run API in production mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run linting checks"
	@echo "  make format         - Format code with black"
	@echo "  make test           - Run tests"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build   - Build Docker image"
	@echo "  make docker-up      - Start services with docker-compose"
	@echo "  make docker-down    - Stop services with docker-compose"
	@echo "  make docker-logs    - View docker logs"
	@echo ""
	@echo "Utility:"
	@echo "  make clean          - Clean cache and temporary files"
	@echo "  make docs           - View API documentation"

install:
	pip install -r requirements.txt

dev:
	python run_api.py --host 0.0.0.0 --port 8000

dev-reload:
	python run_api.py --host 0.0.0.0 --port 8000 --reload

prod:
	python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --log-level info

lint:
	@echo "Running linting checks..."
	flake8 api/ src/ --max-line-length=100 || true
	pylint api/ src/ || true

format:
	@echo "Formatting code..."
	black api/ src/ --line-length=100 || true
	isort api/ src/ || true

test:
	pytest tests/ -v --tb=short

clean:
	@echo "Cleaning cache and temporary files..."
	find . -type d -name __pycache__ -exec rm -rf {} + || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + || true
	rm -rf build/ dist/ *.egg-info

docker-build:
	@echo "Building Docker image..."
	docker build -t travel-rag-api:latest .

docker-up:
	@echo "Starting services with docker-compose..."
	docker-compose up -d

docker-down:
	@echo "Stopping services..."
	docker-compose down

docker-logs:
	docker-compose logs -f api

init-vector-store:
	@echo "Initializing vector store..."
	python init_vector_store.py

init-vector-store-rebuild:
	@echo "Rebuilding vector store..."
	python init_vector_store.py --rebuild

docs:
	@echo "Open API documentation at:"
	@echo "http://localhost:8000/docs"

requirements-dev:
	pip install -r requirements-dev.txt || true
