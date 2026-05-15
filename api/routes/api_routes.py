"""API routes for the Travel RAG System."""
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Query

from ..models.schemas import (
    QueryRequest,
    RetrievalResponse,
    RAGResponse,
    VectorStoreStats,
    InitializeRequest,
    ChunkData,
)
from ..services.rag_service import RAGService
from ..services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["rag"])


_vector_store: Optional[VectorStoreService] = None
_rag_service: Optional[RAGService] = None


def set_dependencies(vector_store: VectorStoreService, rag_service: RAGService) -> None:
    """Set the service dependencies for routes.
    
    Args:
        vector_store: VectorStoreService instance
        rag_service: RAGService instance
    """
    global _vector_store, _rag_service
    _vector_store = vector_store
    _rag_service = rag_service


def get_vector_store() -> VectorStoreService:
    """Get vector store service."""
    if _vector_store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector store not initialized"
        )
    return _vector_store


def get_rag_service() -> RAGService:
    """Get RAG service."""
    if _rag_service is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="RAG service not initialized"
        )
    return _rag_service


# ==================== Health & Status Endpoints ====================

@router.get(
    "/health",
    summary="Health Check",
    description="Check if the API is running and services are initialized"
)
async def health_check():
    """Check API health and service status."""
    try:
        vector_store = get_vector_store()
        return {
            "status": "healthy",
            "vector_store_ready": vector_store.is_initialized(),
        }
    except HTTPException:
        return {
            "status": "healthy_but_services_initializing",
            "vector_store_ready": False,
        }


@router.get(
    "/stats",
    response_model=VectorStoreStats,
    summary="Get Vector Store Statistics",
    description="Retrieve statistics about the current vector store"
)
async def get_stats():
    """Get vector store statistics."""
    vector_store = get_vector_store()
    stats = vector_store.get_stats()
    
    return VectorStoreStats(
        total_documents=len(vector_store._chunks),
        total_chunks=len(vector_store._chunks),
        embedding_model=stats["embedding_model"],
        vector_dimension=stats.get("vector_dimension", 0),
        index_size_mb=stats.get("index_size_mb"),
    )


# ==================== Initialization Endpoints ====================

@router.post(
    "/initialize",
    summary="Initialize Vector Store",
    description="Initialize the vector store from a dataset file"
)
async def initialize_vector_store(request: InitializeRequest):
    """Initialize vector store from dataset."""
    try:
        vector_store = get_vector_store()
        rag_service = get_rag_service()
        
        # Try to load from disk first (if not rebuilding)
        if not request.rebuild and vector_store.load_from_disk():
            return {
                "status": "success",
                "message": "Vector store loaded from disk",
                "stats": vector_store.get_stats()
            }
        
        # Load dataset
        from ..config.settings import get_settings
        settings = get_settings()
        
        data_path = request.data_path or str(settings.DATASET_PATH)
        from pathlib import Path
        dataset_path = Path(data_path)
        
        dataset = rag_service.load_dataset(dataset_path)
        if not dataset:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Could not load dataset from {data_path}"
            )
        
        # Prepare chunks
        chunks = rag_service.prepare_chunks(dataset)
        
        # Initialize vector store
        vector_store.initialize_from_chunks(chunks)
        
        return {
            "status": "success",
            "message": "Vector store initialized successfully",
            "stats": vector_store.get_stats()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initialize vector store: {str(e)}"
        )


# ==================== Retrieval Endpoints ====================

@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve Similar Documents",
    description="Retrieve documents similar to the query without LLM generation"
)
async def retrieve_documents(request: QueryRequest):
    """Retrieve similar documents from vector store."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    try:
        vector_store = get_vector_store()
        results = vector_store.search(request.query, request.top_k)
        
        return RetrievalResponse(
            query=request.query,
            results=results,
            num_results=len(results),
        )
    
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )


@router.get(
    "/retrieve",
    response_model=RetrievalResponse,
    summary="Retrieve Similar Documents (Query Parameter)",
    description="Retrieve documents similar to the query (using query parameters)"
)
async def retrieve_documents_get(
    query: str = Query(..., description="User query", min_length=1),
    top_k: int = Query(5, description="Number of top results", ge=1, le=20)
):
    """Retrieve similar documents (GET version)."""
    request = QueryRequest(query=query, top_k=top_k)
    return await retrieve_documents(request)


# ==================== RAG Endpoints ====================

@router.post(
    "/rag",
    response_model=RAGResponse,
    summary="RAG Query",
    description="Perform Retrieval-Augmented Generation query"
)
async def rag_query(request: QueryRequest):
    """Perform RAG query with retrieval and answer generation."""
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )
    
    try:
        rag_service = get_rag_service()
        response = rag_service.rag_retrieve(request.query, request.top_k)
        return response
    
    except Exception as e:
        logger.error(f"Error during RAG query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@router.get(
    "/rag",
    response_model=RAGResponse,
    summary="RAG Query (Query Parameter)",
    description="Perform Retrieval-Augmented Generation query (using query parameters)"
)
async def rag_query_get(
    query: str = Query(..., description="User query", min_length=1),
    top_k: int = Query(5, description="Number of top results", ge=1, le=20)
):
    """Perform RAG query (GET version)."""
    request = QueryRequest(query=query, top_k=top_k)
    return await rag_query(request)


# ==================== Info Endpoints ====================

@router.get(
    "/info",
    summary="API Information",
    description="Get information about the API and available endpoints"
)
async def api_info():
    """Get API information."""
    from ..config.settings import get_settings
    settings = get_settings()
    
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "description": settings.API_DESCRIPTION,
        "endpoints": {
            "health": "/api/health",
            "stats": "/api/stats",
            "initialize": "/api/initialize",
            "retrieve": "/api/retrieve",
            "rag": "/api/rag",
            "info": "/api/info",
        }
    }


# ==================== Search Endpoints ====================

@router.get(
    "/search",
    response_model=RetrievalResponse,
    summary="Search",
    description="Search for destinations and attractions (alias for /retrieve)"
)
async def search(
    q: str = Query(..., description="Search query", min_length=1),
    limit: int = Query(5, description="Number of results", ge=1, le=20)
):
    """Search for destinations and attractions."""
    request = QueryRequest(query=q, top_k=limit)
    return await retrieve_documents(request)
