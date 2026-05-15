"""FastAPI application for Travel RAG System."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config.settings import get_settings
from .models import (
    HealthResponse,
    ErrorResponse,
)
from .services import VectorStoreService, RAGService, LLMService
from .routes import router, set_dependencies

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


# Global service instances
vector_store: VectorStoreService = None
rag_service: RAGService = None
llm_service: LLMService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle - startup and shutdown."""
    global vector_store, rag_service, llm_service
    
    logger.info("Starting Travel RAG System API...")
    
    try:
        # Initialize data directory
        settings.ensure_data_dir()
        
        # Initialize Vector Store Service
        logger.info(f"Initializing Vector Store with model: {settings.EMBEDDING_MODEL}")
        vector_store = VectorStoreService(
            embedding_model=settings.EMBEDDING_MODEL,
            embeddings_path=settings.EMBEDDINGS_PATH,
            faiss_index_path=settings.FAISS_INDEX_PATH,
            chunks_path=settings.CHUNKS_PATH,
        )
        
        # Try to load from disk first
        if not vector_store.load_from_disk():
            logger.info("Vector store not found on disk. Will be initialized on first use.")
        
        # Initialize LLM Service
        logger.info("Initializing LLM Service...")
        llm_service = LLMService(
            model_name=settings.LLM_MODEL,
            use_gpu=True,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
        if llm_service.initialize():
            logger.info("✓ LLM Service initialized successfully")
        else:
            logger.warning("⚠ LLM Service initialization failed. Using basic answer generation.")
            llm_service = None
        
        # Initialize RAG Service
        logger.info("Initializing RAG Service...")
        rag_service = RAGService(
            vector_store=vector_store,
            llm_service=llm_service,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        set_dependencies(vector_store, rag_service)
        
        logger.info("✓ All services initialized successfully")
    
    except Exception as e:
        logger.error(f"✗ Failed to initialize services: {e}")
        raise
    
    yield
    
    logger.info("Shutting down Travel RAG System API...")
    try:
        if vector_store:
            vector_store.save_to_disk()
            logger.info("✓ Vector store saved to disk")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("✓ Shutdown complete")


# Create FastAPI application
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan,
    debug=settings.DEBUG,
)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)


# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "message": "Welcome to Travel RAG System API",
        "docs": "/docs",
        "redoc": "/redoc",
        "api_prefix": "/api",
    }


@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            error="Validation Error",
            detail=str(exc),
            status_code=400,
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            detail="An unexpected error occurred",
            status_code=500,
        ).model_dump(),
    )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Additional startup logic."""
    logger.info("API startup event triggered")


# Shutdown event (optional)
@app.on_event("shutdown")
async def shutdown_event():
    """Additional shutdown logic."""
    logger.info("API shutdown event triggered")


if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.HOST}:{settings.PORT}")
    uvicorn.run(
        app,
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.RELOAD,
        log_level="info",
    )

