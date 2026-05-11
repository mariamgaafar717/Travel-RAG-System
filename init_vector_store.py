#!/usr/bin/env python
"""Initialization script for Travel RAG System."""

import sys
import logging
from pathlib import Path

# Add parent directory to path to import from api module
sys.path.insert(0, str(Path(__file__).parent))

from api.config.settings import get_settings
from api.services import VectorStoreService, RAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_vector_store(rebuild: bool = False) -> bool:
    """Initialize the vector store from dataset.
    
    Args:
        rebuild: Whether to rebuild from scratch
        
    Returns:
        True if successful, False otherwise
    """
    try:
        settings = get_settings()
        
        logger.info("=" * 50)
        logger.info("Travel RAG System - Vector Store Initialization")
        logger.info("=" * 50)
        
        # Ensure data directory exists
        settings.ensure_data_dir()
        logger.info(f"✓ Data directory: {settings.DATA_DIR}")
        
        # Check if dataset exists
        if not settings.DATASET_PATH.exists():
            logger.error(f"✗ Dataset not found at: {settings.DATASET_PATH}")
            logger.error("Please ensure egypt_places.json exists in the data directory")
            return False
        
        logger.info(f"✓ Dataset found: {settings.DATASET_PATH}")
        
        # Initialize Vector Store
        logger.info("\nInitializing Vector Store...")
        vector_store = VectorStoreService(
            embedding_model=settings.EMBEDDING_MODEL,
            use_gpu=settings.USE_GPU,
            embeddings_path=settings.EMBEDDINGS_PATH,
            faiss_index_path=settings.FAISS_INDEX_PATH,
            chunks_path=settings.CHUNKS_PATH,
        )
        
        # Try to load from disk first (unless rebuilding)
        if not rebuild and vector_store.load_from_disk():
            logger.info("✓ Vector store loaded from disk")
            stats = vector_store.get_stats()
            logger.info(f"  - Total chunks: {stats['total_chunks']}")
            logger.info(f"  - Model: {stats['embedding_model']}")
            logger.info(f"  - Vector dimension: {stats.get('vector_dimension', 'N/A')}")
            return True
        
        if rebuild:
            logger.info("Rebuilding vector store from scratch...")
        
        # Initialize RAG Service for data preparation
        logger.info("Preparing chunks from dataset...")
        rag_service = RAGService(
            vector_store=vector_store,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )
        
        # Load dataset
        dataset = rag_service.load_dataset(settings.DATASET_PATH)
        if not dataset:
            logger.error("✗ Failed to load dataset")
            return False
        
        # Prepare chunks
        chunks = rag_service.prepare_chunks(dataset)
        if not chunks:
            logger.error("✗ Failed to prepare chunks")
            return False
        
        logger.info(f"✓ Prepared {len(chunks)} chunks")
        
        # Initialize vector store
        logger.info("Generating embeddings and creating FAISS index...")
        vector_store.initialize_from_chunks(chunks)
        
        # Save to disk
        logger.info("Saving vector store to disk...")
        vector_store.save_to_disk()
        
        # Display stats
        stats = vector_store.get_stats()
        logger.info("\n" + "=" * 50)
        logger.info("✓ Vector Store Initialized Successfully!")
        logger.info("=" * 50)
        logger.info(f"  - Total chunks: {stats['total_chunks']}")
        logger.info(f"  - Model: {stats['embedding_model']}")
        logger.info(f"  - Vector dimension: {stats.get('vector_dimension', 'N/A')}")
        logger.info(f"  - Index size: {stats.get('index_size_mb', 'N/A')} MB")
        logger.info(f"  - FAISS index: {settings.FAISS_INDEX_PATH}")
        logger.info(f"  - Chunks data: {settings.CHUNKS_PATH}")
        logger.info("=" * 50)
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Error during initialization: {e}", exc_info=True)
        return False


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Initialize Travel RAG System vector store"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild vector store from scratch"
    )
    
    args = parser.parse_args()
    
    success = initialize_vector_store(rebuild=args.rebuild)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
