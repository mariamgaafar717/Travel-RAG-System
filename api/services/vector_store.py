"""Vector store management service using FAISS and Sentence Transformers."""
import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from ..models.schemas import ChunkData

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Service for managing embeddings and vector similarity search."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        embeddings_path: Optional[Path] = None,
        faiss_index_path: Optional[Path] = None,
        chunks_path: Optional[Path] = None,
    ):
        """Initialize vector store service.
        
        Args:
            embedding_model: Name of the sentence transformer model to use
            embeddings_path: Path to save/load embeddings
            faiss_index_path: Path to save/load FAISS index
            chunks_path: Path to save/load chunk metadata
        """
        self.embedding_model_name = embedding_model
        self.embeddings_path = embeddings_path
        self.faiss_index_path = faiss_index_path
        self.chunks_path = chunks_path
        
        # Initialize components as None, will be loaded on demand
        self._model: Optional[SentenceTransformer] = None
        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunks: List[ChunkData] = []
        self._is_initialized = False
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy load the embedding model."""
        if self._model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._model = SentenceTransformer(self.embedding_model_name)
        return self._model

    def _load_place_links_from_dataset(self) -> dict:
        """Load place-to-links mapping from the dataset stored alongside the vector store."""
        if not self.chunks_path:
            return {}

        dataset_path = self.chunks_path.with_name("egypt_places.json")
        if not dataset_path.exists():
            return {}

        try:
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)

            place_links = {}
            for item in dataset:
                place = item.get("place")
                if not place:
                    continue

                links = item.get("links") or []
                if not links and item.get("source"):
                    links = [item["source"]]

                if links:
                    place_links[place] = list(dict.fromkeys(links))

            return place_links

        except Exception as exc:
            logger.warning(f"Could not load dataset links for normalization: {exc}")
            return {}
    
    def initialize_from_chunks(self, chunks: List[dict]) -> None:
        """Initialize vector store from a list of text chunks.
        
        Args:
            chunks: List of dictionaries containing chunk data with 'text' key
        """
        try:
            logger.info(f"Initializing vector store with {len(chunks)} chunks")
            
            # Extract texts and create ChunkData objects
            texts = [chunk.get("text", "") for chunk in chunks]
            self._chunks = [
                ChunkData(**chunk) for chunk in chunks
            ]
            
            if not texts:
                logger.warning("No text content found in chunks")
                self._is_initialized = False
                return
            
            # Generate embeddings
            logger.info("Generating embeddings...")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            
            # Create FAISS index
            logger.info("Creating FAISS index...")
            dimension = embeddings.shape[1]
            self._index = faiss.IndexFlatL2(dimension)
            self._index.add(embeddings.astype(np.float32))
            
            self._is_initialized = True
            logger.info("Vector store initialized successfully")
            
            # Save to disk if paths provided
            if self.faiss_index_path and self.chunks_path:
                self.save_to_disk()
        
        except Exception as e:
            logger.error(f"Error initializing vector store: {e}")
            self._is_initialized = False
            raise
    
    def load_from_disk(self) -> bool:
        """Load vector store from disk files.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            if not self.faiss_index_path or not self.chunks_path:
                logger.warning("No save paths configured")
                return False
            
            if not self.faiss_index_path.exists() or not self.chunks_path.exists():
                logger.info("Vector store files not found on disk")
                return False
            
            logger.info("Loading vector store from disk...")
            
            # Load FAISS index
            self._index = faiss.read_index(str(self.faiss_index_path))
            
            # Load chunks
            with open(self.chunks_path, "rb") as f:
                loaded_chunks = pickle.load(f)

            normalized_chunks: List[ChunkData] = []
            for chunk in loaded_chunks:
                if isinstance(chunk, ChunkData):
                    chunk_data = chunk.model_dump()
                elif hasattr(chunk, "model_dump"):
                    chunk_data = chunk.model_dump()
                else:
                    chunk_data = dict(chunk)

                links = chunk_data.get("links") or []
                if not links and chunk_data.get("source"):
                    links = [chunk_data["source"]]

                chunk_data["links"] = links
                chunk_data["source"] = chunk_data.get("source", links[0] if links else "")
                normalized_chunks.append(ChunkData(**chunk_data))

            place_links_map = self._load_place_links_from_dataset()
            if place_links_map:
                for chunk in normalized_chunks:
                    dataset_links = place_links_map.get(chunk.place, [])
                    merged_links = list(dict.fromkeys((chunk.links or []) + dataset_links))
                    if merged_links:
                        chunk.links = merged_links
                        if not chunk.source:
                            chunk.source = merged_links[0]

            self._chunks = normalized_chunks
            
            self._is_initialized = True
            logger.info(f"Loaded {len(self._chunks)} chunks from disk")
            return True
        
        except Exception as e:
            logger.error(f"Error loading vector store from disk: {e}")
            self._is_initialized = False
            return False
    
    def save_to_disk(self) -> bool:
        """Save vector store to disk.
        
        Returns:
            True if successfully saved, False otherwise
        """
        try:
            if not self.faiss_index_path or not self.chunks_path:
                logger.warning("No save paths configured")
                return False
            
            if not self._is_initialized or self._index is None:
                logger.warning("Vector store not initialized")
                return False
            
            # Create directory if it doesn't exist
            self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
            self.chunks_path.parent.mkdir(parents=True, exist_ok=True)
            
            logger.info("Saving vector store to disk...")
            
            # Save FAISS index
            faiss.write_index(self._index, str(self.faiss_index_path))
            
            # Save chunks
            with open(self.chunks_path, "wb") as f:
                pickle.dump(self._chunks, f)
            
            logger.info("Vector store saved successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error saving vector store to disk: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[ChunkData]:
        """Search for similar chunks given a query.
        
        Args:
            query: Query text
            top_k: Number of top results to return
            
        Returns:
            List of most similar ChunkData objects
        """
        if not self._is_initialized or self._index is None:
            logger.warning("Vector store not initialized")
            return []
        
        try:
            # Generate embedding for query
            query_embedding = self.model.encode(
                query,
                convert_to_numpy=True
            ).reshape(1, -1)
            
            # Search in FAISS index
            distances, indices = self._index.search(
                query_embedding.astype(np.float32),
                min(top_k, len(self._chunks))
            )
            
            # Convert distances to similarity scores (0-1)
            results = []
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self._chunks):
                    chunk = self._chunks[idx]
                    # Convert L2 distance to similarity score
                    # For L2 distance, we use exponential decay
                    similarity = np.exp(-distance / 2.0)
                    chunk_with_score = chunk.copy()
                    chunk_with_score.similarity_score = float(similarity)
                    results.append(chunk_with_score)
            
            return results
        
        except Exception as e:
            logger.error(f"Error during search: {e}")
            return []
    
    def is_initialized(self) -> bool:
        """Check if vector store is initialized."""
        return self._is_initialized
    
    def get_stats(self) -> dict:
        """Get vector store statistics."""
        stats = {
            "is_initialized": self._is_initialized,
            "total_chunks": len(self._chunks),
            "embedding_model": self.embedding_model_name,
        }
        
        if self._index is not None:
            stats["vector_dimension"] = self._index.d
            stats["index_size_mb"] = self._index.ntotal * self._index.d * 4 / (1024 * 1024)
        
        return stats
