"""RAG (Retrieval-Augmented Generation) service."""
import json
import logging
import re
from pathlib import Path
from typing import List, Optional

from ..models.schemas import ChunkData, RAGResponse
from .vector_store import VectorStoreService
from .llm_service import LLMService

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation."""
    
    def __init__(
        self,
        vector_store: VectorStoreService,
        llm_service: Optional[LLMService] = None,
        chunk_size: int = 100,
        chunk_overlap: int = 20,
    ):
        """Initialize RAG service.
        
        Args:
            vector_store: VectorStoreService instance for retrieval
            llm_service: Optional LLMService instance for generation
            chunk_size: Size of text chunks in words
            chunk_overlap: Overlap between consecutive chunks
        """
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
        """Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Number of words per chunk
            overlap: Number of overlapping words between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [""]
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean text by removing citations and extra whitespace.
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove Wikipedia citations like [1], [2], etc.
        text = re.sub(r"\[\d+\]", "", text)
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    def load_dataset(self, dataset_path: Path) -> List[dict]:
        """Load and parse dataset from JSON file.
        
        Args:
            dataset_path: Path to JSON dataset file
            
        Returns:
            List of dataset items
        """
        try:
            logger.info(f"Loading dataset from {dataset_path}")
            with open(dataset_path, "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} items from dataset")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def prepare_chunks(
        self,
        dataset: List[dict],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> List[dict]:
        """Prepare text chunks from dataset items.
        
        Args:
            dataset: List of data items with 'text' field
            chunk_size: Words per chunk (uses default if None)
            chunk_overlap: Words overlap (uses default if None)
            
        Returns:
            List of chunk dictionaries with metadata
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        chunked_data = []
        
        for item in dataset:
            text = self.clean_text(item.get("text", ""))
            chunks = self.chunk_text(text, chunk_size, chunk_overlap)
            item_links = item.get("links") or []
            if not item_links and item.get("source"):
                item_links = [item.get("source")]
            
            for idx, chunk in enumerate(chunks):
                chunked_data.append({
                    "place": item.get("place", "Unknown"),
                    "city": item.get("city", "Unknown"),
                    "type": item.get("type", "tourism"),
                    "chunk_id": idx,
                    "text": chunk,
                    "links": item_links,
                    "source": item_links[0] if item_links else item.get("source", ""),
                })
        
        logger.info(f"Prepared {len(chunked_data)} chunks from dataset")
        return chunked_data
    
    def retrieve(self, query: str, top_k: int = 5) -> List[ChunkData]:
        """Retrieve relevant chunks for a query.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        logger.info(f"Retrieving top {top_k} chunks for query: {query}")
        results = self.vector_store.search(query, top_k)
        logger.info(f"Retrieved {len(results)} chunks")
        return results
    
    def generate_context_string(self, chunks: List[ChunkData]) -> str:
        """Generate context string from retrieved chunks.
        
        Args:
            chunks: List of retrieved chunks
            
        Returns:
            Formatted context string
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            chunk_links = getattr(chunk, "links", []) or []
            primary_link = getattr(chunk, "source", "")
            if not primary_link and chunk_links:
                primary_link = chunk_links[0]
            context_parts.append(
                f"[Document {i} - {chunk.place}]\n"
                f"Source: {primary_link}\n"
                f"Content: {chunk.text}\n"
            )
        return "\n".join(context_parts)
    
    def generate_answer_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        """Generate a prompt for LLM-based answer generation.
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""You are a knowledgeable Egypt travel guide.
Answer the question using ONLY the documents below.
Use ONLY the context. Do not guess.
If the answer is not in the documents, say: 'I don't have enough information about that.'

Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def rag_retrieve(self, query: str, top_k: int = 5) -> RAGResponse:
        """Perform RAG retrieval and generation.
        
        Args:
            query: User query
            top_k: Number of results to retrieve
            
        Returns:
            RAG response with context and generated answer
        """
        context_chunks = self.retrieve(query, top_k)
        
        # Extract unique links from all chunks
        source_set = set()
        for chunk in context_chunks:
            chunk_links = getattr(chunk, "links", []) or []
            chunk_source = getattr(chunk, "source", "")
            if chunk_links:
                source_set.update(link for link in chunk_links if link)
            elif chunk_source:
                source_set.add(chunk_source)
        sources = list(source_set)
        
        # Generate context string
        context_str = self.generate_context_string(context_chunks)
        
        # Generate answer using LLM if available, otherwise use basic generation
        if self.llm_service and self.llm_service.is_initialized():
            prompt = self.generate_answer_prompt(query, context_str)
            answer = self.llm_service.generate(prompt)
        else:
            answer = self._generate_basic_answer(query, context_chunks)
        
        return RAGResponse(
            query=query,
            context=context_chunks,
            answer=answer,
            sources=sources,
        )
    
    def _generate_basic_answer(self, query: str, context: List[ChunkData]) -> str:
        """Generate a basic answer from retrieved context.
        
        This is a simple answer generation without calling an LLM.
        For production, integrate with actual LLM APIs.
        
        Args:
            query: User query
            context: Retrieved context chunks
            
        Returns:
            Generated answer
        """
        if not context:
            return "I couldn't find information about that topic in the knowledge base."
        
        # Extract place information from context
        places = set()
        for chunk in context:
            if chunk.place and chunk.place != "Unknown":
                places.add(chunk.place)
        
        places_str = ", ".join(sorted(places))
        
        # Simple answer template
        answer = f"Based on information about {places_str}, "
        
        # Add similarity-based response
        avg_similarity = sum(
            chunk.similarity_score or 0 for chunk in context
        ) / len(context) if context else 0
        
        if avg_similarity > 0.7:
            answer += "I found highly relevant information. "
        elif avg_similarity > 0.3:
            answer += "I found some relevant information. "
        else:
            answer += "Here's what I found related to your query: "
        
        # Add context summary
        if context:
            answer += f"The most relevant destination is {context[0].place}. "
            answer += "For more details, please check the retrieved context documents below."
        
        return answer
