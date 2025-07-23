import numpy as np
from typing import List, Dict, Optional, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, SearchRequest
from datetime import datetime
import os
from dataclasses import dataclass

@dataclass
class RAGConfig:
    """Configuration for RAG service"""
    qdrant_url: str = os.getenv("QDRANT_URL")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "dengcao/Qwen3-Embedding-0.6B:F16")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "1024"))
    max_retrieved_chunks: int = int(os.getenv("MAX_RETRIEVED_CHUNKS", "10"))
    mmr_diversity: float = float(os.getenv("MMR_DIVERSITY", "0.7"))
    hybrid_search_weight: float = float(os.getenv("HYBRID_SEARCH_WEIGHT", "0.7"))

class VectorRAGService:
    """Simplified RAG service that returns only vector search results after MMR"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.qdrant_client = QdrantClient(
            url=self.config.qdrant_url,
            api_key=self.config.qdrant_api_key
        )
    
    async def search_vectors(self, query_embedding: List[float], max_results: int = None, 
                           use_mmr: bool = True, diversity: float = None) -> Dict:
        """Main vector search method that returns processed results"""
        try:
            start_time = datetime.utcnow()
            
            # Retrieve relevant chunks
            if use_mmr:
                retrieved_chunks = await self._retrieve_with_mmr(
                    query_embedding,
                    max_results or self.config.max_retrieved_chunks,
                    diversity or self.config.mmr_diversity
                )
            else:
                retrieved_chunks = await self._retrieve_simple(
                    query_embedding,
                    max_results or self.config.max_retrieved_chunks
                )
            
            # Calculate metrics
            end_time = datetime.utcnow()
            latency = (end_time - start_time).total_seconds()
            
            return {
                "chunks": self._format_chunks(retrieved_chunks),
                "metadata": {
                    "retrieved_count": len(retrieved_chunks),
                    "latency_seconds": latency,
                    "timestamp": end_time.isoformat(),
                    "mmr_applied": use_mmr,
                    "diversity_factor": diversity or self.config.mmr_diversity
                }
            }
            
        except Exception as e:
            print(f"Vector search error: {e}")
            return {
                "chunks": [],
                "metadata": {
                    "retrieved_count": 0,
                    "latency_seconds": 0,
                    "timestamp": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
            }
    
    async def _retrieve_with_mmr(self, query_embedding: List[float], 
                                max_results: int, diversity: float) -> List[Dict]:
        """Retrieve chunks using Maximum Marginal Relevance (MMR)"""
        try:
            # Get more candidates than needed for MMR selection
            candidate_count = min(max_results * 3, 50)
            
            # Vector search
            vector_results = self.qdrant_client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=candidate_count,
                with_payload=True,
                with_vectors=True
            )
            
            # Convert results to our format
            candidates = []
            for result in vector_results:
                vector = result.vector if result.vector is not None else [0.0] * self.config.embedding_dim
                candidates.append({
                    "payload": result.payload,
                    "vector": vector,
                    "score": result.score
                })
            
            # Retrieve and add neighboring chunks
            expanded_candidates = await self._add_neighboring_chunks(candidates)
            
            # Apply MMR selection
            selected_chunks = self._apply_mmr(
                expanded_candidates,
                query_embedding,
                max_results,
                diversity
            )
            
            return selected_chunks
            
        except Exception as e:
            print(f"MMR retrieval error: {e}")
            # Fallback to simple retrieval
            return await self._retrieve_simple(query_embedding, max_results)
    
    async def _retrieve_simple(self, query_embedding: List[float], 
                              max_results: int) -> List[Dict]:
        """Simple vector similarity search"""
        try:
            results = self.qdrant_client.search(
                collection_name=self.config.collection_name,
                query_vector=query_embedding,
                limit=max_results,
                with_payload=True,
                with_vectors=True
            )
            
            # Format results
            chunks = []
            for result in results:
                vector = result.vector if result.vector is not None else [0.0] * self.config.embedding_dim
                chunks.append({
                    "payload": result.payload,
                    "vector": vector,
                    "score": result.score
                })
            
            # Add neighboring chunks
            expanded_chunks = await self._add_neighboring_chunks(chunks)
            
            # Deduplicate
            return self._deduplicate_chunks(expanded_chunks)
            
        except Exception as e:
            print(f"Simple retrieval error: {e}")
            return []
    
    async def _add_neighboring_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Add neighboring chunks to improve context"""
        all_chunks = []
        seen_chunk_ids = set()
        
        for chunk in chunks:
            chunk_id = self._get_chunk_id(chunk["payload"])
            if chunk_id not in seen_chunk_ids:
                all_chunks.append(chunk)
                seen_chunk_ids.add(chunk_id)
            
            # Try to get neighbors, but continue if it fails
            try:
                neighbors = await self._get_neighboring_chunks(chunk["payload"])
                for neighbor in neighbors:
                    neighbor_id = self._get_chunk_id(neighbor["payload"])
                    if neighbor_id not in seen_chunk_ids:
                        all_chunks.append(neighbor)
                        seen_chunk_ids.add(neighbor_id)
            except Exception as e:
                # Skip neighboring chunks if there's an index issue
                print(f"Skipping neighboring chunks due to: {e}")
                continue
        
        return all_chunks
    
    async def _get_neighboring_chunks(self, payload: Dict, num_neighbors: int = 3) -> List[Dict]:
        """Get neighboring chunks using sliding window with minimum chunk count"""
        try:
            document_id = payload.get("document_id")
            chunk_index = payload.get("chunk_index")
            total_chunks = payload.get("total_chunks")
            
            if not all([document_id, chunk_index is not None, total_chunks is not None]):
                return []
            
            # Calculate sliding window to get minimum number of chunks
            window_size = num_neighbors * 2 + 1  # e.g., 3*2+1 = 7 chunks
            
            # Try to center the window around current chunk
            half_window = num_neighbors
            start_index = chunk_index - half_window
            end_index = chunk_index + half_window
            
            # Adjust window if it goes beyond boundaries
            if start_index < 0:
                # Shift window right if we're too close to start
                adjustment = abs(start_index)
                start_index = 0
                end_index = min(total_chunks - 1, end_index + adjustment)
            elif end_index >= total_chunks:
                # Shift window left if we're too close to end
                adjustment = end_index - (total_chunks - 1)
                end_index = total_chunks - 1
                start_index = max(0, start_index - adjustment)
            
            # Ensure we don't exceed total chunks
            end_index = min(end_index, total_chunks - 1)
            
            print(f"DEBUG: Getting neighbors for chunk {chunk_index}/{total_chunks}, window: {start_index}-{end_index}")
            
            neighbors = []
            
            # Use document_id filter instead of source_file_name
            try:
                doc_filter = Filter(must=[
                    FieldCondition(key="document_id", match=MatchValue(value=document_id))
                ])
                
                offset = None
                while True:
                    scroll_results, next_offset = self.qdrant_client.scroll(
                        collection_name=self.config.collection_name,
                        scroll_filter=doc_filter,
                        limit=100,
                        offset=offset,
                        with_payload=True,
                        with_vectors=True
                    )
                    
                    if not scroll_results:
                        break
                    
                    for point in scroll_results:
                        current_index = point.payload.get("chunk_index")
                        if current_index is not None and start_index <= current_index <= end_index:
                            vector = point.vector if point.vector is not None else [0.0] * self.config.embedding_dim
                            neighbors.append({
                                "payload": point.payload,
                                "vector": vector,
                                "score": getattr(point, 'score', 0.0)
                            })
                    
                    if next_offset is None:
                        break
                    offset = next_offset
                    
            except Exception as e:
                print(f"Error with document_id filter: {e}")
                # If document_id filtering fails, try without filtering and match manually
                try:
                    print("Falling back to manual filtering...")
                    offset = None
                    while True:
                        scroll_results, next_offset = self.qdrant_client.scroll(
                            collection_name=self.config.collection_name,
                            limit=100,
                            offset=offset,
                            with_payload=True,
                            with_vectors=True
                        )
                        
                        if not scroll_results:
                            break
                        
                        for point in scroll_results:
                            # Manual filtering by document_id and chunk_index
                            if (point.payload.get("document_id") == document_id and
                                point.payload.get("chunk_index") is not None and
                                start_index <= point.payload.get("chunk_index") <= end_index):
                                
                                vector = point.vector if point.vector is not None else [0.0] * self.config.embedding_dim
                                neighbors.append({
                                    "payload": point.payload,
                                    "vector": vector,
                                    "score": getattr(point, 'score', 0.0)
                                })
                        
                        if next_offset is None or len(neighbors) >= window_size:
                            break
                        offset = next_offset
                        
                except Exception as e2:
                    print(f"Manual filtering also failed: {e2}")
                    return []
            
            # Sort by chunk index to maintain document order
            neighbors.sort(key=lambda x: x["payload"].get("chunk_index", 0))
            
            print(f"DEBUG: Found {len(neighbors)} neighboring chunks with indices: {[n['payload'].get('chunk_index') for n in neighbors]}")
            return neighbors
            
        except Exception as e:
            print(f"Error getting neighboring chunks: {e}")
            return []
    
    def _apply_mmr(self, candidates: List[Dict], query_embedding: List[float], 
                   max_results: int, diversity: float) -> List[Dict]:
        """Apply Maximum Marginal Relevance selection"""
        if not candidates:
            return []
        
        query_vec = np.array(query_embedding)
        candidate_embeddings = []
        
        for candidate in candidates:
            vector = candidate.get("vector", [0.0] * self.config.embedding_dim)
            candidate_embeddings.append(np.array(vector))
        
        selected_indices = []
        remaining_indices = list(range(len(candidates)))
        
        # Select first item (highest relevance)
        if remaining_indices:
            selected_indices.append(remaining_indices.pop(0))
        
        # Select remaining items using MMR
        for _ in range(min(max_results - 1, len(remaining_indices))):
            if not remaining_indices:
                break
            
            mmr_scores = []
            for idx in remaining_indices:
                # Relevance to query
                relevance = self._cosine_similarity(query_vec, candidate_embeddings[idx])
                
                # Maximum similarity to selected items
                max_similarity = 0
                for selected_idx in selected_indices:
                    similarity = self._cosine_similarity(
                        candidate_embeddings[idx],
                        candidate_embeddings[selected_idx]
                    )
                    max_similarity = max(max_similarity, similarity)
                
                # MMR score
                mmr_score = diversity * relevance - (1 - diversity) * max_similarity
                mmr_scores.append((idx, mmr_score))
            
            # Select best item
            best_idx = max(mmr_scores, key=lambda x: x[1])[0]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
        
        return [candidates[i] for i in selected_indices]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _deduplicate_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Remove duplicate chunks"""
        unique_chunks = {}
        for chunk in chunks:
            chunk_id = self._get_chunk_id(chunk["payload"])
            if chunk_id not in unique_chunks:
                unique_chunks[chunk_id] = chunk
        return list(unique_chunks.values())
    
    def _get_chunk_id(self, payload: Dict) -> str:
        """Generate unique chunk identifier"""
        chunk_id = payload.get("id")
        if chunk_id:
            return str(chunk_id)
        
        # Fallback to composite key
        source = payload.get("source_file_name", "unknown")
        page = payload.get("page_number", "0")
        index = payload.get("chunk_index", "0")
        return f"{source}-{page}-{index}"
    
    def _format_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Format chunks for API response"""
        formatted_chunks = []
        for i, chunk in enumerate(chunks):
            payload = chunk.get("payload", {})
            
            formatted_chunk = {
                "index": i + 1,
                "content": payload.get("content_chunk", ""),
                "source": {
                    "filename": payload.get("source_file_name", "Unknown"),
                    "page": payload.get("page_number"),
                    "chunk_index": payload.get("chunk_index"),
                    "section_header": payload.get("section_header", ""),
                    "sharepoint_url": payload.get("sharepoint_url", ""),
                    "chunk_type": payload.get("chunk_type", "text")
                },
                "score": chunk.get("score", 0.0),
                "metadata": {
                    "is_enhanced": payload.get("is_enhanced", False),
                    "word_count": len(payload.get("content_chunk", "").split()),
                    "chunk_id": self._get_chunk_id(payload)
                }
            }
            formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks
    
    def get_collection_info(self) -> Dict:
        """Get information about the document collection"""
        try:
            info = self.qdrant_client.get_collection(self.config.collection_name)
            
            # Get sample documents
            sample_results = self.qdrant_client.scroll(
                collection_name=self.config.collection_name,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            
            sample_docs = []
            for point in sample_results[0]:
                sample_docs.append({
                    "id": point.id,
                    "filename": point.payload.get("source_file_name", "Unknown"),
                    "chunk_type": point.payload.get("chunk_type", "text"),
                    "is_enhanced": point.payload.get("is_enhanced", False)
                })
            
            return {
                "collection_name": self.config.collection_name,
                "total_points": info.points_count,
                "vectors_count": info.vectors_count,
                "sample_documents": sample_docs,
                "config": {
                    "max_retrieved_chunks": self.config.max_retrieved_chunks,
                    "mmr_diversity": self.config.mmr_diversity,
                    "embedding_dim": self.config.embedding_dim
                }
            }
            
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"error": str(e)}


# FastAPI wrapper for deployment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn

app = FastAPI(title="Vector RAG Service", version="1.0.0")

# Initialize service
rag_service = VectorRAGService()

class SearchRequest(BaseModel):
    query_embedding: List[float]
    max_results: Optional[int] = None
    use_mmr: bool = True
    diversity: Optional[float] = None

class SearchResponse(BaseModel):
    chunks: List[Dict]
    metadata: Dict

@app.post("/search", response_model=SearchResponse)
async def search_vectors(request: SearchRequest):
    """Search for similar vectors using MMR"""
    try:
        result = await rag_service.search_vectors(
            query_embedding=request.query_embedding,
            max_results=request.max_results,
            use_mmr=request.use_mmr,
            diversity=request.diversity
        )
        return SearchResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vector-rag"}

@app.get("/collection/info")
async def get_collection_info():
    """Get collection information"""
    return rag_service.get_collection_info()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
