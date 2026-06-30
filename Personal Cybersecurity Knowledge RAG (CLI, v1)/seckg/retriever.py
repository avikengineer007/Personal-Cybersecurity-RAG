from typing import List, Optional
from seckg.config import settings
from seckg.models import RetrievalResult, Chunk
from seckg.vector_store import VectorStoreManager

class Retriever:
    """Handles query similarity search using ChromaDB and converts the results to internal models."""
    
    def __init__(self, manager: VectorStoreManager):
        self.manager = manager

    def retrieve(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieves the top-k relevant chunks matching the query string."""
        if k is None:
            k = settings.top_k

        # Set up retriever from index
        retriever = self.manager.index.as_retriever(similarity_top_k=k)
        
        # Retrieve nodes with similarity scores
        nodes_with_score = retriever.retrieve(query)
        
        results = []
        for node_with_score in nodes_with_score:
            node = node_with_score.node
            score = node_with_score.score if node_with_score.score is not None else 0.0
            
            # Build Chunk model from node content and metadata
            metadata = node.metadata
            chunk = Chunk(
                content=node.get_content(),
                source_file=metadata.get("source_file", "unknown"),
                heading_path=metadata.get("heading_path"),
                page_number=metadata.get("page_number"),
                chunk_id=node.node_id,
                last_modified=metadata.get("last_modified", 0.0)
            )
            results.append(RetrievalResult(chunk=chunk, score=score))

        return results
