from typing import List
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from seckg.config import settings
from seckg.models import Chunk

class VectorStoreManager:
    """Manages the persistent ChromaDB collection and LlamaIndex integration."""
    
    def __init__(self):
        # Initialize persistent ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        self.collection_name = settings.chroma_collection_name
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        
        # Setup LlamaIndex ChromaVectorStore wrapper
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        
        # Configure local Hugging Face embedding model globally
        self.embed_model = HuggingFaceEmbedding(model_name=settings.embedding_model_name)
        Settings.embed_model = self.embed_model
        
        # Initialize or load VectorStoreIndex from existing store
        try:
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                embed_model=self.embed_model
            )
        except Exception:
            # Fallback to creating a new index using the persistent store
            self.index = VectorStoreIndex(
                [],
                storage_context=self.storage_context,
                embed_model=self.embed_model
            )

    def reset_store(self):
        """Clears the collection entirely by deleting and recreating it."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except ValueError:
            pass
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        self.vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        self.storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self.index = VectorStoreIndex(
            [],
            storage_context=self.storage_context,
            embed_model=self.embed_model
        )

    def add_chunks(self, chunks: List[Chunk]):
        """Converts our Chunk models into LlamaIndex TextNodes and inserts them into the index."""
        if not chunks:
            return
            
        nodes = []
        for chunk in chunks:
            metadata = {
                "source_file": chunk.source_file,
                "last_modified": chunk.last_modified
            }
            if chunk.heading_path:
                metadata["heading_path"] = chunk.heading_path
            if chunk.page_number is not None:
                metadata["page_number"] = chunk.page_number
                
            node = TextNode(
                text=chunk.content,
                id_=chunk.chunk_id,
                metadata=metadata
            )
            nodes.append(node)
            
        self.index.insert_nodes(nodes)

    def delete_by_file(self, relative_path: str):
        """Deletes all chunks associated with a specific relative source file path from the Chroma collection."""
        posix_rel_path = relative_path.replace("\\", "/")
        # Delete directly from ChromaDB using metadata filter
        self.chroma_collection.delete(where={"source_file": posix_rel_path})
