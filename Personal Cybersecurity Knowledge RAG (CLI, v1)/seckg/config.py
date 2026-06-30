import os
import yaml
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

class Config:
    def __init__(self):
        # Default configuration values
        self.chroma_persist_dir = "./chroma_db"
        self.chroma_collection_name = "secknowledge"
        self.embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.top_k = 5
        self.pdf_chunk_size = 500
        self.pdf_chunk_overlap = 50
        self.watcher_debounce_seconds = 2.0
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.llm_model = "gemini-2.5-flash"

        # Load from config.yaml if it exists
        config_path = "config.yaml"
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                    if data:
                        chroma = data.get("chroma", {})
                        self.chroma_persist_dir = chroma.get("persist_dir", self.chroma_persist_dir)
                        self.chroma_collection_name = chroma.get("collection_name", self.chroma_collection_name)

                        embedding = data.get("embedding", {})
                        self.embedding_model_name = embedding.get("model_name", self.embedding_model_name)

                        retrieval = data.get("retrieval", {})
                        self.top_k = retrieval.get("top_k", self.top_k)

                        chunking = data.get("chunking", {})
                        pdf_cfg = chunking.get("pdf", {})
                        self.pdf_chunk_size = pdf_cfg.get("chunk_size", self.pdf_chunk_size)
                        self.pdf_chunk_overlap = pdf_cfg.get("chunk_overlap", self.pdf_chunk_overlap)
                        
                        watcher_cfg = chunking.get("watcher", {})
                        self.watcher_debounce_seconds = watcher_cfg.get("debounce_seconds", self.watcher_debounce_seconds)
            except Exception:
                # Fallback to default values if loading yaml fails
                pass

        # Normalize persist directory paths
        if not os.path.isabs(self.chroma_persist_dir):
            self.chroma_persist_dir = os.path.abspath(self.chroma_persist_dir)

settings = Config()
