import os
import tempfile
import yaml
# pyrefly: ignore [missing-import]
import pytest
from seckg.config import Config

def test_default_config():
    """Verify that config defaults are set correctly when no config.yaml exists."""
    # We create a temporary directory and instantiate Config inside it to ensure no config.yaml is found
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            cfg = Config()
            assert cfg.chroma_collection_name == "secknowledge"
            assert cfg.top_k == 5
            assert cfg.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        finally:
            os.chdir(orig_cwd)

def test_custom_config():
    """Verify that config overrides load correctly from a custom config.yaml."""
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            custom_data = {
                "chroma": {
                    "persist_dir": "./custom_chroma",
                    "collection_name": "custom_collection"
                },
                "retrieval": {
                    "top_k": 10
                },
                "chunking": {
                    "pdf": {
                        "chunk_size": 250,
                        "chunk_overlap": 25
                    },
                    "watcher": {
                        "debounce_seconds": 1.5
                    }
                }
            }
            with open("config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(custom_data, f)
                
            cfg = Config()
            assert cfg.chroma_collection_name == "custom_collection"
            assert cfg.top_k == 10
            assert cfg.pdf_chunk_size == 250
            assert cfg.pdf_chunk_overlap == 25
            assert cfg.watcher_debounce_seconds == 1.5
            assert os.path.basename(cfg.chroma_persist_dir) == "custom_chroma"
        finally:
            os.chdir(orig_cwd)
