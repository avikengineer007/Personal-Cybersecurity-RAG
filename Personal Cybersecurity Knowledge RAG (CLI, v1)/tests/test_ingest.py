import os
import tempfile
# pyrefly: ignore [missing-import]
import pytest
from seckg.ingest import ingest_file, ingest_directory

def test_ingest_markdown_structure():
    """Verify that Markdown files are chunked by headers and heading paths are preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a test markdown file
        md_content = (
            "# Introduction to Cryptography\n\n"
            "This is general information about cryptography.\n\n"
            "## Symmetric Encryption\n\n"
            "Symmetric encryption uses a single shared secret key.\n\n"
            "### AES Algorithm\n\n"
            "Advanced Encryption Standard (AES) is a symmetric block cipher.\n"
        )
        file_path = os.path.join(tmpdir, "crypto_notes.md")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        # Ingest the file
        chunks = ingest_file(file_path, tmpdir)
        
        # We expect at least one chunk for each heading division
        assert len(chunks) >= 3
        
        # Check source file normalization
        for chunk in chunks:
            assert chunk.source_file == "crypto_notes.md"
            assert chunk.page_number is None
            assert chunk.last_modified > 0.0

        # Validate heading hierarchies
        headings = [c.heading_path for c in chunks if c.heading_path]
        assert "Introduction to Cryptography" in headings
        assert "Introduction to Cryptography > Symmetric Encryption" in headings
        assert "Introduction to Cryptography > Symmetric Encryption > AES Algorithm" in headings

        # Validate content grounding
        aes_chunk = [c for c in chunks if c.heading_path and "AES Algorithm" in c.heading_path][0]
        assert "Advanced Encryption Standard" in aes_chunk.content

def test_ingest_unsupported_file():
    """Verify that unsupported file extensions are ignored during ingestion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "image.png")
        with open(file_path, "wb") as f:
            f.write(b"\x89PNG\r\n\x1a\n...")
            
        chunks = ingest_file(file_path, tmpdir)
        assert len(chunks) == 0
