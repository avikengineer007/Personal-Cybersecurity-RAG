import os
from typing import List
# pyrefly: ignore [missing-import]
from llama_index.core import SimpleDirectoryReader
# pyrefly: ignore [missing-import]
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from seckg.models import Chunk
from seckg.config import settings

def ingest_file(file_path: str, base_dir: str) -> List[Chunk]:
    """Ingests a single file (markdown or PDF), chunks it, and returns a list of Chunk models."""
    if not os.path.exists(file_path):
        return []

    # Check extension
    _, ext = os.path.splitext(file_path.lower())
    if ext not in [".md", ".pdf"]:
        return []

    # Get last modified time
    last_modified = os.path.getmtime(file_path)

    # Compute relative path using forward slashes for cross-platform consistency
    rel_path = os.path.relpath(file_path, base_dir)
    posix_rel_path = rel_path.replace("\\", "/")

    # Load data using LlamaIndex SimpleDirectoryReader
    # We specify the exact file to load
    try:
        reader = SimpleDirectoryReader(input_files=[file_path])
        documents = reader.load_data()
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return []

    chunks = []

    if ext == ".md":
        # Parse Markdown by headings
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        
        for node in nodes:
            # Extract heading path from parent path metadata + current node header
            header_path = node.metadata.get("header_path", "/")
            # Strip carriage returns and leading/trailing whitespace from each part
            parts = [p.replace("\r", "").strip() for p in header_path.split("/") if p.strip()]
            
            content = node.get_content().strip()
            first_line = content.split("\n")[0] if content else ""
            first_line = first_line.replace("\r", "").strip()
            if first_line.startswith("#"):
                current_heading = first_line.lstrip("#").strip()
                parts.append(current_heading)
                
            heading_path = " > ".join(parts) if parts else None

            chunks.append(
                Chunk(
                    content=node.get_content(),
                    source_file=posix_rel_path,
                    heading_path=heading_path,
                    page_number=None,
                    chunk_id=node.node_id,
                    last_modified=last_modified
                )
            )

    elif ext == ".pdf":
        # Parse PDF using sliding window sentence splitter
        parser = SentenceSplitter(
            chunk_size=settings.pdf_chunk_size,
            chunk_overlap=settings.pdf_chunk_overlap
        )
        nodes = parser.get_nodes_from_documents(documents)

        for node in nodes:
            # LlamaIndex SimpleDirectoryReader PDF loader adds page_label to node metadata
            page_label = node.metadata.get("page_label")
            page_num = None
            if page_label:
                try:
                    page_num = int(page_label)
                except ValueError:
                    # page_label might be a non-numeric string (e.g., 'i', 'ii'), default to None
                    pass

            chunks.append(
                Chunk(
                    content=node.get_content(),
                    source_file=posix_rel_path,
                    heading_path=None,
                    page_number=page_num,
                    chunk_id=node.node_id,
                    last_modified=last_modified
                )
            )

    return chunks

def ingest_directory(directory_path: str) -> List[Chunk]:
    """Scans the directory recursively and returns all parsed chunks."""
    all_chunks = []
    abs_base = os.path.abspath(directory_path)

    for root, _, files in os.walk(abs_base):
        for file in files:
            _, ext = os.path.splitext(file.lower())
            if ext in [".md", ".pdf"]:
                file_path = os.path.join(root, file)
                # Skip files inside hidden folders (like .git, .obsidian, etc.) unless requested
                if any(part.startswith(".") for part in os.path.relpath(file_path, abs_base).split(os.sep)):
                    continue
                
                chunks = ingest_file(file_path, abs_base)
                all_chunks.extend(chunks)

    return all_chunks
