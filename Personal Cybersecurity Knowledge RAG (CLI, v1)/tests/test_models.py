from seckg.models import Chunk, RetrievalResult, Answer

def test_chunk_validation():
    """Verify that the Chunk model validates inputs correctly."""
    chunk = Chunk(
        content="Test context regarding cybersecurity firewalls.",
        source_file="security_plus/firewalls.md",
        heading_path="Security+ Notes > Network Security > Firewalls",
        chunk_id="node_id_12345",
        last_modified=1719717600.0
    )
    assert chunk.content == "Test context regarding cybersecurity firewalls."
    assert chunk.source_file == "security_plus/firewalls.md"
    assert chunk.heading_path == "Security+ Notes > Network Security > Firewalls"
    assert chunk.page_number is None
    assert chunk.chunk_id == "node_id_12345"
    assert chunk.last_modified == 1719717600.0

def test_retrieval_result_validation():
    """Verify that RetrievalResult wraps a Chunk and score successfully."""
    chunk = Chunk(
        content="Test content",
        source_file="notes/test.md",
        chunk_id="test_id_123",
        last_modified=1234567.89
    )
    result = RetrievalResult(chunk=chunk, score=0.92)
    assert result.score == 0.92
    assert result.chunk.chunk_id == "test_id_123"
