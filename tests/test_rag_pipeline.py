import pytest
from langchain_core.documents import Document
from app.services.rag_service import RAGService
from app.config import settings

def test_structure_detection_and_chunking():
    service = RAGService()
    
    synthetic_text = """## 4.1
- Point 1 for 4.1
- Point 2 for 4.1

## 4.2
- Point 1 for 4.2
- Point 2 for 4.2

## 4.3
- Point 1 for 4.3
- Point 2 for 4.3
"""
    doc = Document(page_content=synthetic_text, metadata={"document_id": "test_doc", "filename": "test.md", "page_number": 1})
    
    # 1. Detect structure
    structured_docs = service._detect_structure([doc])
    
    assert len(structured_docs) == 3, f"Expected 3 structured docs, got {len(structured_docs)}"
    assert structured_docs[0].metadata["section"] == "4.1"
    assert structured_docs[1].metadata["section"] == "4.2"
    assert structured_docs[2].metadata["section"] == "4.3"
    
    # 2. Chunking
    chunks = service._chunk(structured_docs)
    
    # Should produce exactly 3 chunks (one per section)
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"
    
    c0 = chunks[0].page_content
    c1 = chunks[1].page_content
    c2 = chunks[2].page_content
    
    assert "4.2" not in c0
    assert "4.1" not in c1
    assert "4.3" not in c1
    assert "4.2" not in c2
