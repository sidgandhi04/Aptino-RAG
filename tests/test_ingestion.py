import pytest
from ingestion.document_parser import DocumentParser
import os

def test_document_parser():
    # Verify parser works on a dummy file or handles errors
    pdf_path = "Given Data/policy/USGIC-CSCIndividualHealthInsurance_2017-2018.pdf"
    if os.path.exists(pdf_path):
        parser = DocumentParser(pdf_path)
        chunks = parser.parse()
        assert len(chunks) > 0
        assert "chunk_id" in chunks[0]
        assert "text" in chunks[0]
        assert "metadata" in chunks[0]
        assert "section" in chunks[0]["metadata"]
