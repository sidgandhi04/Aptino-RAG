import sys
from ingestion.document_parser import DocumentParser
from ingestion.vector_store import VectorStore

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m ingestion.indexer <path_to_policy_pdf>")
        sys.exit(1)
        
    pdf_path = sys.argv[1]
    
    try:
        print(f"Parsing document: {pdf_path}")
        parser = DocumentParser(pdf_path)
        chunks = parser.parse()
        print(f"Total chunks extracted: {len(chunks)}")
        
        print("Initializing Vector Store...")
        store = VectorStore()
        
        print("Adding chunks to Vector Store...")
        store.add_chunks(chunks)
        print("Indexing finished successfully.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
