"""
Script untuk memproses PDF langsung ke ChromaDB.

Pipeline: PDF -> Extract Text -> Generate Embeddings -> Store to ChromaDB
Tanpa membuat file JSON intermediate.
"""

import argparse
from pathlib import Path
from tqdm import tqdm

from pdf_processor import PDFProcessor
from embedding_service import GeminiEmbeddingService
from chromadb_service import ChromaDBService
from config import Config


def process_pdf_to_chromadb(
    pdf_path: str,
    collection_name: str = None,
    use_cloud: bool = None
):
    """
    Process PDF and store embeddings directly to ChromaDB.

    Args:
        pdf_path: Path to the PDF file.
        collection_name: Name of the ChromaDB collection.
        use_cloud: Whether to use ChromaDB Cloud.
    """
    print(f"\n[INFO] Processing PDF: {pdf_path}")
    
    # Validate PDF exists
    if not Path(pdf_path).exists():
        print(f"[ERROR] PDF file not found: {pdf_path}")
        return
    
    pdf_name = Path(pdf_path).stem
    page_count = PDFProcessor.get_page_count(pdf_path)
    approach = PDFProcessor.suggest_approach(pdf_path)
    
    print(f"[INFO] Pages: {page_count}")
    print(f"[INFO] Approach: {approach}")
    
    # Initialize services
    print(f"\n[INFO] Initializing services...")
    embedding_service = GeminiEmbeddingService()
    chroma_service = ChromaDBService(
        collection_name=collection_name,
        use_cloud=use_cloud
    )
    
    mode = "Cloud" if chroma_service.use_cloud else "Local"
    print(f"[SUCCESS] Connected to ChromaDB ({mode})")
    
    # Get or create collection
    chroma_service.get_or_create_collection(
        metadata={
            "source_file": Path(pdf_path).name,
            "approach": approach
        }
    )
    print(f"[INFO] Using collection: {chroma_service.collection_name}")
    
    # Process based on approach
    embeddings = []
    documents = []
    ids = []
    metadatas = []
    
    if approach == "native" and page_count <= 6:
        print(f"\n[INFO] Using native PDF embedding...")
        
        # Read PDF as bytes
        pdf_bytes = PDFProcessor.read_pdf_bytes(pdf_path)
        
        # Generate embedding for entire PDF
        print(f"[INFO] Generating embedding for PDF...")
        embedding = embedding_service.embed_pdf_bytes(pdf_bytes)
        
        # Extract text for storage (optional, for search purposes)
        full_text = PDFProcessor.extract_text(pdf_path)
        
        # Prepare data
        embeddings.append(embedding)
        documents.append(full_text)
        ids.append(f"{pdf_name}_full")
        metadatas.append({
            "source_file": Path(pdf_path).name,
            "approach": "native_pdf_embedding",
            "page_count": page_count
        })
        
        print(f"[INFO] Generated 1 embedding for entire PDF")
    
    else:
        print(f"\n[INFO] Using text extraction and chunking...")
        
        # Extract text
        print(f"[INFO] Extracting text from PDF...")
        full_text = PDFProcessor.extract_text(pdf_path)
        
        # Chunk text
        print(f"[INFO] Chunking text...")
        chunks = PDFProcessor.chunk_text(
            full_text,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        
        print(f"[INFO] Created {len(chunks)} chunks")
        
        # Generate embeddings for chunks
        print(f"\n[INFO] Generating embeddings...")
        chunk_embeddings = embedding_service.embed_texts(chunks)
        
        # Prepare data
        for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
            embeddings.append(embedding)
            documents.append(chunk)
            ids.append(f"{pdf_name}_chunk_{i}")
            metadatas.append({
                "source_file": Path(pdf_path).name,
                "approach": "text_extraction",
                "chunk_index": i,
                "total_chunks": len(chunks)
            })
        
        print(f"[INFO] Generated {len(embeddings)} embeddings")
    
    # Store to ChromaDB
    print(f"\n[INFO] Storing embeddings to ChromaDB...")
    
    # Store in batches to avoid memory issues
    batch_size = 100
    for i in tqdm(range(0, len(embeddings), batch_size), desc="Storing batches"):
        batch_end = min(i + batch_size, len(embeddings))
        
        chroma_service.add_embeddings(
            embeddings=embeddings[i:batch_end],
            documents=documents[i:batch_end],
            ids=ids[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
    
    # Verify
    count = chroma_service.count_documents()
    print(f"\n[SUCCESS] Successfully stored {count} documents to ChromaDB!")
    
    # Display collection info
    print(f"\n[INFO] Collection Info:")
    print(f"   - Name: {chroma_service.collection_name}")
    print(f"   - Total documents: {count}")
    print(f"   - Mode: {mode}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Process PDF and store embeddings directly to ChromaDB"
    )
    parser.add_argument(
        "pdf_file",
        type=str,
        help="Path to the PDF file"
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name (default: from config)"
    )
    parser.add_argument(
        "--cloud",
        action="store_true",
        help="Use ChromaDB Cloud (default: from config)"
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local ChromaDB (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
        return
    
    # Determine use_cloud setting
    use_cloud = None
    if args.cloud:
        use_cloud = True
    elif args.local:
        use_cloud = False
    
    # Process PDF
    try:
        process_pdf_to_chromadb(
            pdf_path=args.pdf_file,
            collection_name=args.collection,
            use_cloud=use_cloud
        )
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        raise


if __name__ == "__main__":
    main()
