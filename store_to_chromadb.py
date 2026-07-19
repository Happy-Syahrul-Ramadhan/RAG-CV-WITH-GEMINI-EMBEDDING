"""
Script untuk menyimpan embedding dari JSON ke ChromaDB.

Script ini membaca file JSON yang berisi embedding dari PDF
dan menyimpannya ke ChromaDB (local atau cloud).
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from chromadb_service import ChromaDBService
from config import Config


def load_embedding_json(json_path: str) -> dict:
    """Load embedding data from JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def store_embeddings_to_chromadb(
    json_path: str,
    collection_name: str = None,
    use_cloud: bool = None
):
    """
    Store embeddings from JSON file to ChromaDB.

    Args:
        json_path: Path to the embedding JSON file.
        collection_name: Name of the ChromaDB collection.
        use_cloud: Whether to use ChromaDB Cloud.
    """
    print(f"\n[INFO] Loading embeddings from: {json_path}")
    data = load_embedding_json(json_path)
    
    # Extract data
    filename = data.get('filename', 'unknown')
    approach = data.get('approach', 'unknown')
    chunks = data.get('chunks', [])
    
    if not chunks:
        print("[ERROR] No chunks found in the JSON file.")
        return
    
    print(f"[INFO] Document: {filename}")
    print(f"[INFO] Total chunks: {len(chunks)}")
    print(f"[INFO] Approach: {approach}")
    
    # Initialize ChromaDB service
    print(f"\n[INFO] Connecting to ChromaDB...")
    chroma_service = ChromaDBService(
        collection_name=collection_name,
        use_cloud=use_cloud
    )
    
    mode = "Cloud" if chroma_service.use_cloud else "Local"
    print(f"[SUCCESS] Connected to ChromaDB ({mode})")
    
    # Get or create collection
    collection = chroma_service.get_or_create_collection(
        metadata={
            "source_file": filename,
            "approach": approach
        }
    )
    print(f"[INFO] Using collection: {chroma_service.collection_name}")
    
    # Prepare data for ChromaDB
    embeddings = []
    documents = []
    ids = []
    metadatas = []
    
    print(f"\n[INFO] Preparing data for ChromaDB...")
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        # Generate unique ID
        chunk_id = f"{filename}_chunk_{i}"
        
        # Extract data
        embedding = chunk.get('embedding', [])
        text = chunk.get('text', '')
        page_number = chunk.get('page_number')
        chunk_index = chunk.get('chunk_index')
        
        # Prepare metadata
        metadata = {
            "source_file": filename,
            "approach": approach
        }
        if page_number is not None:
            metadata['page_number'] = page_number
        if chunk_index is not None:
            metadata['chunk_index'] = chunk_index
        
        embeddings.append(embedding)
        documents.append(text)
        ids.append(chunk_id)
        metadatas.append(metadata)
    
    # Store to ChromaDB
    print(f"\n[INFO] Storing embeddings to ChromaDB...")
    chroma_service.add_embeddings(
        embeddings=embeddings,
        documents=documents,
        ids=ids,
        metadatas=metadatas
    )
    
    # Verify
    count = chroma_service.count_documents()
    print(f"[SUCCESS] Successfully stored {count} documents to ChromaDB!")
    
    # Display collection info
    print(f"\n[INFO] Collection Info:")
    print(f"   - Name: {chroma_service.collection_name}")
    print(f"   - Total documents: {count}")
    print(f"   - Mode: {mode}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Store PDF embeddings from JSON to ChromaDB"
    )
    parser.add_argument(
        "json_file",
        type=str,
        help="Path to the embedding JSON file"
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
    
    # Check if file exists
    if not Path(args.json_file).exists():
        print(f"[ERROR] File not found: {args.json_file}")
        return
    
    # Store embeddings
    try:
        store_embeddings_to_chromadb(
            json_path=args.json_file,
            collection_name=args.collection,
            use_cloud=use_cloud
        )
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        raise


if __name__ == "__main__":
    main()
