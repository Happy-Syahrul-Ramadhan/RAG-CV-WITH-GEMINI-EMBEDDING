"""
PDF Embedding Starter - Main Entry Point.

A tool to embed PDF documents using Google Gemini Embedding API
and store them in ChromaDB for semantic search.

Usage:
    python main.py embed <pdf_path>                    # Embed a PDF
    python main.py query "<your question>"             # Search existing embeddings
    python main.py info                                 # Show database stats
    python main.py list                                 # List collections
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

# Ensure the project root is in path
sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from pdf_processor import PDFProcessor
from embedding_service import GeminiEmbeddingService
from chroma_service import ChromaService

# ASCII markers for Windows terminal compatibility
ERR = "[ERROR]"
PDF = "[PDF]"
SRCH = "[SEARCH]"
STAT = "[STATS]"
MODE = "[MODE]"
PROC = "[PROC]"
OK = "[OK]"
TEXT = "[TEXT]"
SIZE = "[SIZE]"
CHNK = "[CHUNK]"
CHKS = "[CHUNKS]"
RES = "[RESULT]"
SRC = "[SOURCE]"
IDX = "[INDEX]"
LINE = "-" * 60


def cmd_embed(args):
    """Embed a PDF document into ChromaDB."""
    pdf_path = args.pdf

    if not os.path.exists(pdf_path):
        print(f"{ERR} File not found: {pdf_path}")
        sys.exit(1)

    print(f"{PDF} Processing PDF: {pdf_path}")
    print(f"{SRCH} Analyzing document...")

    # Initialize services
    embed_service = GeminiEmbeddingService()
    chroma = ChromaService(embedding_service=embed_service)

    # Check approach
    approach = PDFProcessor.suggest_approach(pdf_path)
    page_count = PDFProcessor.get_page_count(pdf_path)
    pdf_name = os.path.basename(pdf_path)

    print(f"{STAT} Pages: {page_count}")
    print(f"{MODE} Approach: {'Native PDF embedding' if approach == 'native' else 'Text extraction + chunking'}")

    if approach == "native":
        # Use native multimodal PDF embedding (<=6 pages)
        print(f"{PROC} Sending PDF directly to Gemini-embedding-2...")
        pdf_bytes = PDFProcessor.read_pdf_bytes(pdf_path)
        embedding = embed_service.embed_pdf_bytes(pdf_bytes)
        print(f"{OK} Embedding generated! Dimensions: {len(embedding)}")

        # Extract text for storage
        text = PDFProcessor.extract_text(pdf_path)

        # Store in ChromaDB
        chroma.add_pdf_chunks(
            chunks=[text],
            source_pdf=pdf_name,
        )
        print(f"{OK} Stored in ChromaDB collection: '{chroma.collection_name}'")

    else:
        # Extract text and chunk for longer PDFs
        print(f"{TEXT} Extracting text from PDF...")
        pages = PDFProcessor.extract_text_by_page(pdf_path)
        full_text = "\n\n".join([text for _, text in pages])

        print(f"{SIZE} Total characters: {len(full_text):,}")
        print(f"{CHNK} Chunking text (size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP})...")

        chunks = PDFProcessor.chunk_text(
            full_text,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )
        print(f"{CHKS} Created {len(chunks)} chunks")

        # Store chunks with page metadata
        print(f"{PROC} Generating embeddings and storing in ChromaDB...")
        page_refs = []
        for chunk in chunks:
            # Find which page this chunk comes from
            char_pos = full_text.find(chunk[:50])
            cumulative = 0
            page_ref = 0
            for page_num, page_text in pages:
                cumulative += len(page_text) + 2  # +2 for the double newline separator
                if char_pos < cumulative:
                    page_ref = page_num
                    break
            page_refs.append(page_ref)

        chroma.add_pdf_chunks(
            chunks=chunks,
            source_pdf=pdf_name,
            page_numbers=page_refs,
        )
        print(f"{OK} Done! {len(chunks)} chunks stored in ChromaDB collection: '{chroma.collection_name}'")

    # Show stats
    stats = chroma.get_collection_stats()
    print(f"\n{STAT} Collection Stats:")
    for key, value in stats.items():
        print(f"   - {key}: {value}")


def cmd_query(args):
    """Search the embedding database."""
    query = args.query
    n_results = args.top_k

    print(f'{SRCH} Searching for: "{query}"')
    print(f"Top {n_results} results...\n")

    chroma = ChromaService()
    results = chroma.search(query, n_results=n_results)

    if not results["documents"]:
        print(f"{ERR} No results found. Make sure you've embedded some PDFs first.")
        return

    for i, (doc, meta, dist) in enumerate(
        zip(results["documents"], results["metadatas"], results["distances"])
    ):
        score = 1 - dist  # Convert distance to similarity score
        print(LINE)
        print(f"{RES} #{i+1} (Similarity: {score:.4f})")
        print(LINE)
        if meta:
            print(f"{SRC} Source: {meta.get('source', 'N/A')}")
            print(f"Page: {meta.get('page', 'N/A')}")
            print(f"{IDX} Chunk: {meta.get('chunk_index', 'N/A')}")
        print(f"\nPreview: {doc[:500]}..." if len(doc) > 500 else f"\n{doc}")
        print()


def cmd_info(args):
    """Show database information."""
    chroma = ChromaService()
    stats = chroma.get_collection_stats()
    print(f"{STAT} ChromaDB Collection Info:")
    print(f"{'-'*40}")
    for key, value in stats.items():
        print(f"  {key.replace('_', ' ').title()}: {value}")


def cmd_list(args):
    """List all collections."""
    chroma = ChromaService()
    collections = chroma.list_collections()
    print("Available Collections:")
    if collections:
        for col in collections:
            print(f"  - {col}")
    else:
        print("  (no collections found)")


def cmd_export(args):
    """Export PDF text and embedding to a single JSON file."""
    pdf_path = args.pdf

    if not os.path.exists(pdf_path):
        print(f"{ERR} File not found: {pdf_path}")
        sys.exit(1)

    print(f"{PDF} Processing PDF: {pdf_path}")

    pdf_name = os.path.basename(pdf_path)
    page_count = PDFProcessor.get_page_count(pdf_path)
    approach = PDFProcessor.suggest_approach(pdf_path)
    embed_service = GeminiEmbeddingService()

    # Prepare result structure
    result = {
        "filename": pdf_name,
        "file_path": os.path.abspath(pdf_path),
        "page_count": page_count,
        "approach": "native_pdf_embedding" if approach == "native" else "text_extract_chunking",
        "embedding_model": embed_service.model,
        "embedding_dimensions": embed_service.dimensionality,
        "processed_at": datetime.now().isoformat(),
        "chunks": [],
    }

    if approach == "native":
        # Native PDF embedding (<=6 pages)
        print(f"  -> Using native PDF embedding...")
        pdf_bytes = PDFProcessor.read_pdf_bytes(pdf_path)
        text = PDFProcessor.extract_text(pdf_path)
        embedding = embed_service.embed_pdf_bytes(pdf_bytes)

        result["chunks"].append({
            "chunk_index": 0,
            "page": None,
            "char_count": len(text),
            "text": text,
            "embedding": embedding,
        })

    else:
        # Extract text and chunk for longer PDFs
        print(f"  -> Extracting text and chunking...")
        pages = PDFProcessor.extract_text_by_page(pdf_path)
        full_text = "\n\n".join([text for _, text in pages])

        chunks = PDFProcessor.chunk_text(
            full_text,
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
        )

        print(f"  -> Generating embeddings for {len(chunks)} chunks...")

        # Determine page for each chunk
        page_refs = []
        for chunk in chunks:
            char_pos = full_text.find(chunk[:50])
            cumulative = 0
            page_ref = 0
            for page_num, page_text in pages:
                cumulative += len(page_text) + 2
                if char_pos < cumulative:
                    page_ref = page_num
                    break
            page_refs.append(page_ref)

        # Get embeddings for all chunks at once
        embeddings = embed_service.embed_texts(chunks)

        for i, (chunk, emb, page) in enumerate(zip(chunks, embeddings, page_refs)):
            result["chunks"].append({
                "chunk_index": i,
                "page": page,
                "char_count": len(chunk),
                "text": chunk,
                "embedding": emb,
            })

    # Save to JSON file
    output_path = args.output or pdf_name.rsplit(".", 1)[0] + "_embedding.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    file_size_kb = os.path.getsize(output_path) / 1024
    print(f"{OK} Saved to: {output_path}")
    print(f"{STAT} File size: {file_size_kb:.1f} KB")
    print(f"{STAT} Total chunks: {len(result['chunks'])}")
    print(f"{STAT} Embedding dims: {result['embedding_dimensions']}")


def cmd_search_json(args):
    """Search exported JSON embedding file."""
    json_file = args.json_file
    query = args.query
    top_k = args.top_k

    if not os.path.exists(json_file):
        print(f"{ERR} File not found: {json_file}")
        sys.exit(1)

    print(f"{SRCH} Loading: {json_file}")
    print(f'{SRCH} Query: "{query}"')
    print(f"{STAT} Top-K: {top_k}")

    import search_json as sj

    try:
        results = sj.search_json(json_file, query, top_k=top_k)
        sj.print_results(results, query)
    except Exception as e:
        print(f"{ERR} Search failed: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="PDF Embedding Starter - Embed PDFs with Gemini & ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py embed document.pdf
  python main.py query "What is this document about?"
  python main.py info
  python main.py list
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Embed command
    embed_parser = subparsers.add_parser("embed", help="Embed a PDF into ChromaDB")
    embed_parser.add_argument("pdf", help="Path to the PDF file")

    # Query command
    query_parser = subparsers.add_parser("query", help="Search the embedding database")
    query_parser.add_argument("query", help="Your search query")
    query_parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )

    # Info command
    subparsers.add_parser("info", help="Show database statistics")

    # List command
    subparsers.add_parser("list", help="List all collections")

    # Export command
    export_parser = subparsers.add_parser(
        "export",
        help="Export PDF text + embedding to a single JSON file"
    )
    export_parser.add_argument("pdf", help="Path to the PDF file")
    export_parser.add_argument(
        "-o", "--output",
        help="Output JSON file path (default: [pdf_name]_embedding.json)"
    )

    # Search-JSON command
    search_json_parser = subparsers.add_parser(
        "search-json",
        help="Search exported JSON embedding file"
    )
    search_json_parser.add_argument("json_file", help="Path to the JSON embedding file")
    search_json_parser.add_argument("query", help="Your search query")
    search_json_parser.add_argument(
        "-k", "--top-k", type=int, default=5,
        help="Number of results to return (default: 5)"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Validate config before running commands
    try:
        Config.validate()
    except ValueError as e:
        print(f"{ERR} Configuration Error: {e}")
        sys.exit(1)

    # Route commands
    commands = {
        "embed": cmd_embed,
        "query": cmd_query,
        "info": cmd_info,
        "list": cmd_list,
        "export": cmd_export,
        "search-json": cmd_search_json,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
