"""
search_json.py - Search PDF embeddings from exported JSON file.

Loads a JSON embedding file (produced by 'python main.py export <pdf>')
and performs semantic search using cosine similarity.

Usage:
    python search_json.py <json_file> "<query>" [-k N]

Examples:
    python search_json.py "CV Happy Syahrul Ramadhan_embedding.json" "Apa keahlian utama?"
    python search_json.py hasil.json "pendidikan" -k 3
"""

import json
import math
import sys
import os
from typing import List, Dict, Any, Optional

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from embedding_service import GeminiEmbeddingService


# ─── Similarity ──────────────────────────────────────────────────────────────

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Returns a value between -1 and 1 (higher = more similar).
    """
    if len(vec_a) != len(vec_b):
        raise ValueError(
            f"Vector dimension mismatch: {len(vec_a)} vs {len(vec_b)}"
        )

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# ─── Search ──────────────────────────────────────────────────────────────────

def load_json(file_path: str) -> Dict[str, Any]:
    """Load and validate the embedding JSON file."""
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Basic validation
    if "chunks" not in data or not data["chunks"]:
        print(f"[ERROR] No chunks found in {file_path}")
        sys.exit(1)

    required_keys = {"chunk_index", "text", "embedding"}
    first_chunk_keys = set(data["chunks"][0].keys())
    if not required_keys.issubset(first_chunk_keys):
        print(f"[ERROR] Invalid format. Missing keys: {required_keys - first_chunk_keys}")
        sys.exit(1)

    return data


def search_json(
    json_file: str,
    query: str,
    top_k: int = 5,
    embedding_service: Optional[GeminiEmbeddingService] = None,
) -> List[Dict[str, Any]]:
    """
    Perform semantic search on exported JSON embedding file.

    Args:
        json_file: Path to the JSON embedding file.
        query: Natural language query.
        top_k: Number of results to return.
        embedding_service: Gemini embedding service (created if None).

    Returns:
        List of results sorted by similarity (highest first).
    """
    # Load data
    data = load_json(json_file)

    # Initialize embedding service
    if embedding_service is None:
        embedding_service = GeminiEmbeddingService()

    # Embed the query
    print(f"[SEARCH] Query: \"{query}\"")
    print(f"[SEARCH] Generating query embedding...")
    query_embedding = embedding_service.embed_text(query)

    # Compute similarity against all stored chunks
    print(f"[SEARCH] Comparing against {len(data['chunks'])} chunk(s)...")
    results = []
    for chunk in data["chunks"]:
        stored_embedding = chunk["embedding"]
        score = cosine_similarity(query_embedding, stored_embedding)
        results.append({
            "score": score,
            "chunk_index": chunk["chunk_index"],
            "page": chunk.get("page"),
            "char_count": chunk.get("char_count", 0),
            "text": chunk["text"],
            "source": data.get("filename", json_file),
        })

    # Sort by similarity score (highest first)
    results.sort(key=lambda r: r["score"], reverse=True)

    return results[:top_k]


def print_results(results: List[Dict[str, Any]], query: str):
    """Pretty-print search results."""
    print()
    print("=" * 70)
    print(f'  SEARCH RESULTS for: "{query}"')
    print("=" * 70)

    if not results:
        print("  No results found.")
        return

    for i, r in enumerate(results):
        score_pct = r["score"] * 100
        print()
        print(f"  [{i+1}] Score: {score_pct:.2f}%")
        print(f"  {'-' * 60}")
        if r["source"]:
            print(f"      Source: {r['source']}")
        if r["page"]:
            print(f"      Page: {r['page']}")
        print(f"      Chunk: #{r['chunk_index']} ({r['char_count']} chars)")
        print()
        # Show a preview of the text
        text_preview = r["text"][:600].strip()
        if len(r["text"]) > 600:
            text_preview += "..."
        for line in text_preview.split("\n"):
            print(f"      {line}")
        print()


def main():
    # Parse args manually for standalone simplicity
    args = sys.argv[1:]

    if not args or "-h" in args or "--help" in args:
        print(__doc__)
        return

    # Validate GEMINI_API_KEY
    try:
        Config.validate()
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # Determine json_file and query
    json_file = args[0]
    query = None
    top_k = 5

    # Parse remaining args
    rest = args[1:]
    i = 0
    while i < len(rest):
        if rest[i] in ("-k", "--top-k") and i + 1 < len(rest):
            top_k = int(rest[i + 1])
            i += 2
        else:
            query = rest[i]
            i += 1

    if not query:
        print("[ERROR] Please provide a search query.")
        print(__doc__)
        sys.exit(1)

    print(f"[INFO] Loading: {json_file}")
    print(f"[INFO] Top-K: {top_k}")

    # Run search
    try:
        results = search_json(json_file, query, top_k=top_k)
        print_results(results, query)
    except Exception as e:
        print(f"[ERROR] Search failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
