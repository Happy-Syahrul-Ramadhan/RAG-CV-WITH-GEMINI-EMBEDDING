"""
Test ChromaDB connection untuk debugging.
"""

from chromadb_service import ChromaDBService
from config import Config

print("Testing ChromaDB Cloud connection...")
print(f"USE_CHROMA_CLOUD: {Config.USE_CHROMA_CLOUD}")
print(f"API_KEY: {Config.CHROMA_CLOUD_API_KEY[:20]}..." if Config.CHROMA_CLOUD_API_KEY else "API_KEY: Not set")
print(f"TENANT: {Config.CHROMA_CLOUD_TENANT}")
print(f"DATABASE: {Config.CHROMA_CLOUD_DATABASE}")

try:
    service = ChromaDBService(use_cloud=True)
    print("\n[SUCCESS] ChromaDB client initialized")
    
    service.get_or_create_collection()
    print("[SUCCESS] Collection created/retrieved")
    
    count = service.count_documents()
    print(f"[SUCCESS] Document count: {count}")
    
    print("\n[SUCCESS] All tests passed!")
    
except Exception as e:
    print(f"\n[ERROR] Connection failed: {e}")
    import traceback
    traceback.print_exc()
