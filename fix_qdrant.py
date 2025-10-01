from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

try:
    client.delete_collection("my_rag_documents_v2")
    print("✓ Collection deleted successfully")
except Exception as e:
    print(f"Error deleting collection: {e}")
    print("Collection might not exist - that's okay!")

print("\nNow run your ingestion.py script to recreate the collection and re-ingest documents.")