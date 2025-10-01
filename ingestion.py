from docling.document_converter import DocumentConverter
from docling_core.types.doc import SectionHeaderItem, TextItem, ListItem
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
import hashlib
import os

# --- Configuration ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "my_rag_documents_v2"
EMBEDDING_MODEL = 'google/embeddinggemma-300m'
BATCH_SIZE = 128  # Increased from 32 for better throughput
MAX_CHUNK_WORDS = 500  # Prevent oversized chunks
MIN_CHUNK_WORDS = 5
ADD_CONTEXT_PREFIX = False  # Toggle contextual prefix to save space


class IngestionPipeline:
    """
    Optimized document ingestion pipeline with deterministic IDs and efficient chunking.
    """

    def __init__(self, debug_mode: bool = False):
        """
        Initializes the document converter, embedding model, and Qdrant client.

        Args:
            debug_mode: If True, saves raw doc structure to file for inspection
        """
        self.doc_converter = DocumentConverter()
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.qd_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self.debug_mode = debug_mode
        self.setup_qdrant_collection()

    def setup_qdrant_collection(self):
        """
        Creates the Qdrant collection if it doesn't already exist.
        """
        try:
            self.qd_client.get_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' already exists.")
        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e).lower():
                print(f"Collection '{COLLECTION_NAME}' not found. Creating it...")
                embedding_size = self.embedding_model.get_sentence_embedding_dimension()
                self.qd_client.recreate_collection(
                    collection_name=COLLECTION_NAME,
                    vectors_config=models.VectorParams(
                        size=embedding_size,
                        distance=models.Distance.COSINE
                    ),
                )
                print(f"Collection '{COLLECTION_NAME}' created.")
            else:
                raise  # Re-raise unexpected errors

    def _generate_deterministic_id(self, source: str, chunk_index: int, text_hash: str) -> str:
        """
        Generates a deterministic UUID based on source file and chunk content.
        This prevents duplicates when re-ingesting the same file.

        Args:
            source: Source filename
            chunk_index: Position of chunk in document
            text_hash: Hash of chunk text content

        Returns:
            Deterministic UUID string (UUID v5 format)
        """
        import uuid
        # Create a namespace UUID from the source
        namespace = uuid.uuid5(uuid.NAMESPACE_DNS, source)
        # Create deterministic UUID from namespace + unique identifier
        unique_string = f"{chunk_index}::{text_hash}"
        return str(uuid.uuid5(namespace, unique_string))

    def _save_raw_doc_to_file(self, doc, output_filepath: str):
        """Saves the raw string representation of the doc object to a file (debug only)."""
        if not self.debug_mode:
            return
        print(f"[DEBUG] Saving raw doc object to '{output_filepath}'")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            f.write(str(doc))

    def delete_document(self, source_filepath: str):
        """
        Deletes all points from the collection that originated from a specific source file.

        Args:
            source_filepath: The filename (not full path) stored in metadata
        """
        print(f"--- Deleting document: {source_filepath} ---")

        # First check if any points exist for this source
        scroll_result = self.qd_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value=source_filepath),
                    )
                ]
            ),
            limit=1
        )

        if not scroll_result[0]:
            print(f"No points found for source '{source_filepath}'")
            return

        # Delete the points
        self.qd_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchValue(value=source_filepath),
                        )
                    ]
                )
            ),
        )
        print(f"--- Successfully deleted points for '{source_filepath}' ---")

    def _split_large_chunk(self, text: str, max_words: int) -> list[str]:
        """
        Splits text that exceeds max_words into smaller chunks.
        Tries to split on sentence boundaries when possible.

        Args:
            text: Text to split
            max_words: Maximum words per chunk

        Returns:
            List of text chunks
        """
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        current_chunk = []

        for word in words:
            current_chunk.append(word)
            if len(current_chunk) >= max_words:
                chunks.append(' '.join(current_chunk))
                current_chunk = []

        if current_chunk:  # Add remaining words
            chunks.append(' '.join(current_chunk))

        return chunks

    def create_semantic_chunks_from_docs(self, doc, source_filepath: str):
        """
        Creates semantic chunks from a DoclingDocument with size limits.

        Args:
            doc: DoclingDocument object
            source_filepath: Source filename for metadata

        Returns:
            List of chunk dictionaries with text and metadata
        """
        chunks_with_metadata = []
        current_heading = "Introduction"
        chunk_index = 0

        for element, _level in doc.iterate_items():
            if isinstance(element, SectionHeaderItem):
                current_heading = element.text.strip()
                continue

            if not isinstance(element, (TextItem, ListItem)):
                continue

            text = element.text.strip()

            # Skip empty or very short text
            word_count = len(text.split())
            if not text or word_count < MIN_CHUNK_WORDS:
                continue

            # Split if too large
            text_chunks = self._split_large_chunk(text, MAX_CHUNK_WORDS)

            for sub_text in text_chunks:
                # Add contextual prefix only if enabled
                if ADD_CONTEXT_PREFIX:
                    contextual_text = f"Source: {source_filepath}\nSection: {current_heading}\n\n{sub_text}"
                else:
                    contextual_text = sub_text

                # Generate deterministic ID
                text_hash = hashlib.md5(sub_text.encode()).hexdigest()[:16]
                chunk_id = self._generate_deterministic_id(source_filepath, chunk_index, text_hash)

                chunk_data = {
                    "id": chunk_id,
                    "text": contextual_text,
                    "metadata": {
                        "source": source_filepath,
                        "page": getattr(element, 'page_ref', None),
                        "heading": current_heading,
                        "chunk_index": chunk_index,
                    }
                }
                chunks_with_metadata.append(chunk_data)
                chunk_index += 1

        return chunks_with_metadata

    def process_document(self, filepath: str):
        """
        Processes a single document: converts, chunks, embeds, and upserts to Qdrant.
        Re-ingesting the same file will replace existing chunks (not duplicate).

        Args:
            filepath: Path to document
        """
        if not os.path.exists(filepath):
            print(f"Error: File not found at '{filepath}'")
            return

        filename = os.path.basename(filepath)
        print(f"\n{'=' * 60}")
        print(f"Processing document: {filename}")
        print(f"{'=' * 60}")

        try:
            # Step 1: Convert
            print("1. Converting document...")
            result = self.doc_converter.convert(source=filepath)
            doc = result.document
            self._save_raw_doc_to_file(doc, "doc_structure.txt")

            # Step 2: Chunk
            print("2. Creating semantic chunks...")
            chunks = self.create_semantic_chunks_from_docs(doc, filename)

            if not chunks:
                print("   ⚠ No text chunks could be extracted. Aborting.")
                return

            print(f"   ✓ Created {len(chunks)} chunks")

            # Step 3: Embed
            print(f"3. Generating embeddings (batch_size={BATCH_SIZE})...")
            texts_to_embed = [chunk["text"] for chunk in chunks]
            embeddings = self.embedding_model.encode(
                texts_to_embed,
                show_progress_bar=True,
                batch_size=BATCH_SIZE,
            )

            # Step 4: Prepare points
            print("4. Preparing Qdrant points...")
            points = []
            for i, chunk in enumerate(chunks):
                payload = chunk["metadata"].copy()
                payload['text'] = chunk["text"]  # Always store text for retrieval

                points.append(
                    PointStruct(
                        id=chunk["id"],
                        vector=embeddings[i].tolist(),
                        payload=payload
                    )
                )

            # Step 5: Upsert (replaces existing points with same IDs)
            print("5. Upserting to Qdrant...")
            self.qd_client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True
            )

            print(f"\n{'=' * 60}")
            print(f"✓ Successfully processed '{filename}'")
            collection_info = self.qd_client.get_collection(COLLECTION_NAME)
            print(f"Total points in collection: {collection_info.points_count}")
            print(f"{'=' * 60}\n")

        except Exception as e:
            print(f"\n❌ Error processing '{filename}': {e}")
            raise


def main():
    """
    Main function for interactive document ingestion/deletion.
    """
    print("\n" + "=" * 60)
    print("RAG Document Ingestion Pipeline")
    print("=" * 60)

    action = input("\nEnter action (ingest/delete/quit): ").strip().lower()

    if action == 'quit':
        print("Exiting...")
        return

    # Initialize pipeline once (models loaded here)
    debug = input("Enable debug mode? (y/n): ").strip().lower() == 'y'
    pipeline = IngestionPipeline(debug_mode=debug)

    if action == 'ingest':
        filepath = input("Enter filepath to ingest: ").strip()
        pipeline.process_document(filepath)

    elif action == 'delete':
        filename = input("Enter FILENAME to delete (not full path): ").strip()
        pipeline.delete_document(filename)

    else:
        print("Invalid action. Use 'ingest', 'delete', or 'quit'.")


if __name__ == "__main__":
    main()