from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import os
import json
from dotenv import load_dotenv

class RAGService:
    def __init__(self, query_rewrite_prompt_path="query_rewrite_prompt.txt", answer_synthesis_prompt_path="retrieval_prompt.txt"):
        load_dotenv()
        self.embedding_model = SentenceTransformer('google/embeddinggemma-300m')
        self.qdrant_client = QdrantClient(host="localhost", port=6333)
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found")
        self.client = genai.Client()
        self.collection_name = "my_rag_documents_v2"
        self.generative_model = "models/gemini-flash-latest"
        self.query_rewrite_template = self._load_prompt_template(query_rewrite_prompt_path)
        self.answer_synthesis_template = self._load_prompt_template(answer_synthesis_prompt_path)

    def _load_prompt_template(self, filepath: str) -> str:  # NEW METHOD
        """Loads the prompt template from a file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompt template file not found at: {filepath}")

    def rewrite_query(self, query: str) -> str:
        """
        Rewrites the user's query using an LLM for better retrieval.
        Uses application/json to ensure a clean, single-string output.
        """

        # The prompt now instructs the model to place the output in a JSON object.
        # This makes the `application/json` response mode more reliable.
        prompt = self.query_rewrite_template.format(query=query) + \
                 '\n\nOutput your final rewritten query in a JSON object like this: {"rewritten_query": "your rewritten query here"}'

        # Configure the model to output a JSON object.
        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

        try:
            response = self.client.models.generate_content(
                model="models/gemini-flash-lite-latest",
                contents=prompt,
                config=generation_config
            )

            # The response.text is now a guaranteed JSON string
            rewritten_query_json = json.loads(response.text)
            rewritten_query = rewritten_query_json.get("rewritten_query", query)  # Fallback to original

            print(f"   ...Rewritten query: '{rewritten_query}'")
            return rewritten_query

        except Exception as e:
            print(f"WARN: Query rewriting failed. Falling back to original query. Error: {e}")
            return query

    def retrieve_context(self, query: str, top_k: int=7):
        """
        Embeds the query and retrieves the top_k most relevant chunks from Qdrant.
        (This method is unchanged)
        """
        query_vector = self.embedding_model.encode(query).tolist()
        print(f"DEBUG: Query vector dimension: {len(query_vector)}")

        collection_info = self.qdrant_client.get_collection(self.collection_name)
        print(f"DEBUG: Collection vector size: {collection_info.config.params.vectors.size}")

        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True
        )

        context = ""
        retrieved_sources = set()

        for result in search_results:
            source_file = result.payload['source']
            retrieved_sources.add(source_file)

            context += f"[Source: {source_file}]\n"
            context += result.payload['text'] + "\n---\n"

        return {
            "context": context,
            "sources": list(retrieved_sources)
        }

    def generate_answer(self,query:str, context: str):
        """
                Generates an answer using the LLM based on the query and retrieved context.
                (This method is updated for the new API call)
                """
        print("Step 2: Generating answer with Gemini...")

        prompt = self.answer_synthesis_template.format(context=context, query=query)

        if not context:
            return {
                "answer": "I could not find any relevant information in the provided documents to answer your question.",
                "cited_docs": []
            }

        try:
            response = self.client.models.generate_content(
                model=self.generative_model,
                contents=prompt,
            )

            full_text = response.text

            if "\nCitations:" in full_text:
                parts = full_text.split("\nCitations:")
                answer_text = parts[0].strip()
                citations_text = parts[1].strip()
                cited_docs = [line.strip().lstrip('- ').strip() for line in citations_text.split('\n') if line.strip()]
            else:
                answer_text = full_text.strip()
                cited_docs = []

            return {
                "answer": answer_text,
                "cited_docs": cited_docs
            }

        except Exception as e:
            return {
                "answer": f"An error occurred while generating the answer: {e}",
                "cited_docs": []
            }

    def ask(self, query: str):
        """
        Orchestrates the entire RAG process and returns a structured dictionary
        with the answer and document references.
        """
        rewritten_query = self.rewrite_query(query)
        retrieval_results = self.retrieve_context(rewritten_query)
        generation_results = self.generate_answer(query, retrieval_results["context"])

        return {
            "original_query": query,
            "rewritten_query": rewritten_query,
            "context": retrieval_results["context"],
            "answer": generation_results["answer"],
            "cited_docs": generation_results["cited_docs"],
            "referenced_docs": retrieval_results["sources"]
        }