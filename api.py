from fastapi import FastAPI
from pydantic import BaseModel
from rag_service import RAGService
from tasks import judge_and_log_task
from fastapi import UploadFile, File
from ingestion import IngestionPipeline # Our ingestion logic
import tempfile
import os

app = FastAPI(
    title='EvoRAG API',
    description='API for a self evolving RAG pipeline with background evaluation',
    version='1.0',
)

rag_service = RAGService(
    query_rewrite_prompt_path="query_rewrite_prompt.txt",
    answer_synthesis_prompt_path="retrieval_prompt.txt"
)

print("Loading Ingestion Pipeline...")
ingestion_pipeline = IngestionPipeline()
print("Ingestion Pipeline loaded.")

class QueryRequest(BaseModel):
    query: str


@app.post("/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """
    Receives a file, saves it temporarily, processes it using the
    IngestionPipeline, and then cleans up the temporary file.
    """
    if not file.filename:
        return {"error": "No file name provided"}

    try:
        # Create a temporary directory to save the uploaded file
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_filepath = os.path.join(temp_dir, file.filename)

            # Write the uploaded file content to the temporary file
            with open(temp_filepath, "wb") as buffer:
                buffer.write(await file.read())

            print(f"API: Ingesting document: {file.filename}")
            # Run the ingestion process on the saved file
            ingestion_pipeline.process_document(temp_filepath)

        print(f"API: Successfully ingested {file.filename}")
        return {"message": f"Successfully ingested and processed {file.filename}"}

    except Exception as e:
        print(f"API Error during ingestion: {e}")
        return {"error": f"Failed to ingest document. Error: {e}"}

@app.post("/ask")
def ask_question(request: QueryRequest):
    """
    This is the main endpoint for our application. It does three things:
    1. Gets an answer for the user's query from the RAG service.
    2. Immediately returns the answer to the user for a low-latency experience.
    3. Dispatches a background task to judge and log the interaction for later analysis.
    """
    result = rag_service.ask(request.query)

    judge_and_log_task.delay(
        original_query=result["original_query"],
        rewritten_query=result["rewritten_query"],
        context=result["context"],
        generation_results={
            "answer": result["answer"],
            "cited_docs": result["cited_docs"]
        }
    )
    return {
        "answer": result["answer"],
        "cited_docs": result["cited_docs"],
        "referenced_docs": result["referenced_docs"]
    }

@app.get("/log")
def read_root():
    return {"message": "Welcome to the EvoRAG API. Please use the /docs endpoint to see the API documentation."}