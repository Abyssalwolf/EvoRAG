from celery_app import app
from llm_judge import LLMJudge
import datetime
import json

LOG_FILE = "evaluation_logs.jsonl"

judge = None

def get_judge():
    global judge
    if judge is None:
        judge = LLMJudge()
    return judge

def _log_evaluation(log_data: dict):
    """Append a log entry to the evaluation log"""
    log_data['timestamp'] = datetime.datetime.now().isoformat()
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_data) + '\n')
    print(f"Successfully logged evaluation for query.")

@app.task
def judge_and_log_task(original_query: str, rewritten_query: str, context: str, generation_results: dict):
    """
    Celery task to judge answer and log evaluation results
    """
    llm_judge = get_judge()
    evaluation = llm_judge.judge_answer(
        original_query=original_query,
        rewritten_query=rewritten_query,
        context=context,
        answer=generation_results["answer"]
    )

    log_entry = {
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "retrieved_context": context,
        "generated_answer": generation_results,
        "evaluation": evaluation
    }

    _log_evaluation(log_entry)
