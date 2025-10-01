import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

class LLMJudge:
    def __init__(self):
        """
        Initializes the Judge LLM.
        """
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in .env file.")

            # We use a powerful model for the judge, as reasoning is a hard task.
        self.client = genai.Client()
        self.judge_model = "models/gemini-2.5-pro"
        print("LLM Judge initialized.")

    def _create_judge_prompt(self, original_query, rewritten_query, context, answer):
        """
        Creates the detailed, structured prompt for the Judge LLM.
        """
        return f"""
        You are a meticulous AI evaluator for a multi-step RAG (Retrieval-Augmented Generation) pipeline. Your task is to assess two distinct stages and output your findings in a single, valid JSON object that strictly adheres to the provided schema.
    
        **---------------- PART 1: EVALUATE THE QUERY REWRITE ----------------**
    
        **Instructions:**
        Analyze the `[ORIGINAL QUERY]` and the `[REWRITTEN QUERY]`. The goal of the rewrite is to improve the query for a vector database search by adding context, keywords, and synonyms without losing the original intent.
    
        **---------------- PART 2: EVALUATE THE FINAL ANSWER ----------------**
    
        **Instructions:**
        Analyze the `[FINAL ANSWER]` based ONLY on the `[RETRIEVED CONTEXT]`. The answer must be fully grounded in the provided context and must directly address the `[ORIGINAL QUERY]`.
    
        **---------------- DATA TO EVALUATE ----------------**
    
        [ORIGINAL QUERY]: {original_query}
    
        [REWRITTEN QUERY]: {rewritten_query}
    
        [RETRIEVED CONTEXT]:
        {context}
    
        [FINAL ANSWER]:
        {answer}
    
        **---------------- YOUR JSON EVALUATION ----------------**
    
        **Instructions for JSON Output:**
        Your entire response must be a single JSON object. Do not include any text before or after it. Use the exact keys as specified in the schema below.
    
        **JSON Schema:**
        ```json
        {{
          "query_evaluation": {{
            "reasoning": "Provide a step-by-step analysis of the query rewrite here. Explain why it was good or bad.",
            "score": "Integer between 1 and 5. 1=made it worse, 3=no real change, 5=significant improvement.",
            "identified_issue": "Choose ONE: 'NONE', 'LOST_INTENT', 'TOO_BROAD', 'HALLUCINATED_DETAILS'"
          }},
          "answer_evaluation": {{
            "reasoning": "Provide a step-by-step analysis of the final answer's quality based on the context. Explain its correctness, relevance, and completeness.",
            "relevance_score": "Integer between 1 and 5.",
            "correctness_score": "Integer between 1 and 5.",
            "completeness_score": "Integer between 1 and 5.",
            "identified_issue": "Choose ONE: 'NONE', 'HALLUCINATION', 'INCOMPLETE', 'OUT_OF_CONTEXT'"
          }}
        }}
        ```
        """

    def judge_answer(self, original_query: str, rewritten_query: str, context: str, answer: str):
        """
        Evaluates both the query rewrite and the final answer, returning a nested JSON object.
        """
        prompt = self._create_judge_prompt(original_query, rewritten_query, context, answer)

        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json"
        )

        try:
            response = self.client.models.generate_content(
                model = self.judge_model,
                contents = prompt,
                config = generation_config,
            )

            evaluation = json.loads(response.text)
            return evaluation



        except Exception as e:
            print(f"An unexpected error occurred during judging: {e}")
            return {
                "error": str(e),
                "query_evaluation": {
                    "reasoning": "Failed to generate evaluation due to an API error.",
                    "score": 0, "identified_issue": "JUDGE_FAILURE"
                },
                "answer_evaluation": {
                    "reasoning": "Failed to generate evaluation due to an API error.",
                    "relevance_score": 0, "correctness_score": 0, "completeness_score": 0,
                    "identified_issue": "JUDGE_FAILURE"
                }
            }