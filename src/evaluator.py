import json
from pydantic import BaseModel, Field
from litellm import completion

class EvaluationResult(BaseModel):
    relevance_score: int = Field(..., ge=1, le=5, description="Score from 1-5 on how relevant the reply is to the incoming email.")
    relevance_reason: str = Field(..., description="Reasoning for the relevance score.")
    tone_score: int = Field(..., ge=1, le=5, description="Score from 1-5 on the professionalism and empathy of the tone.")
    tone_reason: str = Field(..., description="Reasoning for the tone score.")
    completeness_score: int = Field(..., ge=1, le=5, description="Score from 1-5 on whether the reply includes all required context.")
    completeness_reason: str = Field(..., description="Reasoning for the completeness score.")
    overall_score: float = Field(..., description="Average of the three scores.")

def evaluate_reply(incoming_email: str, context: str, generated_reply: str, reference_reply: str, model: str = "gpt-4o") -> dict:
    """
    Evaluates a generated reply using an LLM-as-a-judge approach.
    """
    system_prompt = (
        "You are an expert Quality Assurance manager for a customer support team. "
        "Your task is to evaluate a generated email reply against an incoming email and internal context. "
        "You will score the generated reply on Relevance, Tone, and Completeness on a scale of 1-5. "
        "You must return your evaluation strictly as a JSON object matching the requested schema."
    )
    
    user_prompt = f"""
Incoming Email:
{incoming_email}

Internal Context / Ground Truth:
{context}

Reference Reply (for comparison):
{reference_reply}

Generated Reply to Evaluate:
{generated_reply}

Evaluate the Generated Reply.
"""

    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        # Parse the JSON string from the model
        content = response.choices[0].message.content
        eval_dict = json.loads(content)
        
        # Calculate overall score if not provided correctly by the LLM
        rel = eval_dict.get("relevance_score", 0)
        tone = eval_dict.get("tone_score", 0)
        comp = eval_dict.get("completeness_score", 0)
        eval_dict["overall_score"] = round((rel + tone + comp) / 3.0, 2)
        
        return eval_dict
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return {
            "error": str(e),
            "relevance_score": 0,
            "tone_score": 0,
            "completeness_score": 0,
            "overall_score": 0.0
        }
