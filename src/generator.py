import json
import os
from litellm import completion

def generate_reply(email_text: str, context: str, model: str = "gpt-4o-mini") -> str:
    """
    Generates a suggested email reply based on the incoming email and provided context.
    """
    system_prompt = (
        "You are an expert customer support agent. "
        "Your goal is to draft a polite, professional, and helpful email reply "
        "based on the incoming customer email and the internal context provided. "
        "Adhere to the context strictly. Do not make up company policies."
    )
    
    user_prompt = f"Incoming Email:\n{email_text}\n\nInternal Context:\n{context}\n\nPlease draft the reply:"
    
    try:
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during generation: {e}")
        return "ERROR_IN_GENERATION"
