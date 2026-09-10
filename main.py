import json
import os
from dotenv import load_dotenv
from src.generator import generate_reply
from src.evaluator import evaluate_reply

# Load environment variables (API Keys)
load_dotenv()

def main():
    print("Starting AI Email Suggested-Response System...")
    
    # Check for API keys
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: No OPENAI_API_KEY or GEMINI_API_KEY found in environment. Please set one in a .env file.")
        return

    # Load dataset
    dataset_path = os.path.join("dataset", "data.json")
    with open(dataset_path, "r") as f:
        data = json.load(f)
    
    results = []
    total_score = 0.0
    
    # Process each email
    for item in data:
        print(f"\nProcessing Email ID: {item['id']}")
        
        # 1. Generate Reply
        # Using a default litellm supported model. Change as needed.
        # Ensure litellm is configured to use the available key.
        model_name = "gpt-4o-mini" if os.environ.get("OPENAI_API_KEY") else "gemini/gemini-1.5-flash"
        eval_model_name = "gpt-4o" if os.environ.get("OPENAI_API_KEY") else "gemini/gemini-1.5-pro"
        
        print(f"Generating reply using {model_name}...")
        generated = generate_reply(item["incoming_email"], item["context"], model=model_name)
        
        # 2. Evaluate Reply
        print(f"Evaluating reply using {eval_model_name}...")
        evaluation = evaluate_reply(
            incoming_email=item["incoming_email"],
            context=item["context"],
            generated_reply=generated,
            reference_reply=item["reference_reply"],
            model=eval_model_name
        )
        
        print(f"Evaluation Score: {evaluation.get('overall_score', 0)}")
        
        # 3. Store Results
        result_item = {
            "id": item["id"],
            "incoming_email": item["incoming_email"],
            "generated_reply": generated,
            "evaluation": evaluation
        }
        results.append(result_item)
        total_score += evaluation.get("overall_score", 0)
        
    # Aggregate and Save
    avg_system_score = total_score / len(data) if data else 0
    final_report = {
        "average_system_score": round(avg_system_score, 2),
        "total_emails_processed": len(data),
        "detailed_results": results
    }
    
    results_path = os.path.join("results", "evaluation_report.json")
    with open(results_path, "w") as f:
        json.dump(final_report, f, indent=4)
        
    print(f"\nProcessing Complete. Average System Score: {avg_system_score:.2f}/5.0")
    print(f"Detailed results saved to {results_path}")

if __name__ == "__main__":
    main()
