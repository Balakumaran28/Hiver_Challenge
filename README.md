# AI Email Suggested-Response System

Built for the Hiver Challenge by [Balakumaran28](https://github.com/Balakumaran28).

## Overview
This repository contains a full end-to-end system for generating suggested email replies using Generative AI and, most importantly, evaluating the quality of those replies using an LLM-as-a-judge architecture.

## 1. The Dataset
The dataset (`dataset/data.json`) is a synthetically generated set of B2B SaaS customer support emails. 
**Why this approach?** Synthetic data allows us to tightly control the "Ground Truth" (the `context` field). In a real-world system, determining if a reply is "accurate" is impossible without knowing the internal company policy or customer state at the time of the email. By embedding this in the dataset, we provide a concrete baseline for the evaluator.

## 2. The Generator
The generator (`src/generator.py`) uses a standard prompting architecture via `litellm`. 
**Trade-offs:** We opted for zero-shot/few-shot prompting with explicit context injection (simulating a RAG retrieval step) rather than fine-tuning. For email generation, context matters more than learned style weights. Prompting allows dynamic context (like current billing state) to be easily injected without retraining.

## 3. The Evaluator (Accuracy System)
The evaluator (`src/evaluator.py`) is the core of this system. We use an **LLM-as-a-judge** approach.

### What does "Accurate" mean?
An exact string match to a historical reply is a terrible metric. A human might say "Hi", the AI might say "Hello". Both are 100% accurate. 
Therefore, accuracy is defined by three dimensions:
1. **Relevance (1-5):** Does it answer the specific questions asked?
2. **Completeness (1-5):** Does it include all necessary information from the internal context?
3. **Tone (1-5):** Is it professional and empathetic?

The system prompts a highly capable model (like GPT-4o or Gemini 1.5 Pro) to score the generated reply on these axes and, crucially, **provide reasoning**. The reasoning ensures the metric reflects real quality and isn't just an arbitrary number. The overall score is an average of the three.

## How to Run

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the root directory and add an API key. The system uses `litellm`, so you can use OpenAI or Gemini:
   ```env
   OPENAI_API_KEY="your-sk-key"
   # OR
   GEMINI_API_KEY="your-gemini-key"
   ```
4. Run the pipeline:
   ```bash
   python main.py
   ```
5. Check the `results/evaluation_report.json` for per-response scores, reasoning, and the overall system score.

## AI Tools Used
- `litellm` was used to wrap LLM API calls.
- AI (Gemini 3.1 Pro) was used during development to quickly scaffold the dataset and JSON schemas.
