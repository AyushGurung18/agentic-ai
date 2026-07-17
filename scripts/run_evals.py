import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_precision,
    context_recall,
)
from datasets import Dataset

from app.services.langgraph_rag import build_rag_graph
from app.services.rag import get_llm_by_intent
from app.services.embeddings import embeddings as local_embeddings
from app.core.config import LANGCHAIN_TRACING_V2, NVIDIA_API_KEY, DEV_USER_ID

def evaluate_pipeline():
    if not LANGCHAIN_TRACING_V2:
        print("⚠️  WARNING: LANGCHAIN_TRACING_V2 is not enabled. Traces will not be sent to LangSmith dashboard.")
        
    testset_path = "testset.csv"
    if not os.path.exists(testset_path):
        print(f"Error: {testset_path} not found. Run generate_testset.py first.")
        return
        
    print(f"Loading {testset_path}...")
    df = pd.read_csv(testset_path)
    
    # We will populate these arrays by running our actual RAG pipeline
    answers = []
    contexts = []
    
    # Initialize our pipeline (default to complex model for evals)
    print("Initializing LangGraph pipeline...")
    llm = get_llm_by_intent("complex")
    graph = build_rag_graph(llm)
    
    print(f"Running pipeline on {len(df)} questions...")
    for i, row in df.iterrows():
        question = row["user_input"]
        print(f"[{i+1}/{len(df)}] Q: {question[:60]}...")
        
        # Run graph
        state = {
            "question": question,
            "original_q": question,
            "chat_history": [],
            "user_id": DEV_USER_ID,  # matches the user the sample docs were ingested under
            "session_id": "eval-session",
            "iterations": 0
        }
        
        try:
            result = graph.invoke(state)
            
            # Extract final generation and contexts used
            answer = result.get("generation", "Error generating answer.")
            retrieved_docs = result.get("documents", [])
            
            # Ragas expects contexts as a list of strings
            context_strings = [doc.page_content for doc in retrieved_docs]
            
        except Exception as e:
            print(f"  ❌ Error processing: {e}")
            answer = "Error"
            context_strings = []
            
        answers.append(answer)
        contexts.append(context_strings)
        
    # Prepare final dataset for Ragas (new schema: user_input/response/retrieved_contexts/reference)
    eval_dataset = Dataset.from_dict({
        "user_input": df["user_input"].tolist(),
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": df["reference"].tolist() if "reference" in df.columns else answers,  # fallback to generator answer
    })

    print("\nRunning Ragas evaluation metrics...")

    # NVIDIA-hosted Llama 3.1 70B as the judge LLM — avoids Gemini's restrictive
    # free-tier quota and Groq's account restriction. Embeddings reuse the
    # project's own local MiniLM model.
    judge_llm = LangchainLLMWrapper(ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    metrics = [
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall
    ]
    
    # Same throttle as testset generation — NVIDIA's free NIM tier rate-limits
    # hard, and ragas' default judge concurrency (16 workers) trips it immediately.
    judge_run_config = RunConfig(max_workers=1, max_retries=15, max_wait=90, timeout=300)

    results = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        run_config=judge_run_config,
        raise_exceptions=False
    )
    
    print("\n=== Evaluation Results ===")
    print(results)
    
    output_df = results.to_pandas()
    output_df.to_csv("evaluation_results.csv", index=False)
    print("\n✅ Saved detailed results to evaluation_results.csv")
    print("👉 Check your LangSmith dashboard to view the full traces!")

if __name__ == "__main__":
    evaluate_pipeline()
