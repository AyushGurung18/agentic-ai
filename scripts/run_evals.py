import sys
import os
import pandas as pd
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics import (
    answer_relevance,
    faithfulness,
    context_precision,
    context_recall,
)
from datasets import Dataset

from app.services.langgraph_rag import build_rag_graph
from app.services.rag import get_llm_by_intent
from app.core.config import LANGCHAIN_TRACING_V2

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
    graph = build_rag_graph(llm).compile()
    
    print(f"Running pipeline on {len(df)} questions...")
    for i, row in df.iterrows():
        question = row["question"]
        print(f"[{i+1}/{len(df)}] Q: {question[:60]}...")
        
        # Run graph
        state = {
            "question": question,
            "original_q": question,
            "chat_history": [],
            "user_id": "00000000-0000-0000-0000-000000000000", # System user
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
        
    # Prepare final dataset for Ragas
    eval_dataset = Dataset.from_dict({
        "question": df["question"].tolist(),
        "answer": answers,
        "contexts": contexts,
        "ground_truth": df["ground_truth"].tolist() if "ground_truth" in df.columns else df["answer"].tolist() # fallback to generator answer
    })
    
    print("\nRunning Ragas evaluation metrics...")
    
    # We use Gemini as the judge LLM for Ragas
    judge_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    judge_embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    metrics = [
        answer_relevance,
        faithfulness,
        context_precision,
        context_recall
    ]
    
    results = evaluate(
        eval_dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
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
