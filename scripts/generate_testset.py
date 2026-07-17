import sys
import os
import json
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from ragas.testset import TestsetGenerator
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

from app.db.database import get_conn
from app.services.embeddings import embeddings as local_embeddings
from app.core.config import NVIDIA_API_KEY

def fetch_document_chunks(limit=20):
    """Fetch parent chunks from the DB to generate questions from."""
    docs = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            # We fetch parent chunks (where parent_chunk_id is null) because they contain more context
            # which is better for generating high-quality test questions.
            cur.execute(
                """
                SELECT id, document_id, content
                FROM document_chunks
                WHERE parent_chunk_id IS NULL
                ORDER BY RANDOM()
                LIMIT %s
                """,
                (limit,)
            )
            rows = cur.fetchall()
            for chunk_id, doc_id, content in rows:
                # Need sufficient text to generate questions
                if len(content.split()) > 50:
                    docs.append(Document(
                        page_content=content,
                        metadata={"source_document_id": str(doc_id), "chunk_id": str(chunk_id)}
                    ))
    return docs

def generate_testset():
    print("Fetching documents from database...")
    docs = fetch_document_chunks(limit=12)

    if not docs:
        print("No parent chunks found in database. Please upload a PDF first.")
        return

    print(f"Loaded {len(docs)} documents. Initializing Ragas TestsetGenerator...")

    # NVIDIA-hosted Llama 3.1 70B for testset generation — avoids Gemini's
    # restrictive free-tier quota (20 req/day) and Groq's account restriction.
    # Embeddings reuse the project's own local MiniLM model.
    generator_llm = LangchainLLMWrapper(ChatNVIDIA(model="meta/llama-3.1-70b-instruct", nvidia_api_key=NVIDIA_API_KEY, temperature=0.7))
    generator_embeddings = LangchainEmbeddingsWrapper(local_embeddings)

    generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings)

    # Low concurrency + generous retry/backoff — NVIDIA's free NIM tier rate-limits hard.
    run_config = RunConfig(max_workers=1, max_retries=15, max_wait=90, timeout=300)

    test_size = 6
    print(f"Generating {test_size} synthetic Q&A pairs (throttled, this may take a while)...")

    testset = generator.generate_with_langchain_docs(docs, testset_size=test_size, run_config=run_config)

    df = testset.to_pandas()

    output_file = "testset.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Successfully generated testset and saved to {output_file}!")
    print("Preview of generated questions:")
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. {row['user_input']}")
        
if __name__ == "__main__":
    generate_testset()
