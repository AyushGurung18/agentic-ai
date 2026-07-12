import sys
import os
import json
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context

from app.db.database import get_conn

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
    docs = fetch_document_chunks(limit=30)
    
    if not docs:
        print("No parent chunks found in database. Please upload a PDF first.")
        return
        
    print(f"Loaded {len(docs)} documents. Initializing Ragas TestsetGenerator...")
    
    # We use Gemini 1.5 Flash for fast/cheap testset generation.
    # For higher quality questions, swap this to Llama 70B or Gemini Pro.
    generator_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)
    critic_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm,
        critic_llm=critic_llm,
        embeddings=embeddings
    )
    
    # Define what types of questions we want
    distributions = {
        simple: 0.5,
        reasoning: 0.25,
        multi_context: 0.25
    }
    
    test_size = 10
    print(f"Generating {test_size} synthetic Q&A pairs (this may take a few minutes)...")
    
    testset = generator.generate_with_langchain_docs(
        docs, 
        test_size=test_size, 
        distributions=distributions,
        # set is_async to False if running into rate limits
        is_async=True
    )
    
    df = testset.to_pandas()
    
    output_file = "testset.csv"
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Successfully generated testset and saved to {output_file}!")
    print("Preview of generated questions:")
    for i, row in df.head(3).iterrows():
        print(f"{i+1}. {row['question']}")
        
if __name__ == "__main__":
    generate_testset()
