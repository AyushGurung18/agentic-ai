import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

_RAW_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://ayush:password@localhost:5432/octo",
).replace("postgresql+psycopg://", "postgresql://")

def truncate_all():
    print("🧹 Starting full database truncate...")
    try:
        with psycopg.connect(_RAW_URL) as conn:
            with conn.cursor() as cur:
                # Disable triggers to speed up and avoid FK issues if needed, 
                # but TRUNCATE CASCADE is cleaner.
                cur.execute("""
                    TRUNCATE TABLE 
                        users, 
                        sessions, 
                        messages, 
                        documents, 
                        document_chunks, 
                        embeddings, 
                        document_metadata 
                    RESTART IDENTITY CASCADE;
                """)
                
                # Also clean up the LangChain PGVector tables if they exist
                try:
                    cur.execute("TRUNCATE TABLE langchain_pg_embedding, langchain_pg_collection RESTART IDENTITY CASCADE;")
                    print("✅ Cleaned LangChain PGVector tables.")
                except Exception:
                    print("ℹ️ LangChain PGVector tables not found or already clean.")
                
            conn.commit()
        print("✨ All tables truncated successfully.")
    except Exception as e:
        print(f"❌ Truncate failed: {e}")

if __name__ == "__main__":
    truncate_all()
