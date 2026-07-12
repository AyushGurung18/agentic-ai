import psycopg
from app.core.config import POSTGRES_URL
from langchain_postgres import PostgresChatMessageHistory

def init_db():
    conn_info = POSTGRES_URL.replace("postgresql+psycopg://", "postgresql://")
    with psycopg.connect(conn_info) as conn:
        print("Connected to DB")
        # The constructor should create the table if we use the right method
        # or we can try to force it if langchain_postgres supports it.
        # Let's try to just instantiate it and see if it works.
        history = PostgresChatMessageHistory(
            "chat_history",
            "00000000-0000-0000-0000-000000000000",
            sync_connection=conn
        )
        print("Instantiated history")
        # Check if table exists
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.chat_history')")
            exists = cur.fetchone()[0]
            print(f"Table exists check: {exists}")
            if not exists:
                print("Table 'chat_history' does not exist. Attempting to create it manually...")
                # Note: This is the standard schema for PostgresChatMessageHistory
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chat_history (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        message JSONB NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                """)
                cur.execute("CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history (session_id);")
                print("Table created successfully")

if __name__ == "__main__":
    init_db()
