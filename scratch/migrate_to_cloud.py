import psycopg
import sys

import os
from dotenv import load_dotenv

load_dotenv()

# Connection details for Direct Connection
DB_URL = os.environ.get("DATABASE_URL", "").replace("postgresql+psycopg://", "postgresql://")
SQL_FILE = os.path.join(os.path.dirname(__file__), "..", "app", "db", "init.sql")

def migrate():
    print(f"🚀 Attempting direct connection to Supabase (Port 5432)...")
    try:
        # We use psycopg.connect which handles IPv6/IPv4 automatically
        with psycopg.connect(DB_URL, autocommit=True) as conn:
            print("🔗 Connected successfully!")
            with conn.cursor() as cur:
                print(f"📖 Reading schema from {SQL_FILE}...")
                with open(SQL_FILE, "r") as f:
                    sql = f.read()
                
                print("⚡ Executing migration statements...")
                cur.execute(sql)
                print("✅ Cloud migration successful! Your tables are now ready on Supabase.")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("\nPossible reasons:")
        print("1. Network: If it times out, your internet might not support IPv6 (required for port 5432).")
        print("2. Password: Check if 'ABSSRRVVg18@' is correct.")
        sys.exit(1)

if __name__ == "__main__":
    migrate()
