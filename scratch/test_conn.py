import psycopg
import os
from dotenv import load_dotenv

load_dotenv()
db_url = os.environ.get("DATABASE_URL").replace("postgresql+psycopg://", "postgresql://")

print(f"Attempting to connect to: {db_url.split('@')[1]}")
try:
    with psycopg.connect(db_url, connect_timeout=5) as conn:
        print("✅ Connection successful!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
