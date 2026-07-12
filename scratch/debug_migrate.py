import os
import psycopg
from dotenv import load_dotenv

load_dotenv()
db_url = os.environ.get("DATABASE_URL").replace("postgresql+psycopg://", "postgresql://")

print(f"🚀 Connecting to: {db_url.split('@')[1]}")
try:
    # Try connecting with a short timeout
    conn = psycopg.connect(db_url, connect_timeout=10)
    print("✅ Connection successful!")
    with conn.cursor() as cur:
        print("📝 Running init.sql...")
        with open("app/db/init.sql", "r") as f:
            sql = f.read()
        cur.execute(sql)
        conn.commit()
    conn.close()
    print("🎉 Database schema migrated successfully!")
except Exception as e:
    print(f"❌ Error: {e}")
