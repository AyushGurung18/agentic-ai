import psycopg
import os
from dotenv import load_dotenv
import sys

load_dotenv()
db_url = os.environ.get("DATABASE_URL")

if not db_url:
    print("❌ Error: DATABASE_URL not found in .env")
    sys.exit(1)

# Clean URL
clean_url = db_url.replace("postgresql+psycopg://", "postgresql://")

print(f"📡 Testing connection to: {clean_url.split('@')[-1] if '@' in clean_url else clean_url}")

try:
    print("⏳ Attempting to connect (timeout=15s)...")
    with psycopg.connect(clean_url, connect_timeout=15) as conn:
        print("✅ CONNECTION SUCCESSFUL!")
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print(f"📦 DB Version: {cur.fetchone()[0]}")
            
            cur.execute("SELECT installed_version FROM pg_available_extensions WHERE name = 'vector';")
            ext = cur.fetchone()
            if ext:
                print(f"🧬 Vector Extension: {ext[0] if ext[0] else 'Available but NOT installed'}")
            else:
                print("❌ Vector Extension: NOT available in this DB")
except Exception as e:
    print(f"❌ CONNECTION FAILED: {e}")
    print("\n💡 Troubleshooting Tips:")
    print("1. If 'Network is unreachable', ensure your network supports IPv6 OR use port 6543 in .env.")
    print("2. Check if your IP is allowed in Supabase Dashboard > Settings > Database > Network Restrictions.")
    print("3. Ensure the password is correct (and percent-encoded if it contains special characters).")
