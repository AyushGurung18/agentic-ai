import psycopg
import sys

# Testing NEW pooler host + port 5432 (Session Mode)
# This is the modern Supabase recommendation for IPv4
DB_URL = "postgresql://postgres.lmpnnnfbfyclfwqwbbgd:g03S68otDmei89J8@aws-0-ap-southeast-2.pooler.supabase.com:6543/postgres?sslmode=require"

print(f"📡 Testing IPv4 Pooler (Sydney)...")
try:
    with psycopg.connect(DB_URL, connect_timeout=10) as conn:
        print("✅ CONNECTION SUCCESSFUL!")
except Exception as e:
    print(f"❌ FAILED: {e}")
