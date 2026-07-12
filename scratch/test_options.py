import psycopg
import sys

# Testing with options parameter to specify tenant
DB_URL = "postgresql://postgres:g03S68otDmei89J8@aws-0-ap-southeast-2.pooler.supabase.com:6543/postgres?sslmode=require&options=-c%20project=lmpnnnfbfyclfwqwbbgd"

print(f"📡 Testing Pooler with options (Sydney)...")
try:
    with psycopg.connect(DB_URL, connect_timeout=10) as conn:
        print("✅ CONNECTION SUCCESSFUL!")
except Exception as e:
    print(f"❌ FAILED: {e}")
