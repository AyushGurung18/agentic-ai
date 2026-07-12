import os
import pathlib
import psycopg
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def migrate():
    """
    Manually apply the schema in init.sql to the Supabase database.
    """
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("❌ Error: DATABASE_URL not found in .env")
        return

    # Convert sqlalchemy-style URL to plain postgres if needed
    raw_url = database_url.replace("postgresql+psycopg://", "postgresql://")
    
    sql_path = pathlib.Path(__file__).parent / "init.sql"
    if not sql_path.exists():
        print(f"❌ Error: {sql_path} not found.")
        return

    sql = sql_path.read_text()

    print(f"🚀 Connecting to: {raw_url.split('@')[-1] if '@' in raw_url else 'unknown'}")
    try:
        # Use a longer timeout for remote connections
        with psycopg.connect(raw_url, connect_timeout=20) as conn:
            print("🔗 Connected successfully!")
            with conn.cursor() as cur:
                print("📝 Applying schema from init.sql...")
                cur.execute(sql)
            conn.commit()
            print("✅ Migration successful! Your Supabase database is now initialized.")
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        print("\n💡 Troubleshooting:")
        print("- Verify your network supports IPv6 or use Port 6543 in your .env URL.")
        print("- Check if your IP is whitelisted in Supabase Dashboard.")

if __name__ == "__main__":
    migrate()
