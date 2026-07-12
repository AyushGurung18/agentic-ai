import socket

def check_port(host, port):
    print(f"🔍 Checking connectivity to {host}:{port}...")
    try:
        with socket.create_connection((host, port), timeout=10):
            print(f"✅ Port {port} is OPEN and REACHABLE!")
            return True
    except Exception as e:
        print(f"❌ Port {port} is CLOSED or UNREACHABLE: {e}")
        return False

if __name__ == "__main__":
    host = "db.lmpnnnfbfyclfwqwbbgd.supabase.co"
    check_port(host, 5432)
    check_port(host, 6543)
