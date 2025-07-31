#!/usr/bin/env python3
"""
Test and fix database connection issues
"""
import sys
import os
sys.path.append('/Users/christopherbratkovics/Desktop/fantasy-football-ai')

from dotenv import load_dotenv
load_dotenv()

def test_database_connection():
    """Test database connection with detailed debugging"""
    print("🔍 Debugging Database Connection")
    print("="*50)
    
    # Check environment variable
    db_url = os.getenv("DATABASE_URL")
    print(f"DATABASE_URL: {db_url[:50]}..." if db_url else "Not set")
    
    if not db_url:
        print("❌ DATABASE_URL not set")
        return False
    
    # Parse URL to check credentials
    from urllib.parse import urlparse
    parsed = urlparse(db_url)
    print(f"Host: {parsed.hostname}")
    print(f"Port: {parsed.port}")
    print(f"Database: {parsed.path[1:] if parsed.path else 'None'}")
    print(f"Username: {parsed.username}")
    print(f"Password: {'***' if parsed.password else 'None'}")
    
    try:
        # Test direct connection
        import psycopg2
        print("\n🔗 Testing direct psycopg2 connection...")
        
        conn = psycopg2.connect(db_url)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ Direct connection successful: {version[:50]}...")
        cursor.close()
        conn.close()
        
        # Test SQLAlchemy connection
        print("\n🔗 Testing SQLAlchemy connection...")
        from sqlalchemy import create_engine, text
        engine = create_engine(db_url)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test")).fetchone()
            print(f"✅ SQLAlchemy connection successful: {result[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        
        # Try alternative approaches
        print("\n🔧 Trying alternative connection methods...")
        
        # Check if it's a local PostgreSQL issue
        try:
            # Try connecting to localhost with default postgres user
            test_conn = psycopg2.connect(
                host="localhost",
                port=5432,
                database="postgres",
                user="postgres",
                password=""
            )
            test_conn.close()
            print("✅ Local PostgreSQL is accessible")
        except:
            print("❌ Local PostgreSQL not accessible - this is expected for Supabase setup")
        
        return False

def test_redis_docker():
    """Test if Redis via Docker is working"""
    print("\n🔍 Testing Redis Connection")
    print("="*50)
    
    try:
        import redis
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        print(f"Redis URL: {redis_url}")
        
        r = redis.from_url(redis_url)
        r.ping()
        print("✅ Redis connection successful")
        
        # Test operations
        r.set("test_key", "test_value", ex=60)
        value = r.get("test_key")
        r.delete("test_key")
        print(f"✅ Redis operations successful: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Database Connection Diagnostic Test")
    
    db_success = test_database_connection()
    redis_success = test_redis_docker()
    
    print(f"\n📊 Results:")
    print(f"Database: {'✅ PASS' if db_success else '❌ FAIL'}")
    print(f"Redis: {'✅ PASS' if redis_success else '❌ FAIL'}")
    
    if not db_success:
        print("\n💡 Database Troubleshooting Tips:")
        print("1. Verify Supabase project is active")
        print("2. Check database credentials in Supabase dashboard")
        print("3. Ensure IP is whitelisted in Supabase")
        print("4. Test connection URL in a PostgreSQL client")
    
    if not redis_success:
        print("\n💡 Redis Troubleshooting Tips:")
        print("1. Start Redis with: docker run -d -p 6379:6379 redis:latest")
        print("2. Check if Redis container is running: docker ps")
        print("3. Test connection with: redis-cli ping")