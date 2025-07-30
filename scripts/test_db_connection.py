#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test database connection and basic operations"""
import psycopg2
import sys
from psycopg2 import OperationalError

def test_database_connection():
    """Test connection to PostgreSQL database"""
    # Connection parameters from docker-compose.yml
    conn_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'fantasy_football',
        'user': 'fantasy_user',
        'password': 'fantasy_pass'
    }
    
    try:
        # Attempt to connect
        print("Testing database connection...")
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        db_version = cursor.fetchone()
        print(f"✓ Successfully connected to PostgreSQL")
        print(f"  Database version: {db_version[0]}")
        
        # Check if tables exist
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            print(f"\n✓ Found {len(tables)} tables in the database:")
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
                count = cursor.fetchone()[0]
                print(f"  - {table[0]}: {count} rows")
        else:
            print("\n⚠ No tables found in the database. Database may need initialization.")
        
        cursor.close()
        conn.close()
        return True
        
    except OperationalError as e:
        print(f"✗ Failed to connect to database: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_database_connection()
    sys.exit(0 if success else 1)