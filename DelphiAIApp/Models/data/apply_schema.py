"""Apply database schema to PostgreSQL"""
import psycopg2
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
env_path = Path(__file__).parent.parent.parent.parent / '.env'
load_dotenv(env_path)

# Database connection settings
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5433'),
    'dbname': os.getenv('DB_NAME', 'delphi_db'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', ''),
}

def main():
    # Read schema file
    schema_path = Path(__file__).parent.parent / 'db' / 'schemas.sql'
    
    print(f"Reading schema from: {schema_path}")
    with open(schema_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    # Connect
    print(f"Connecting to {DB_CONFIG['dbname']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}...")
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True  # Needed for DROP/CREATE
    cursor = conn.cursor()
    
    # Execute schema
    print('Applying schema (this will DROP and recreate all tables)...')
    try:
        cursor.execute(schema_sql)
        print('[OK] Schema applied successfully!')
    except Exception as e:
        print(f'[ERROR] {e}')
        conn.close()
        return
    
    # Verify tables exist
    cursor.execute("""
        SELECT table_name FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_type = 'BASE TABLE'
        ORDER BY table_name
    """)
    tables = cursor.fetchall()
    print('\nTables created:')
    for t in tables:
        print(f'  - {t[0]}')
    
    conn.close()
    print('\n[OK] Done!')

if __name__ == '__main__':
    main()
