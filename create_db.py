import sqlite3

def create_history_table(db_path="stock.db"):
    """
    Creates the 'conversation_history' table in the SQLite database
    if it does not already exist.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Drop the table if it exists to ensure a clean slate
        cursor.execute("DROP TABLE IF EXISTS conversation_history")
        cursor.execute("""
            CREATE TABLE conversation_history (
                session_id TEXT PRIMARY KEY,
                messages TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS session_id_timestamp_idx ON conversation_history (session_id, timestamp);")
        conn.commit()
        print("Table 'conversation_history' created or already exists.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def recreate_table():
    """
    Drops the existing 'stock_index_price_daily' table and recreates it
    with a composite primary key to enforce uniqueness.
    """
    db_path = "stock.db"
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Drop the existing table if it exists
        cursor.execute("DROP TABLE IF EXISTS stock_index_price_daily")
        print("Dropped existing 'stock_index_price_daily' table.")

        # Recreate the table with the composite primary key
        cursor.execute("""
            CREATE TABLE stock_index_price_daily (
                index_name TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                date_key TEXT,
                PRIMARY KEY (index_name, date_key)
            )
        """)
        print("Recreated 'stock_index_price_daily' table with composite primary key.")

        conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    create_history_table()
    recreate_table()
