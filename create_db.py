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

if __name__ == "__main__":
    create_history_table()
