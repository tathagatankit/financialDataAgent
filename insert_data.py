import sqlite3
import pandas as pd
from datetime import datetime
import os

def insert_stock_data_from_file(db_path, csv_path, cursor):
    """
    Reads stock data from a single CSV file and inserts it into the 'stock_index_price_daily' table.
    """
    try:
        # Read CSV data
        df = pd.read_csv(csv_path)

        # Prepare data for insertion
        data_to_insert = []
        for _, row in df.iterrows():
            # Convert date format
            try:
                date_obj = datetime.strptime(row['Date'], '%d %b %Y')
                date_key = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                print(f"Could not parse date: {row['Date']} in {os.path.basename(csv_path)}. Skipping row.")
                continue

            data_to_insert.append((
                row['Index Name'],
                row['Open'],
                row['High'],
                row['Low'],
                row['Close'],
                date_key
            ))

        # Insert data into the table
        cursor.executemany("""
            INSERT INTO stock_index_price_daily (index_name, open, high, low, close, date_key)
            VALUES (?, ?, ?, ?, ?, ?)
        """, data_to_insert)
        print(f"Successfully inserted {len(data_to_insert)} rows from {os.path.basename(csv_path)}.")
    except FileNotFoundError:
        print(f"Error: The file '{csv_path}' was not found.")
    except Exception as e:
        print(f"An error occurred while processing '{csv_path}': {e}")
        
def insert_stock_data_from_directory(db_path, directory_path):
    """
    Reads stock data from all CSV files in a directory and inserts it into the 'stock_index_price_daily' table.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_index_price_daily (
                index_name TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                date_key TEXT
            )
        """)

        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                csv_path = os.path.join(directory_path, filename)
                insert_stock_data_from_file(db_path, csv_path, cursor)

        conn.commit()

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    db_path = "stock.db"
    directory_path = "Market Data/NiftyBank"
    insert_stock_data_from_directory(db_path, directory_path)
