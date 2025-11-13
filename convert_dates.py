import sqlite3
from datetime import datetime

def convert_and_update_date_format(db_path="stock.db", table_name="stock_index_price"):
    """
    Reads data from a SQLite database, converts the 'date' column to YYYY-MM-DD format,
    and updates a new 'date_key' column with the converted dates.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Add 'date_key' column if it doesn't exist
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = [col[1] for col in cursor.fetchall()]
        if "date_key" not in columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN date_key TEXT;")
            print(f"Added 'date_key' column to {table_name}.")
        else:
            print(f"'date_key' column already exists in {table_name}.")

        # Fetch all rows with original date and primary key (assuming 'index_name' and 'date' form a composite primary key)
        # If there's a single primary key, adjust the SELECT and UPDATE WHERE clause accordingly.
        # For 'stock_index_price', it seems 'index_name' and 'date' are the primary keys.
        cursor.execute(f"SELECT index_name, date FROM {table_name}")
        rows = cursor.fetchall()

        print(f"Converting and updating dates in table: {table_name}")
        print("-" * 30)

        for index_name, original_date_str in rows:
            converted_date_str = None

            # Try parsing with 'DD Mon YYYY' format (e.g., '30 May 2022')
            try:
                dt_object = datetime.strptime(original_date_str, '%d %b %Y')
                converted_date_str = dt_object.strftime('%Y-%m-%d')
            except ValueError:
                # If first format fails, try 'DD-Mon-YY' format (e.g., '01-Nov-22')
                try:
                    dt_object = datetime.strptime(original_date_str, '%d-%b-%y')
                    converted_date_str = dt_object.strftime('%Y-%m-%d')
                except ValueError:
                    print(f"Could not parse date: {original_date_str} for {index_name}")

            if converted_date_str:
                cursor.execute(
                    f"UPDATE {table_name} SET date_key = ? WHERE index_name = ? AND date = ?",
                    (converted_date_str, index_name, original_date_str)
                )
                print(f"Updated: {index_name}, Original Date: {original_date_str}, Converted Date_Key: {converted_date_str}")
            else:
                print(f"Skipped update for {index_name}, Original Date: {original_date_str} (Parse Error)")
        
        conn.commit()
        print("\nDate conversion and update complete.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    convert_and_update_date_format()
