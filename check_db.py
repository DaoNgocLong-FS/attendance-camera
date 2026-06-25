import sqlite3
import os

if os.path.exists('attendance.db'):
    conn = sqlite3.connect('attendance.db')
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]
    print(f"Tables: {tables}")
    conn.close()
else:
    print("Database file does not exist")
