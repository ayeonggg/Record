import subprocess
import sqlite3
from datetime import datetime

DB_PATH = '/tmp/tool_data.db'

def create_table_if_not_exists():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS powerstat_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            output TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS powertop_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            usage TEXT,
            wakeups TEXT,
            category TEXT,
            details TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS perf_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            output TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS valgrind_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            output TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_powerstat_data(output):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO powerstat_data (output)
        VALUES (?)
    ''', (output,))
    conn.commit()
    conn.close()

def insert_powertop_data(usage, wakeups, category, details):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO powertop_data (usage, wakeups, category, details)
        VALUES (?, ?, ?, ?)
    ''', (usage, wakeups, category, details))
    conn.commit()
    conn.close()

def insert_perf_data(output):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO perf_data (output)
        VALUES (?)
    ''', (output,))
    conn.commit()
    conn.close()

def insert_valgrind_data(output):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO valgrind_data (output)
        VALUES (?)
    ''', (output,))
    conn.commit()
    conn.close()

def run_powerstat():
    try:
        result = subprocess.run(['sudo', 'powerstat', '-n', '100', '-z'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("Error executing powerstat:", result.stderr)
            return
        
        output = result.stdout
        
        if not output:
            print("No output from powerstat.")
            return
        
        print("Powerstat Output:")
        print(output)
        
        insert_powerstat_data(output)
        
    except Exception as e:
        print("An error occurred: {}".format(e))

def run_powertop():
    try:
        result = subprocess.run(['sudo', 'powertop', '--html=/tmp/powertop.html'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("Error executing powertop:", result.stderr)
            return
        
        with open('/tmp/powertop.html', 'r') as file:
            output = file.read()
        
        if not output:
            print("No output from powertop.")
            return

        print("Powertop Output:")
        print(output)
        
        # Parse and insert powertop data
        # Parsing and insertion logic as per your requirement
        # Example:
        # insert_powertop_data(usage, wakeups, category, details)

    except Exception as e:
        print("An error occurred: {}".format(e))

def run_perf():
    try:
        result = subprocess.run(['sudo', 'perf', 'stat', 'sleep', '5'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("Error executing perf:", result.stderr)
            return
        
        output = result.stdout
        
        if not output:
            print("No output from perf.")
            return
        
        print("Perf Output:")
        print(output)
        
        insert_perf_data(output)
        
    except Exception as e:
        print("An error occurred: {}".format(e))

def run_valgrind():
    try:
        result = subprocess.run(['valgrind', '--tool=callgrind', '--trace-children=yes', 'sleep', '5'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.returncode != 0:
            print("Error executing valgrind:", result.stderr)
            return
        
        output = result.stdout
        
        if not output:
            print("No output from valgrind.")
            return
        
        print("Valgrind Output:")
        print(output)
        
        insert_valgrind_data(output)
        
    except Exception as e:
        print("An error occurred: {}".format(e))

if __name__ == "__main__":
    create_table_if_not_exists()
    run_powerstat()
    run_powertop()
    run_perf()
    run_valgrind()
