import subprocess
from bs4 import BeautifulSoup
import sqlite3
from datetime import datetime
import psutil
import re


DB_PATH = '/tmp/powertop_data.db'

def create_table_if_not_exists():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS powertop_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            usage TEXT,
            wakeups TEXT,
            category TEXT,
            details TEXT,
            cpu_usage TEXT
        )
    ''')
    conn.commit()
    conn.close()

def insert_data(usage, wakeups, category, details, cpu_usage):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO powertop_data (usage, wakeups, category, details, cpu_usage)
        VALUES (?, ?, ?, ?, ?)
    ''', (usage, wakeups, category, details, cpu_usage))
    conn.commit()
    conn.close()

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

        soup = BeautifulSoup(output, 'html.parser')
        process_section = soup.find_all('tr')
        
        if not process_section:
            print("No 'Process' section found.")
            return
        
        print("{:<10} {:<10} {:<15} {:<30} {:<10}".format('Usage', 'Wakeups/s', 'Category', 'Details', 'CPU Usage'))
        print('-' * 75)
        
        for row in process_section:
            cols = row.find_all('td')
            if len(cols) == 4:
                usage = cols[0].text.strip()
                wakeups = cols[1].text.strip()
                category = cols[2].text.strip()
                details = cols[3].text.strip()
                
                # Measure CPU usage for processes with matching details
                cpu_usage = get_cpu_usage_from_details(details)
                
                print("{:<10} {:<10} {:<15} {:<30} {:<10}".format(usage, wakeups, category, details, cpu_usage))
                insert_data(usage, wakeups, category, details, cpu_usage)

    except Exception as e:
        print("An error occurred: {}".format(e))

def get_cpu_usage_from_details(details):
    # Extract PID from details, if present
    pid_match = re.search(r'\[PID (\d+)\]', details)
    if pid_match:
        pid = int(pid_match.group(1))
        try:
            proc = psutil.Process(pid)
            # Allow some time for CPU percent calculation
            proc.cpu_percent(interval=1)
            return f"{proc.cpu_percent(interval=1)}%"
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return "N/A"
    return "N/A"

if __name__ == "__main__":
    create_table_if_not_exists()
    run_powertop()
