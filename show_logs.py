#!/usr/bin/env python3
import sys

def extract_app_logs(log_file, lines=100):
    """Extract our application logs from the Streamlit log file"""
    try:
        with open(log_file, 'r') as f:
            content = f.readlines()
        
        # Filter only our application logs
        app_logs = [line for line in content if "main - INFO" in line or "main - ERROR" in line or
                   "analysis - INFO" in line or "analysis - ERROR" in line or
                   "utils - INFO" in line or "utils - ERROR" in line]
        
        # Return the last N lines
        return app_logs[-lines:]
    
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]

if __name__ == "__main__":
    log_file = ".streamlit/logs/app.log"
    lines = 100
    
    if len(sys.argv) > 1:
        try:
            lines = int(sys.argv[1])
        except:
            pass
    
    logs = extract_app_logs(log_file, lines)
    for log in logs:
        print(log.strip())