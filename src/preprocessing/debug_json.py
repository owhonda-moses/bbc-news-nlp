import json
import sys
import os

def main():
    """
    Validates a JSON file and reports the location of any syntax error.
    """
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing.debug_json <path_to_json_file>")
        return

    filepath = sys.argv[1]

    if not os.path.exists(filepath):
        print(f"File not found at '{filepath}'")
        return

    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"{filepath} is a valid JSON file.")
    except json.JSONDecodeError as e:
        print(f"--- Invalid JSON Syntax Found ---")
        print(f"File:    '{filepath}'")
        print(f"Error:   {e.msg}")
        print(f"Line:    {e.lineno}")
        print(f"Column:  {e.colno}")
        
        # print the problematic line
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if e.lineno <= len(lines):
                error_line = lines[e.lineno - 1].rstrip()
                print(f"\nContext:\n> {error_line}")
                pointer = ' ' * (e.colno - 1) + '^' # pointer
                print(f"  {pointer}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()