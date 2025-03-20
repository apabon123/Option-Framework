import pandas as pd
import os
import sys
from pathlib import Path

# Get the absolute path of the script and determine project root
script_path = Path(os.path.abspath(__file__))
project_root = script_path.parent.parent.parent
# Add project root to sys.path
sys.path.insert(0, str(project_root))

def test_load_csv():
    """Test loading the CSV file directly."""
    # Update the file path to use the data directory
    file_path = os.path.join(project_root, "tests", "data", "test_data.csv")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"File exists: {os.path.exists(file_path)}")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded CSV with {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        return True
    except Exception as e:
        print(f"Error loading CSV: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    test_load_csv() 