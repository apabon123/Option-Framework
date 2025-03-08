import pandas as pd
import os

def test_load_csv():
    """Test loading the CSV file directly."""
    file_path = "test_data.csv"
    
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