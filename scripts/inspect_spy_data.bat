@echo off
echo Running Data File Inspector for SPY data...
python "%~dp0\inspect_data_file.py" "C:\Users\alexp\OneDrive\Gdrive\Trading\Data Outputs\SPY_Combined.csv" --start_date "2024-01-01" --end_date "2024-01-10" --date_column "DataDate"
pause 