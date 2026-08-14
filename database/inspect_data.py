from pathlib import Path
import pandas as pd


# Path to the folder containing our CSV files
DATA_DIR = Path("data/structured")


# Find all CSV files inside the structured data folder
csv_files = sorted(DATA_DIR.glob("*.csv"))


print("Olist Dataset Inspection")
print("=" * 50)

print(f"Number of CSV files found: {len(csv_files)}")
print()


for file in csv_files:

    # Load the current CSV file into a Pandas DataFrame
    df = pd.read_csv(file)

    # Display basic information about the dataset
    print(f"File: {file.name}")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    # Display column names and their data types
    print("\nColumn Names and Data Types:")

    for column in df.columns:
        print(f"  {column}: {df[column].dtype}")

    # Check for missing values
    print("\nMissing Values:")

    missing_values = df.isnull().sum()

    for column, missing_count in missing_values.items():
        if missing_count > 0:
            print(f"  {column}: {missing_count:,}")

    if missing_values.sum() == 0:
        print("  No missing values")

    # Check for completely duplicated rows
    duplicate_rows = df.duplicated().sum()

    print(f"\nDuplicate Rows: {duplicate_rows:,}")

    print()
    print("-" * 50)