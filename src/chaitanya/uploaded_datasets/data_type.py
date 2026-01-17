import os
import pandas as pd
from datetime import datetime

def get_latest_csv(directory):
    csv_files = [f for f in os.listdir(directory) if f.lower().endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the directory")

    # Get full paths
    full_paths = [os.path.join(directory, f) for f in csv_files]

    # Choose the latest modified CSV
    latest_file = max(full_paths, key=os.path.getmtime)
    return latest_file


def print_csv_metadata(csv_path):
    print(f"\n📄 Latest CSV File: {csv_path}")
    print("-" * 60)

    # File metadata
    file_size = os.path.getsize(csv_path) / 1024  # KB
    modified_time = datetime.fromtimestamp(os.path.getmtime(csv_path))

    print(f"File Size       : {file_size:.2f} KB")
    print(f"Last Modified   : {modified_time}")

    # Load CSV
    df = pd.read_csv(csv_path)

    print(f"Rows            : {df.shape[0]}")
    print(f"Columns         : {df.shape[1]}")
    print("\nColumn Metadata:")
    print("-" * 60)

    for col in df.columns:
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        unique = df[col].nunique()

        print(f"Column Name : {col}")
        print(f"  Data Type : {dtype}")
        print(f"  Nulls     : {nulls}")
        print(f"  Unique    : {unique}")
        print("-" * 40)


if __name__ == "__main__":
    directory_path = r"C:\\allenvs\\Agents\\Data-Visualizing-AI\\src\\chaitanya\\uploaded_datasets"   # change this
    latest_csv = get_latest_csv(directory_path)
    print_csv_metadata(latest_csv)
