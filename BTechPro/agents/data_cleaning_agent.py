import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataCleaningAgent:
    """
    Handles missing values, duplicate removal, and encoding of categorical variables.
    Designed to work with any tabular dataset.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def remove_duplicates(self):
        """Remove duplicate rows from the dataset."""
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        after = len(self.df)
        print(f"Removed {before - after} duplicate rows.")
        return self.df

    def handle_missing_values(self, strategy="mean"):
        """Fill missing values with mean, median, or default values."""
        for col in self.df.columns:
            if self.df[col].dtype in ["int64", "float64"]:
                if strategy == "mean":
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == "median":
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                else:
                    self.df[col].fillna(0, inplace=True)
            else:
                self.df[col].fillna("Unknown", inplace=True)
        return self.df

    def encode_categorical(self):
        """Encode all categorical columns using LabelEncoder."""
        le = LabelEncoder()
        for col in self.df.columns:
            if self.df[col].dtype == "object":
                self.df[col] = le.fit_transform(self.df[col].astype(str))
        return self.df

    def run(self):
        """Execute the full cleaning pipeline."""
        self.remove_duplicates()
        self.handle_missing_values()
        self.encode_categorical()
        print("Data cleaning complete.")
        return self.df
