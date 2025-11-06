import pandas as pd
import numpy as np
import re 
import unicodedata
import io
from typing import Union

class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()

    def remove_duplicates(self):
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        return self

    def trim_spaces(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].astype(str).str.strip()
        return self

    def fix_capitalization(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].astype(str).str.lower().str.title()
        return self

    def unify_date_formats(self, date_format='%Y-%m-%d'):
        for col in self.df.columns:
            if self.df[col].dtype in ['object', 'string']:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='ignore').dt.strftime(date_format)
                except Exception:
                    continue
        return self

    def handle_missing_values(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                self.df[col] = self.df[col].fillna("Unknown")
            else:
                self.df[col] = self.df[col].fillna(0)
        return self

    def correct_data_types(self):
        for col in self.df.columns:
            try:
                if self.df[col].dtype == 'object':
                    if self.df[col].str.match(r'^\d+(\.\d+)?$').all():
                        self.df[col] = self.df[col].astype(float)
                elif self.df[col].dtype == 'float' and (self.df[col] % 1 == 0).all():
                    self.df[col] = self.df[col].astype(int)
            except Exception:
                continue
        return self

    def remove_special_characters(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].apply(lambda x: re.sub(r'[^\w\s]', '', str(x)))
        return self

    def rename_columns(self):
        self.df.columns = [col.strip().lower().replace(" ", "_") for col in self.df.columns]
        return self

    def ensure_unique_column_names(self):
        seen = {}
        new_columns = []
        for col in self.df.columns:
            if col not in seen:
                seen[col] = 1
                new_columns.append(col)
            else:
                seen[col] += 1
                new_columns.append(f"{col}_{seen[col]}")
        self.df.columns = new_columns
        return self


    def validate_numeric_ranges(self, min_value=0, max_value=1000000):
        for col in self.df.select_dtypes(include=np.number):
            self.df[col] = np.where(self.df[col] < min_value, min_value, self.df[col])
            self.df[col] = np.where(self.df[col] > max_value, max_value, self.df[col])
        return self

    def drop_empty_columns(self):
        self.df = self.df.dropna(axis=1, how='all')
        return self

    def standardize_categorical_values(self):
        mapping = {'m': 'Male', 'f': 'Female', 'male': 'Male', 'female': 'Female',
                   'yes': 'Yes', 'no': 'No', 'y': 'Yes', 'n': 'No'}
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].replace(mapping)
        return self

    def fix_encoding_errors(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('utf-8', 'ignore').decode('utf-8'))
        return self

    def remove_invisible_characters(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].apply(lambda x: re.sub(r'[\r\n\t\f\v]', ' ', str(x)).strip())
        return self

    def remove_units_and_symbols(self):
        for col in self.df.select_dtypes(include='object'):
            self.df[col] = self.df[col].apply(lambda x: re.sub(r'[₹$,%,]', '', str(x)))
        return self

    def run_full_cleaning(self):
        return (self.remove_duplicates()
                    .trim_spaces()
                    .fix_capitalization()
                    .unify_date_formats()
                    .handle_missing_values()
                    .correct_data_types()
                    .remove_special_characters()
                    .rename_columns()
                    .ensure_unique_column_names()
                    .validate_numeric_ranges()
                    .drop_empty_columns()
                    .standardize_categorical_values()
                    .fix_encoding_errors()
                    .remove_invisible_characters()
                    .remove_units_and_symbols()
                    .df
                )

def clean_dataset(input_data: Union[str, pd.DataFrame, 'io.BytesIO']) -> pd.DataFrame:
    """
    Helper function. Accepts a file path, file-like object, or DataFrame and returns a cleaned DataFrame.
    """
    if isinstance(input_data, pd.DataFrame):
        df = input_data.copy()
    else:
        # attempt to read as CSV
        try:
            df = pd.read_csv(input_data)
        except Exception as e:
            # try Excel
            try:
                df = pd.read_excel(input_data)
            except Exception:
                raise ValueError(f"Unable to read input data: {e}")

    cleaner = DataCleaner(df)
    cleaned = cleaner.run_full_cleaning()
    return cleaned