# data_store.py
import pandas as pd
from typing import Optional

# Global variable to hold the DataFrame in memory
CLEANED_DF_STORAGE: Optional[pd.DataFrame] = None

# A function to ensure the store is cleared after use (good practice)
def clear_store():
    global CLEANED_DF_STORAGE
    CLEANED_DF_STORAGE = None

# A key that the agents will use in their tools (it can be anything, but we'll use a fixed key)
IN_MEMORY_KEY = "in_memory_data"