# tools.py - Fixed for proper CrewAI agent execution
import pandas as pd
import json
from typing import Optional
import data_store 
from langchain.tools import Tool


# Helper function to safely retrieve the DataFrame
def get_df_from_store(key: str) -> Optional[pd.DataFrame]:
    """
    Retrieves the DataFrame if the key matches and data is present.
    
    Args:
        key: The key to access the data store
        
    Returns:
        DataFrame if found, None otherwise
    """
    if key == data_store.IN_MEMORY_KEY and data_store.CLEANED_DF_STORAGE is not None:
        return data_store.CLEANED_DF_STORAGE
    return None


# --- Tool Function 1: For KPI Identification Agent (Summary) ---
def load_data_summary(input_text: str = "") -> str:
    """
    Loads a summary of the in-memory DataFrame.
    The agent can call this tool with any input - it always uses 'in_memory_data' key.
    
    Args:
        input_text: Any text input (ignored, uses fixed key internally)
        
    Returns:
        String containing dataset summary or error message
    """
    # Always use the fixed key regardless of input
    key = "in_memory_data"
    
    df = get_df_from_store(key)
    if df is None:
        return (
            "ERROR: Cleaned data not found in memory store. "
            "The dataset must be loaded before analysis can begin. "
            "Please ensure the data is properly stored with key 'in_memory_data'."
        )
    
    try:
        # Get column information
        column_info = {}
        for col in df.columns:
            dtype = df[col].dtype.name
            null_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            column_info[col] = {
                "dtype": dtype,
                "null_count": int(null_count),
                "unique_values": int(unique_count)
            }
        
        # Get sample data - handle missing tabulate library
        try:
            sample_data_md = df.head(5).to_markdown(index=False)
        except ImportError:
            # If tabulate is not installed, use simple string representation
            sample_data_md = df.head(5).to_string(index=False)
        
        # Create comprehensive summary
        summary = (
            f"=== DATASET SUMMARY ===\n\n"
            f"Total Rows: {len(df)}\n"
            f"Total Columns: {len(df.columns)}\n\n"
            f"=== COLUMN DETAILS ===\n"
            f"{json.dumps(column_info, indent=2)}\n\n"
            f"=== SAMPLE DATA (First 5 Rows) ===\n"
            f"{sample_data_md}\n\n"
            f"You can now identify KPIs based on this data structure and content."
        )
        return summary
        
    except Exception as e:
        return f"ERROR: Failed to summarize data - {str(e)}"


# --- Tool Function 2: For Data Visualization Agent (Full Data) ---
def load_full_dataset(input_text: str = "") -> str:
    """
    Loads the entire cleaned dataset from memory.
    The agent can call this tool with any input - it always uses 'in_memory_data' key.
    
    Args:
        input_text: Any text input (ignored, uses fixed key internally)
        
    Returns:
        JSON string of the full dataset or error message
    """
    # Always use the fixed key regardless of input
    key = "in_memory_data"
    
    df = get_df_from_store(key)
    if df is None:
        return (
            "ERROR: Cleaned data not found in memory store. "
            "The dataset must be loaded before visualization can begin. "
            "Please ensure the data is properly stored with key 'in_memory_data'."
        )
    
    try:
        # Convert to JSON with proper handling of datetime objects
        full_data = df.to_dict(orient="records")
        
        # Create a response with metadata
        response = {
            "status": "success",
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "data": full_data
        }
        
        return json.dumps(response, default=str, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to load dataset: {str(e)}"
        })


# Create LangChain Tool instances with simplified names and clear descriptions
DataSummaryTool = Tool(
    name="get_data_summary",
    func=load_data_summary,
    description=(
        "Get a comprehensive summary of the dataset including: "
        "row count, column count, data types, null values, unique values, and sample rows. "
        "Use this tool first to understand the dataset structure before identifying KPIs. "
        "No input required - just call the tool."
    )
)

FullDatasetTool = Tool(
    name="get_full_dataset",
    func=load_full_dataset,
    description=(
        "Get the complete dataset as JSON for detailed analysis and visualization planning. "
        "Returns all rows and columns with metadata. "
        "Use this after identifying KPIs to access actual data for calculations. "
        "No input required - just call the tool."
    )
)


# Export
__all__ = ['DataSummaryTool', 'FullDatasetTool']