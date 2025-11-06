from crewai import Crew
from agents import kpi_identification_agent, data_visualization_agent
from tasks import kpi_identification_task, data_visualization_task
import pandas as pd
import data_store

def create_crew():
    """Create and return the Crew instance."""
    crew = Crew(
        agents=[kpi_identification_agent, data_visualization_agent],
        tasks=[kpi_identification_task, data_visualization_task],
        verbose=True
    )
    return crew

def run_kpi_from_df(cleaned_df: pd.DataFrame):
    """
    Stores the DataFrame in memory, runs the Crew, and clears the store.
    
    Args:
        cleaned_df: The cleaned pandas DataFrame to analyze
        
    Returns:
        The crew execution result or error dict
    """
    try:
        if not isinstance(cleaned_df, pd.DataFrame):
            raise ValueError("cleaned_df must be a pandas DataFrame")
        
        if cleaned_df.empty:
            raise ValueError("DataFrame is empty")

        # 1. Store the DataFrame globally
        data_store.CLEANED_DF_STORAGE = cleaned_df
        print(f"✓ Stored DataFrame with {len(cleaned_df)} rows and {len(cleaned_df.columns)} columns")
        
        # 2. Create and run the crew
        crew = create_crew()
        
        # 3. Execute the crew - NO inputs needed, tools access data directly
        print("Starting crew execution...")
        result = crew.kickoff()
        
        print("✓ Crew execution completed")
        return result

    except Exception as e:
        print(f"✗ Error in crew execution: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}
    
    finally:
        # 4. Always clear the store after the run
        data_store.clear_store()
        print("✓ Data store cleared")

if __name__ == "__main__":
    # Simple local test with sample data
    import pandas as pd
    
    # Create test data
    test_data = {
        'date': pd.date_range('2024-01-01', periods=50),
        'sales': [1000 + i * 10 for i in range(50)],
        'region': ['North', 'South', 'East', 'West'] * 12 + ['North', 'South'],
        'revenue': [5000 + i * 50 for i in range(50)]
    }
    test_df = pd.DataFrame(test_data)
    
    print("Running crew with test data...")
    result = run_kpi_from_df(test_df)
    
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(result)