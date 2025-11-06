# test_crew.py - Test your CrewAI setup
import pandas as pd
import data_store
from tools import DataSummaryTool, FullDatasetTool

# Create sample data for testing
sample_data = {
    'date': pd.date_range('2024-01-01', periods=100),
    'sales': [1000 + i * 10 for i in range(100)],
    'region': ['North', 'South', 'East', 'West'] * 25,
    'product': ['A', 'B', 'C'] * 33 + ['A'],
    'customer_count': [50 + i for i in range(100)],
    'revenue': [5000 + i * 50 for i in range(100)]
}

df = pd.DataFrame(sample_data)

print("=" * 60)
print("TESTING TOOL SETUP")
print("=" * 60)

# Test 1: Store data
print("\n1. Storing sample data...")
data_store.CLEANED_DF_STORAGE = df
print(f"   ✓ Stored {len(df)} rows, {len(df.columns)} columns")

# Test 2: Test DataSummaryTool
print("\n2. Testing DataSummaryTool...")
try:
    summary = DataSummaryTool.func("")
    print("   ✓ DataSummaryTool works!")
    print("\n   Sample output:")
    print("   " + "\n   ".join(summary.split('\n')[:10]))
    print("   ...")
except Exception as e:
    print(f"   ✗ DataSummaryTool failed: {e}")

# Test 3: Test FullDatasetTool
print("\n3. Testing FullDatasetTool...")
try:
    full_data = FullDatasetTool.func("")
    print("   ✓ FullDatasetTool works!")
    print(f"   Data length: {len(full_data)} characters")
except Exception as e:
    print(f"   ✗ FullDatasetTool failed: {e}")

# Test 4: Test agent import
print("\n4. Testing agent imports...")
try:
    from agents import kpi_identification_agent, data_visualization_agent
    print("   ✓ Agents imported successfully!")
    print(f"   KPI Agent tools: {[t.name for t in kpi_identification_agent.tools]}")
    print(f"   Viz Agent tools: {[t.name for t in data_visualization_agent.tools]}")
except Exception as e:
    print(f"   ✗ Agent import failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Test crew runner
print("\n5. Testing crew runner...")
try:
    from crew import run_kpi_from_df
    print("   ✓ Crew runner imported successfully!")
    
    # Try running the crew
    print("\n   Running crew with sample data...")
    print("   This may take a minute...\n")
    result = run_kpi_from_df(df)
    
    print("\n" + "=" * 60)
    print("CREW EXECUTION RESULT")
    print("=" * 60)
    print(result)
    
except Exception as e:
    print(f"   ✗ Crew execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)