import streamlit as st
import pandas as pd
import os
from agents.data_cleaning_agent import DataCleaningAgent
from agents.kpi_agent import KPIIdentificationAgent
from agents.visualization_agent import VisualizationAgent

# Streamlit App Setup
st.set_page_config(page_title="Automated Analytics System", layout="wide")
st.title("🤖 Automated Data Visualization & Analytics System")

uploaded_file = st.file_uploader("📂 Upload a dataset (CSV format)", type=["csv"])

if uploaded_file:
    # Try UTF-8 first; fall back to Windows-1252 automatically
    try:
        df = pd.read_csv(uploaded_file, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(uploaded_file, encoding="latin1")

    st.success("✅ Dataset uploaded successfully!")

    # Raw Data Display
    st.subheader("🧾 Raw Data Preview")
    st.dataframe(df.head())

    # Step 1: Data Cleaning
    st.header("🧹 Step 1: Data Cleaning Agent")
    cleaner = DataCleaningAgent(df)
    df_cleaned = cleaner.run()
    st.dataframe(df_cleaned.head())

    # Save Cleaned Data
    os.makedirs("data/processed", exist_ok=True)
    df_cleaned.to_csv("data/processed/cleaned_data.csv", index=False)
    st.success("Cleaned data saved in 'data/processed/cleaned_data.csv'")

    # Step 2: KPI Identification
    st.header("📈 Step 2: KPI Identification Agent (ML + RAG)")
    kpi_agent = KPIIdentificationAgent(df_cleaned)
    kpis = kpi_agent.run(target="sales" if "sales" in df_cleaned.columns else df_cleaned.columns[-1])
    st.json(kpis)

    # Step 3: Visualization
    st.header("📊 Step 3: Visualization Agent")
    viz = VisualizationAgent(df_cleaned)
    viz.auto_visualize()

else:
    st.info("👆 Upload a CSV file to start the automated analytics process.")
