# app.py
import streamlit as st
import pandas as pd
from data_cleaning import clean_dataset     #  Import your cleaning function
from crew import run_kpi_from_df             #  Import the crew runner for KPI agent

st.set_page_config(page_title="Data Cleaning & KPI Agent", layout="wide")
st.title("AI-Powered Data Cleaning & KPI Analysis")

st.write("Upload a CSV file, clean it, and automatically extract KPIs using AI.")

uploaded_file = st.file_uploader("Upload your dataset (CSV format only)", type=["csv"])

if uploaded_file is not None:
    st.success("File uploaded successfully")
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head(10))

    if st.button("Clean and Analyze Data"):
        with st.spinner("Cleaning dataset..."):
            cleaned_df = clean_dataset(df)  #  This gives the cleaned DataFrame

        st.subheader("Cleaned Data Preview (Top 20 Rows)")
        st.dataframe(cleaned_df.head(20))

        with st.spinner("Running KPI Agent..."):
            #  Send cleaned data directly to the crew runner
            kpi_result = run_kpi_from_df(cleaned_df) 

        st.subheader("KPI Agent Output")
        st.write(kpi_result)

        csv = cleaned_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Cleaned CSV",
            data=csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv",
        )
        st.success("Data cleaned and analyzed successfully.")
