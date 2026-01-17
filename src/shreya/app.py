# app.py
import streamlit as st
import pandas as pd
import os
from data_cleaning import clean_dataset
from crew import run_kpi_from_df

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
            cleaned_df = clean_dataset(df)

        st.subheader("Cleaned Data Preview (Top 20 Rows)")
        st.dataframe(cleaned_df.head(20))

        with st.spinner("Running KPI Agent..."):
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

# Optional: Run Streamlit on a custom port (e.g., 8502)
if __name__ == "__main__":
    os.system("streamlit run app.py --server.port 8502")
