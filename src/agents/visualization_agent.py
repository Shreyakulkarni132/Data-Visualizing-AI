import streamlit as st
import plotly.express as px

class VisualizationAgent:
    """
    Automatically generates relevant visualizations based on dataset column types.
    """

    def __init__(self, df):
        self.df = df

    def auto_visualize(self):
        """Dynamically generates charts for numeric and categorical columns."""
        st.write("### 📊 Auto-Generated Visualizations")

        numeric_cols = [col for col in self.df.columns if self.df[col].dtype in ["int64", "float64"]]
        cat_cols = [col for col in self.df.columns if self.df[col].dtype == "object"]

        # Numeric distributions
        for col in numeric_cols[:5]:  # Limit to 5 for clarity
            fig = px.histogram(self.df, x=col, title=f"Distribution of {col}")
            st.plotly_chart(fig, use_container_width=True)

        # Categorical counts
        for col in cat_cols[:5]:
            fig = px.bar(self.df[col].value_counts().reset_index(),
                         x="index", y=col, title=f"Category Count: {col}")
            st.plotly_chart(fig, use_container_width=True)
