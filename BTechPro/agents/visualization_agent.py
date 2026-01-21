import plotly.express as px

class VisualizationAgent:
    """
    Generates 5 basic charts from the cleaned dataset
    for HTML dashboard visualization.
    """

    def __init__(self, df):
        self.df = df

    def generate_charts(self):
        charts = {}

        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        cat_cols = self.df.select_dtypes(include=["object"]).columns

        # 1️⃣ Histogram (Numeric Distribution)
        if len(numeric_cols) > 0:
            fig1 = px.histogram(self.df, x=numeric_cols[0],
                                title=f"Distribution of {numeric_cols[0]}")
            charts["histogram"] = fig1.to_json()

        # 2️⃣ Box Plot (Outliers)
        if len(numeric_cols) > 0:
            fig2 = px.box(self.df, y=numeric_cols[0],
                          title=f"Box Plot of {numeric_cols[0]}")
            charts["boxplot"] = fig2.to_json()

        # 3️⃣ Bar Chart (Categorical Count)
        if len(cat_cols) > 0:
            counts = self.df[cat_cols[0]].value_counts().reset_index()
            fig3 = px.bar(counts, x="index", y=cat_cols[0],
                          title=f"Category Count of {cat_cols[0]}")
            charts["barchart"] = fig3.to_json()

        # 4️⃣ Line Chart (Trend)
        if len(numeric_cols) > 1:
            fig4 = px.line(self.df, y=numeric_cols[1],
                           title=f"Trend of {numeric_cols[1]}")
            charts["linechart"] = fig4.to_json()

        # 5️⃣ Scatter Plot (Relationship)
        if len(numeric_cols) > 1:
            fig5 = px.scatter(self.df, x=numeric_cols[0], y=numeric_cols[1],
                              title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            charts["scatter"] = fig5.to_json()

        return charts