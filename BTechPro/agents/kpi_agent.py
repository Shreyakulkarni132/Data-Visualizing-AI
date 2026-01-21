import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import faiss

class KPIIdentificationAgent:
    """
    Identifies key performance indicators (KPIs) using
    feature importance (ML-based) and RAG (FAISS-based retrieval).
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.kpis = {}
        self.kpi_docs = [
            ("Sales", "Total revenue generated from transactions."),
            ("Profit", "Net gain after subtracting costs from sales."),
            ("Quantity", "Number of units sold."),
            ("Discount", "Average discount offered per transaction."),
            ("Category", "Grouping of products for better segmentation."),
            ("Customer Segment", "Type of customers buying the product.")
        ]
        self.index, self.doc_map = self._build_vector_index()

    def _build_vector_index(self):
        """Builds a simple FAISS vector index for KPI retrieval."""
        embeddings = []
        doc_map = {}
        for i, (term, definition) in enumerate(self.kpi_docs):
            vec = np.random.rand(128).astype("float32")  # Placeholder embeddings
            embeddings.append(vec)
            doc_map[i] = (term, definition)

        index = faiss.IndexFlatL2(128)
        index.add(np.array(embeddings))
        return index, doc_map

    def identify_with_ml(self, target="sales"):
        """Uses RandomForest to rank features based on importance for predicting target."""
        target = target.lower()
        if target not in self.df.columns:
            print(f" Target column '{target}' not found. Skipping ML KPI detection.")
            return self.kpis

        features = self.df.drop(columns=[target])
        y = self.df[target]

        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(features, y)

        importance = model.feature_importances_
        feature_importance = pd.Series(importance, index=features.columns).sort_values(ascending=False)
        self.kpis["Top Features"] = feature_importance.head(5).to_dict()
        return self.kpis

    def rag_retrieve(self, query="profit"):
        """Retrieves KPI definition from a small vector knowledge base."""
        qvec = np.random.rand(128).astype("float32")
        D, I = self.index.search(np.array([qvec]), 1)
        term, definition = self.doc_map[I[0][0]]
        return {term: definition}

    def run(self, target="sales"):
        """Main function to identify KPIs using ML + RAG hybrid approach."""
        # Run ML-based feature importance
        if target.lower() in self.df.columns:
            self.identify_with_ml(target)

        # Run RAG-based retrieval
        rag_info = self.rag_retrieve()

        # Merge both results
        self.kpis.update(rag_info)
        print(" KPI identification complete.")
        return self.kpis
