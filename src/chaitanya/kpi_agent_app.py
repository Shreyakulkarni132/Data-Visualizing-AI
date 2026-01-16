import asyncio
import nest_asyncio
import os
import json
from dotenv import load_dotenv
from typing import Optional

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


# --------------------------------------------------
# Fix event loop issues
# --------------------------------------------------
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())
nest_asyncio.apply()

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# Paths
# --------------------------------------------------
CLEANED_DATASET_DIR = "cleaned_datasets"
KPI_RAG_CSV = "C:\\allenvs\\Agents\\Data-Visualizing-AI\\src\\Rag_data\\BE project KPI data - Sheet1.csv"
KPI_VECTOR_DIR = "kpi_rag_index"   # 📁 persistent embeddings directory


# --------------------------------------------------
# Helper: Get latest CSV
# --------------------------------------------------
def get_latest_cleaned_csv() -> Optional[str]:
    try:
        files = [
            os.path.join(CLEANED_DATASET_DIR, f)
            for f in os.listdir(CLEANED_DATASET_DIR)
            if f.endswith(".csv")
        ]
        if not files:
            return None
        return max(files, key=os.path.getmtime)
    except Exception:
        return None


def load_latest_dataframe() -> Optional[pd.DataFrame]:
    latest_file = get_latest_cleaned_csv()
    if latest_file is None:
        return None
    try:
        return pd.read_csv(latest_file)
    except Exception:
        return None


# --------------------------------------------------
# TOOL 1: Dataset Summary Tool
# --------------------------------------------------
def load_data_summary(input_text: str = "") -> str:
    df = load_latest_dataframe()
    if df is None:
        return (
            "ERROR: No cleaned dataset found. "
            "Please ensure at least one CSV file exists in the 'cleaned_datasets' directory."
        )

    try:
        column_info = {}
        for col in df.columns:
            column_info[col] = {
                "dtype": df[col].dtype.name,
                "null_count": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique())
            }

        try:
            sample_data_md = df.head(5).to_markdown(index=False)
        except:
            sample_data_md = df.head(5).to_string(index=False)

        summary = (
            f"=== DATASET SUMMARY (Latest Cleaned File) ===\n\n"
            f"File Used: {get_latest_cleaned_csv()}\n"
            f"Total Rows: {len(df)}\n"
            f"Total Columns: {len(df.columns)}\n\n"
            f"=== COLUMN DETAILS ===\n"
            f"{json.dumps(column_info, indent=2)}\n\n"
            f"=== SAMPLE DATA (First 5 Rows) ===\n"
            f"{sample_data_md}\n\n"
            f"You can now identify KPIs based on this dataset."
        )
        return summary

    except Exception as e:
        return f"ERROR: Failed to summarize dataset - {str(e)}"


DataSummaryTool = Tool(
    name="get_data_summary",
    func=load_data_summary,
    description=(
        "Gets a summary of the latest cleaned dataset CSV. "
        "Use this FIRST before identifying KPIs."
    )
)


# --------------------------------------------------
# RAG: Persistent KPI Knowledge Base
# --------------------------------------------------
def build_or_load_kpi_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    # Load if already exists
    if os.path.exists(KPI_VECTOR_DIR):
        print("🔄 Loading existing KPI embeddings from disk...")
        return FAISS.load_local(
            KPI_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    # Else create and save
    print("🧠 Building KPI embeddings for the first time...")
    df = pd.read_csv(KPI_RAG_CSV)

    documents = []
    for _, row in df.iterrows():
        text = (
            f"Dataset: {row['dataset_name']}\n"
            f"KPI Name: {row['kpi']}\n"
            f"Is KPI: {row['is_kpi']}\n"
            f"Data Type: {row['dtype']}\n"
            f"Domain: {row['domain']}\n"
            f"Description: {row['description']}"
        )
        documents.append(Document(page_content=text))

    vectorstore = FAISS.from_documents(documents, embeddings)

    # Persist to disk
    vectorstore.save_local(KPI_VECTOR_DIR)
    print(f"✅ KPI embeddings saved to directory: {KPI_VECTOR_DIR}")

    return vectorstore


kpi_vectorstore = build_or_load_kpi_vectorstore()


# --------------------------------------------------
# TOOL 2: RAG Retrieval Tool
# --------------------------------------------------
def retrieve_kpi_knowledge(domain=None, columns=None, query=None):
    """
    CrewAI sends arguments as keyword parameters.
    We support:
    - domain + columns
    - OR a raw query string
    """

    # Build search query
    if domain or columns:
        domain = domain or ""
        columns = columns or []
        if isinstance(columns, list):
            columns_text = ", ".join(columns)
        else:
            columns_text = str(columns)

        search_query = (
            f"Domain: {domain}. "
            f"Columns: {columns_text}. "
            f"Suggest similar KPI patterns."
        )

    elif query:
        search_query = query

    else:
        search_query = "General KPI patterns"

    # Run vector search
    results = kpi_vectorstore.similarity_search(search_query, k=5)

    context = "\n\n".join([doc.page_content for doc in results])

    return (
        "=== KPI KNOWLEDGE BASE CONTEXT (RAG) ===\n"
        "The following KPI examples from historical datasets can guide KPI selection:\n\n"
        f"{context}"
    )



KPIKnowledgeTool = Tool(
    name="retrieve_kpi_knowledge",
    func=retrieve_kpi_knowledge,
    description=(
        "Retrieves relevant KPI examples from the KPI knowledge base CSV using semantic search. "
        "Use this AFTER dataset summary."
    )
)


# --------------------------------------------------
# OpenAI LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# KPI IDENTIFICATION AGENT (with RAG)
# --------------------------------------------------
kpi_identification_agent = Agent(
    role="Expert KPI Analyst",
    goal=(
        "Analyze the dataset using get_data_summary and enrich KPI identification "
        "using retrieve_kpi_knowledge (RAG)."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a senior data analyst.\n\n"
        "Workflow:\n"
        "1. FIRST: Call get_data_summary\n"
        "2. SECOND: Call retrieve_kpi_knowledge using dataset domain + column names\n"
        "3. ANALYZE dataset + retrieved KPI patterns\n"
        "4. IDENTIFY 5–8 KPIs\n"
        "5. OUTPUT strictly in required format\n\n"
        "IMPORTANT: Always use BOTH tools."
    ),
    tools=[DataSummaryTool, KPIKnowledgeTool],
    llm=llm,
    allow_delegation=True
)


# --------------------------------------------------
# KPI TASK
# --------------------------------------------------
kpi_identification_task = Task(
    description=(
        "STEP 1: Use get_data_summary to retrieve dataset info.\n"
        "STEP 2: Use retrieve_kpi_knowledge with domain + column names.\n"
        "STEP 3: Combine dataset info and KPI examples.\n"
        "STEP 4: Identify 5–8 KPIs.\n"
        "STEP 5: Output strictly in required format."
    ),
    expected_output=(
        "KPI 1:\n"
        "  Name: <KPI Name>\n"
        "  Columns: <Column Names>\n"
        "  Data Type: <Data Type>\n"
        "  Reasoning: <Why this KPI matters>\n"
        "  Suggested Chart: <Chart Type>\n\n"
        "[Repeat for all KPIs]"
    ),
    agent=kpi_identification_agent
)


# --------------------------------------------------
# CREW
# --------------------------------------------------
crew = Crew(
    agents=[kpi_identification_agent],
    tasks=[kpi_identification_task],
    verbose=True
)


# --------------------------------------------------
# FASTAPI
# --------------------------------------------------
app = FastAPI(title="RAG Powered KPI Identification Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class KPIRequest(BaseModel):
    run: bool = True


@app.post("/identify-kpis")
def identify_kpis(request: KPIRequest):
    try:
        result = crew.kickoff()
        result_str = str(result)

        return {
            "status": "success",
            "kpi_output": result_str
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
