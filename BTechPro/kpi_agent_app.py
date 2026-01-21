# # import asyncio
# # import nest_asyncio
# # import os
# # import json
# # from dotenv import load_dotenv
# # from typing import Optional

# # import pandas as pd
# # from fastapi import FastAPI
# # from fastapi.middleware.cors import CORSMiddleware
# # from pydantic import BaseModel

# # from crewai import Agent, Task, Crew
# # from langchain.chat_models import ChatOpenAI
# # from langchain.tools import Tool
# # from langchain.embeddings import OpenAIEmbeddings
# # from langchain.vectorstores import FAISS
# # from langchain.schema import Document


# # # --------------------------------------------------
# # # Fix event loop issues
# # # --------------------------------------------------
# # try:
# #     asyncio.get_running_loop()
# # except RuntimeError:
# #     asyncio.set_event_loop(asyncio.new_event_loop())
# # nest_asyncio.apply()

# # # --------------------------------------------------
# # # Load environment variables
# # # --------------------------------------------------
# # load_dotenv()

# # # --------------------------------------------------
# # # Paths
# # # --------------------------------------------------
# # CLEANED_DATASET_DIR = "cleaned_datasets"
# # KPI_RAG_CSV = r"C:\allenvs\Agents\Data-Visualizing-AI\src\Rag_data\BE project KPI data - Sheet1.csv"
# # KPI_VECTOR_DIR = "kpi_rag_index"


# # # --------------------------------------------------
# # # Helper: Get latest CSV
# # # --------------------------------------------------
# # def get_latest_cleaned_csv() -> Optional[str]:
# #     try:
# #         files = [
# #             os.path.join(CLEANED_DATASET_DIR, f)
# #             for f in os.listdir(CLEANED_DATASET_DIR)
# #             if f.endswith(".csv")
# #         ]
# #         if not files:
# #             return None
# #         return max(files, key=os.path.getmtime)
# #     except Exception:
# #         return None


# # def load_latest_dataframe() -> Optional[pd.DataFrame]:
# #     latest_file = get_latest_cleaned_csv()
# #     if latest_file is None:
# #         return None
# #     try:
# #         return pd.read_csv(latest_file)
# #     except Exception:
# #         return None


# # # --------------------------------------------------
# # # Column Semantic Detection
# # # --------------------------------------------------
# # def detect_column_type(series: pd.Series) -> str:
# #     if pd.api.types.is_numeric_dtype(series):
# #         return "numerical"
# #     elif pd.api.types.is_datetime64_any_dtype(series):
# #         return "datetime"
# #     elif series.nunique() < 20:
# #         return "categorical"
# #     elif pd.api.types.is_bool_dtype(series):
# #         return "boolean"
# #     else:
# #         return "text"


# # # --------------------------------------------------
# # # TOOL 1: Dataset Summary Tool
# # # --------------------------------------------------
# # def load_data_summary(input_text: str = "") -> str:
# #     df = load_latest_dataframe()
# #     if df is None:
# #         return (
# #             "ERROR: No cleaned dataset found. "
# #             "Please ensure at least one CSV file exists in the 'cleaned_datasets' directory."
# #         )

# #     try:
# #         column_info = {}
# #         for col in df.columns:
# #             column_info[col] = {
# #                 "dtype": df[col].dtype.name,
# #                 "semantic_type": detect_column_type(df[col]),
# #                 "null_count": int(df[col].isnull().sum()),
# #                 "unique_values": int(df[col].nunique())
# #             }

# #         try:
# #             sample_data_md = df.head(5).to_markdown(index=False)
# #         except:
# #             sample_data_md = df.head(5).to_string(index=False)

# #         summary = (
# #             f"=== DATASET SUMMARY (Latest Cleaned File) ===\n\n"
# #             f"File Used: {get_latest_cleaned_csv()}\n"
# #             f"Total Rows: {len(df)}\n"
# #             f"Total Columns: {len(df.columns)}\n\n"
# #             f"=== COLUMN DETAILS (With Semantic Types) ===\n"
# #             f"{json.dumps(column_info, indent=2)}\n\n"
# #             f"=== SAMPLE DATA (First 5 Rows) ===\n"
# #             f"{sample_data_md}\n\n"
# #             f"Use this dataset to generate KPIs using single, dual and multi-column logic."
# #         )
# #         return summary

# #     except Exception as e:
# #         return f"ERROR: Failed to summarize dataset - {str(e)}"


# # DataSummaryTool = Tool(
# #     name="get_data_summary",
# #     func=load_data_summary,
# #     description="Gets a detailed summary of the latest cleaned dataset. Use this FIRST."
# # )


# # # --------------------------------------------------
# # # RAG: Persistent KPI Knowledge Base
# # # --------------------------------------------------
# # def build_or_load_kpi_vectorstore():
# #     embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# #     if os.path.exists(KPI_VECTOR_DIR):
# #         print("🔄 Loading existing KPI embeddings from disk...")
# #         return FAISS.load_local(
# #             KPI_VECTOR_DIR,
# #             embeddings,
# #             allow_dangerous_deserialization=True
# #         )

# #     print("🧠 Building KPI embeddings for the first time...")
# #     df = pd.read_csv(KPI_RAG_CSV)

# #     documents = []
# #     for _, row in df.iterrows():
# #         text = (
# #             f"Dataset: {row['dataset_name']}\n"
# #             f"KPI Name: {row['kpi']}\n"
# #             f"Is KPI: {row['is_kpi']}\n"
# #             f"Data Type: {row['dtype']}\n"
# #             f"Domain: {row['domain']}\n"
# #             f"Description: {row['description']}"
# #         )
# #         documents.append(Document(page_content=text))

# #     vectorstore = FAISS.from_documents(documents, embeddings)
# #     vectorstore.save_local(KPI_VECTOR_DIR)
# #     print(f"✅ KPI embeddings saved to directory: {KPI_VECTOR_DIR}")

# #     return vectorstore


# # kpi_vectorstore = build_or_load_kpi_vectorstore()


# # # --------------------------------------------------
# # # TOOL 2: KPI RAG Retrieval
# # # --------------------------------------------------
# # def retrieve_kpi_knowledge(domain=None, columns=None, query=None):
# #     if domain or columns:
# #         domain = domain or ""
# #         if isinstance(columns, list):
# #             columns_text = ", ".join(columns)
# #         else:
# #             columns_text = str(columns)

# #         search_query = (
# #             f"Domain: {domain}. "
# #             f"Columns: {columns_text}. "
# #             f"Suggest KPI patterns using single, dual, and multi-column strategies."
# #         )
# #     elif query:
# #         search_query = query
# #     else:
# #         search_query = "General KPI patterns"

# #     results = kpi_vectorstore.similarity_search(search_query, k=5)
# #     context = "\n\n".join([doc.page_content for doc in results])

# #     return (
# #         "=== KPI KNOWLEDGE BASE CONTEXT (RAG) ===\n"
# #         "Use these examples as inspiration:\n\n"
# #         f"{context}"
# #     )


# # KPIKnowledgeTool = Tool(
# #     name="retrieve_kpi_knowledge",
# #     func=retrieve_kpi_knowledge,
# #     description="Retrieves relevant KPI patterns from the KPI knowledge base."
# # )


# # # --------------------------------------------------
# # # OpenAI LLM
# # # --------------------------------------------------
# # llm = ChatOpenAI(
# #     model="gpt-4o-mini",
# #     temperature=0.3,
# #     openai_api_key=os.getenv("OPENAI_API_KEY")
# # )


# # # --------------------------------------------------
# # # KPI IDENTIFICATION AGENT
# # # --------------------------------------------------
# # kpi_identification_agent = Agent(
# #     role="Strategic KPI Designer",
# #     goal=(
# #         "Design analytically strong, multi-dimensional KPIs that combine multiple "
# #         "columns and produce executive-grade dashboard insights. "
# #         "Avoid shallow single-column KPIs unless strategically justified."
# #     ),
# #     verbose=True,
# #     memory=True,
# #     backstory=(
# #         "You are not a metric reporter. You are a KPI architect.\n"
# #         "You design KPIs that show relationships, efficiency, behavior patterns, and business performance.\n"
# #         "Your KPIs must feel like insights, not statistics."
# #     ),
# #     tools=[DataSummaryTool, KPIKnowledgeTool],
# #     llm=llm,
# #     allow_delegation=True
# # )




# # # --------------------------------------------------
# # # KPI TASK
# # # --------------------------------------------------
# # kpi_identification_task = Task(
# #     description=(
# #         "Optional user context (weak hint):\n"
# #         "- Dataset Description: {user_dataset_description}\n"
# #         "- Dataset Domain: {user_dataset_domain}\n\n"

# #         "You MUST generate analytically rich KPIs.\n\n"

# #         "Mandatory KPI composition:\n"
# #         "• At least 2 KPIs using a SINGLE column\n"
# #         "• At least 3 KPIs using DUAL columns (ratios, comparisons, dependencies)\n"
# #         "• At least 3 KPIs using MULTI columns (3 or more columns, composite insights)\n\n"

# #         "Avoid trivial KPIs like simple counts or averages unless they are part of a deeper business story.\n\n"

# #         "Each KPI must:\n"
# #         "• Combine multiple signals where possible\n"
# #         "• Reflect operational or business decision value\n"
# #         "• Be something a dashboard user would act on\n\n"

# #         "Charts:\n"
# #         "• Mix of advanced + basic types:\n"
# #         "  - Line, Bar, Heatmap, Funnel, Stacked Bar, Radar, Area, Boxplot\n"
# #         "• Titles must be:\n"
# #         "  - Professional\n"
# #         "  - 4–7 words\n"
# #         "  - Clearly describe the business meaning\n"
# #         "Examples:\n"
# #         "  - 'Booking Cancellation Behavior Over Time'\n"
# #         "  - 'Revenue Performance by Market Segment'\n"
# #         "  - 'Guest Stay Pattern Analysis'\n"
# #         "  - 'Booking Lead Time Distribution'\n\n"

# #         "Process:\n"
# #         "STEP 1: Call get_data_summary\n"
# #         "STEP 2: Infer dataset domain\n"
# #         "STEP 3: Call retrieve_kpi_knowledge\n"
# #         "STEP 4: Generate 8–12 KPIs meeting the composition rules\n"
# #         "STEP 5: Output ONLY in the required format\n"
# #     ),

# #     expected_output=(
# #         "KPI <n>:\n"
# #         "  Name: <Descriptive Professional KPI Name>\n"
# #         "  Columns: <Column names used>\n"
# #         "  Reasoning: <Explain business or operational value clearly>\n"
# #         "  Suggested Chart:\n"
# #         "    - Chart Type: <Advanced or basic chart>\n"
# #         "    - Chart Title: <Professional descriptive title (4–7 words)>\n\n"
# #     ),

# #     agent=kpi_identification_agent
# # )



# # # --------------------------------------------------
# # # CREW
# # # --------------------------------------------------
# # crew = Crew(
# #     agents=[kpi_identification_agent],
# #     tasks=[kpi_identification_task],
# #     verbose=True
# # )


# # # --------------------------------------------------
# # # FASTAPI
# # # --------------------------------------------------
# # app = FastAPI(title="Advanced RAG Powered KPI Identification Agent")

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=["*"],
# #     allow_methods=["*"],
# #     allow_headers=["*"],
# # )


# # class KPIRequest(BaseModel):
# #     dataset_description: Optional[str] = None
# #     dataset_domain: Optional[str] = None


# # @app.post("/identify-kpis")
# # def identify_kpis(request: KPIRequest):
# #     try:
# #         inputs = {
# #             "user_dataset_description": request.dataset_description or "",
# #             "user_dataset_domain": request.dataset_domain or ""
# #         }

# #         result = crew.kickoff(inputs=inputs)
# #         result_str = str(result)

# #         return {
# #             "status": "success",
# #             "kpi_output": result_str
# #         }

# #     except Exception as e:
# #         return {
# #             "status": "error",
# #             "message": str(e)
# #         }


# import asyncio
# import nest_asyncio
# import os
# import json
# from dotenv import load_dotenv
# from typing import Optional

# import pandas as pd
# from fastapi import APIRouter
# from pydantic import BaseModel

# from crewai import Agent, Task, Crew
# from langchain.chat_models import ChatOpenAI
# from langchain.tools import Tool
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.schema import Document


# # --------------------------------------------------
# # Router
# # --------------------------------------------------
# router = APIRouter(
#     prefix="/kpi",
#     tags=["KPI Agent"]
# )

# # --------------------------------------------------
# # Fix event loop issues
# # --------------------------------------------------
# try:
#     asyncio.get_running_loop()
# except RuntimeError:
#     asyncio.set_event_loop(asyncio.new_event_loop())
# nest_asyncio.apply()

# # --------------------------------------------------
# # Load environment variables
# # --------------------------------------------------
# load_dotenv()

# # --------------------------------------------------
# # Paths
# # --------------------------------------------------
# CLEANED_DATASET_DIR = "cleaned_datasets"
# KPI_RAG_CSV = r"C:\allenvs\Agents\Data-Visualizing-AI\src\Rag_data\BE project KPI data - Sheet1.csv"
# KPI_VECTOR_DIR = "kpi_rag_index"


# # --------------------------------------------------
# # Helper: Get latest CSV
# # --------------------------------------------------
# def get_latest_cleaned_csv() -> Optional[str]:
#     try:
#         files = [
#             os.path.join(CLEANED_DATASET_DIR, f)
#             for f in os.listdir(CLEANED_DATASET_DIR)
#             if f.endswith(".csv")
#         ]
#         if not files:
#             return None
#         return max(files, key=os.path.getmtime)
#     except Exception:
#         return None


# def load_latest_dataframe() -> Optional[pd.DataFrame]:
#     latest_file = get_latest_cleaned_csv()
#     if latest_file is None:
#         return None
#     try:
#         return pd.read_csv(latest_file)
#     except Exception:
#         return None


# # --------------------------------------------------
# # Column Semantic Detection
# # --------------------------------------------------
# def detect_column_type(series: pd.Series) -> str:
#     if pd.api.types.is_numeric_dtype(series):
#         return "numerical"
#     elif pd.api.types.is_datetime64_any_dtype(series):
#         return "datetime"
#     elif series.nunique() < 20:
#         return "categorical"
#     elif pd.api.types.is_bool_dtype(series):
#         return "boolean"
#     else:
#         return "text"


# # --------------------------------------------------
# # TOOL 1: Dataset Summary Tool
# # --------------------------------------------------
# def load_data_summary(input_text: str = "") -> str:
#     df = load_latest_dataframe()
#     if df is None:
#         return "ERROR: No cleaned dataset found."

#     column_info = {}
#     for col in df.columns:
#         column_info[col] = {
#             "dtype": df[col].dtype.name,
#             "semantic_type": detect_column_type(df[col]),
#             "null_count": int(df[col].isnull().sum()),
#             "unique_values": int(df[col].nunique())
#         }

#     try:
#         sample_data_md = df.head(5).to_markdown(index=False)
#     except:
#         sample_data_md = df.head(5).to_string(index=False)

#     return (
#         f"Rows: {len(df)}\n"
#         f"Columns: {len(df.columns)}\n\n"
#         f"{json.dumps(column_info, indent=2)}\n\n"
#         f"{sample_data_md}"
#     )


# DataSummaryTool = Tool(
#     name="get_data_summary",
#     func=load_data_summary,
#     description="Gets dataset summary"
# )


# # --------------------------------------------------
# # RAG Vector Store
# # --------------------------------------------------
# def build_or_load_kpi_vectorstore():
#     embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

#     if os.path.exists(KPI_VECTOR_DIR):
#         return FAISS.load_local(
#             KPI_VECTOR_DIR,
#             embeddings,
#             allow_dangerous_deserialization=True
#         )

#     df = pd.read_csv(KPI_RAG_CSV)
#     documents = []

#     for _, row in df.iterrows():
#         text = (
#             f"KPI: {row['kpi']}\n"
#             f"Domain: {row['domain']}\n"
#             f"Description: {row['description']}"
#         )
#         documents.append(Document(page_content=text))

#     vectorstore = FAISS.from_documents(documents, embeddings)
#     vectorstore.save_local(KPI_VECTOR_DIR)
#     return vectorstore


# kpi_vectorstore = build_or_load_kpi_vectorstore()


# def retrieve_kpi_knowledge(domain=None, columns=None, query=None):
#     search_query = query or f"Domain: {domain}, Columns: {columns}"
#     results = kpi_vectorstore.similarity_search(search_query, k=5)
#     return "\n\n".join([doc.page_content for doc in results])


# KPIKnowledgeTool = Tool(
#     name="retrieve_kpi_knowledge",
#     func=retrieve_kpi_knowledge,
#     description="Retrieve KPI patterns"
# )


# # --------------------------------------------------
# # LLM
# # --------------------------------------------------
# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0.3,
#     openai_api_key=os.getenv("OPENAI_API_KEY")
# )


# # --------------------------------------------------
# # Agent + Task + Crew (UNCHANGED LOGIC)
# # --------------------------------------------------
# kpi_identification_agent = Agent(
#     role="Strategic KPI Designer",
#     goal="Design deep KPIs",
#     verbose=True,
#     memory=True,
#     tools=[DataSummaryTool, KPIKnowledgeTool],
#     llm=llm,
#     allow_delegation=True
# )

# kpi_identification_task = Task(
#     description="Generate professional KPIs",
#     expected_output="Structured KPI list",
#     agent=kpi_identification_agent
# )

# crew = Crew(
#     agents=[kpi_identification_agent],
#     tasks=[kpi_identification_task],
#     verbose=True
# )


# # --------------------------------------------------
# # API SCHEMA
# # --------------------------------------------------
# class KPIRequest(BaseModel):
#     dataset_description: Optional[str] = None
#     dataset_domain: Optional[str] = None


# # --------------------------------------------------
# # API ENDPOINT
# # --------------------------------------------------
# @router.post("/identify")
# def identify_kpis(request: KPIRequest):
#     inputs = {
#         "user_dataset_description": request.dataset_description or "",
#         "user_dataset_domain": request.dataset_domain or ""
#     }

#     result = crew.kickoff(inputs=inputs)

#     return {
#         "status": "success",
#         "kpi_output": str(result)
#     }

import asyncio
import nest_asyncio
import os
import json
from dotenv import load_dotenv
from typing import Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from crewai import Agent, Task, Crew
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document


# --------------------------------------------------
# Router
# --------------------------------------------------
router = APIRouter(
    prefix="/kpi",
    tags=["KPI Agent"]
)

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
KPI_RAG_CSV = KPI_RAG_CSV = r"C:\Users\Vaishnav T Kokate\OneDrive\Desktop\BTechPro\Rag_data\BE project KPI data - Sheet1.csv"
KPI_VECTOR_DIR = "kpi_rag_index"


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
# Column Semantic Detection
# --------------------------------------------------
def detect_column_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    elif series.nunique() < 20:
        return "categorical"
    elif pd.api.types.is_bool_dtype(series):
        return "boolean"
    else:
        return "text"


# --------------------------------------------------
# TOOL 1: Dataset Summary Tool
# --------------------------------------------------
def load_data_summary(input_text: str = "") -> str:
    df = load_latest_dataframe()
    if df is None:
        return "ERROR: No cleaned dataset found."

    column_info = {}
    for col in df.columns:
        column_info[col] = {
            "dtype": df[col].dtype.name,
            "semantic_type": detect_column_type(df[col]),
            "null_count": int(df[col].isnull().sum()),
            "unique_values": int(df[col].nunique())
        }

    try:
        sample_data_md = df.head(5).to_markdown(index=False)
    except:
        sample_data_md = df.head(5).to_string(index=False)

    return (
        f"Rows: {len(df)}\n"
        f"Columns: {len(df.columns)}\n\n"
        f"{json.dumps(column_info, indent=2)}\n\n"
        f"{sample_data_md}"
    )


DataSummaryTool = Tool(
    name="get_data_summary",
    func=load_data_summary,
    description="Gets dataset summary"
)


# --------------------------------------------------
# RAG Vector Store
# --------------------------------------------------
def build_or_load_kpi_vectorstore():
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    if os.path.exists(KPI_VECTOR_DIR):
        return FAISS.load_local(
            KPI_VECTOR_DIR,
            embeddings,
            allow_dangerous_deserialization=True
        )

    df = pd.read_csv(KPI_RAG_CSV)
    documents = []

    for _, row in df.iterrows():
        text = (
            f"KPI: {row['kpi']}\n"
            f"Domain: {row['domain']}\n"
            f"Description: {row['description']}"
        )
        documents.append(Document(page_content=text))

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(KPI_VECTOR_DIR)
    return vectorstore


kpi_vectorstore = build_or_load_kpi_vectorstore()


def retrieve_kpi_knowledge(domain=None, columns=None, query=None):
    search_query = query or f"Domain: {domain}, Columns: {columns}"
    results = kpi_vectorstore.similarity_search(search_query, k=5)
    return "\n\n".join([doc.page_content for doc in results])


KPIKnowledgeTool = Tool(
    name="retrieve_kpi_knowledge",
    func=retrieve_kpi_knowledge,
    description="Retrieve KPI patterns"
)


# --------------------------------------------------
# LLM
# --------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)


# --------------------------------------------------
# Agent + Task + Crew (UNCHANGED LOGIC)
# --------------------------------------------------
kpi_identification_agent = Agent(
    role="Strategic KPI Designer",
    backstory=(
        "You are an expert KPI identification agent. "
        "Your task is to generate professional, structured KPIs for a dataset. "
        "You will use dataset summaries and KPI knowledge patterns to suggest KPIs "
        "and recommend appropriate chart types."
    ),
    goal="Identify key KPIs for the given dataset; Suggest chart types for each KPI; Use dataset summary and domain knowledge",
    verbose=True,
    memory=True,
    tools=[DataSummaryTool, KPIKnowledgeTool],
    llm=llm,
    allow_delegation=True
)

kpi_identification_task = Task(
    description="Generate professional KPIs",
    expected_output="Structured KPI list",
    agent=kpi_identification_agent
)

crew = Crew(
    agents=[kpi_identification_agent],
    tasks=[kpi_identification_task],
    verbose=True
)


# --------------------------------------------------
# API SCHEMA
# --------------------------------------------------
class KPIRequest(BaseModel):
    dataset_description: Optional[str] = None
    dataset_domain: Optional[str] = None


# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
@router.post("/identify")
def identify_kpis(request: KPIRequest):
    inputs = {
        "user_dataset_description": request.dataset_description or "",
        "user_dataset_domain": request.dataset_domain or ""
    }

    result = crew.kickoff(inputs=inputs)

    return {
        "status": "success",
        "kpi_output": str(result)
    }
