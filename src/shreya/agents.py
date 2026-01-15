import asyncio
import nest_asyncio
from crewai import Agent
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from tools import DataSummaryTool, FullDatasetTool
from langchain.chat_models import ChatOpenAI
import os


# Fix for "no current event loop" error
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

nest_asyncio.apply()

load_dotenv()


llm = ChatOpenAI(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    openai_api_key=os.getenv("GROQ_API_KEY"),
    openai_api_base="https://api.groq.com/openai/v1"
)



# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.5-flash",
#     verbose=True,
#     temperature=0.3,  # Slightly increased for more creative KPI identification
#     google_api_key=os.getenv("GEMINI_API_KEY")
# )

kpi_identification_agent = Agent(
    role="Expert KPI Analyst",
    goal=(
        "Analyze the dataset using the get_data_summary tool and identify 5-8 key performance indicators (KPIs) "
        "that would be most valuable for business insights and visualization."
    ),
    verbose=True,
    memory=True,
    backstory=(
        "You are a senior data analyst with 10+ years of experience in identifying business KPIs. "
        "Your expertise lies in understanding data structures and determining which metrics matter most. "
        "\n\nYour workflow:\n"
        "1. FIRST: Use the 'get_data_summary' tool to understand the dataset\n"
        "2. ANALYZE: Review column types, null values, and sample data\n"
        "3. IDENTIFY: Determine 5-8 meaningful KPIs based on the data\n"
        "4. DOCUMENT: For each KPI, specify:\n"
        "   - KPI Name\n"
        "   - Column(s) used\n"
        "   - Data type\n"
        "   - Business reasoning\n"
        "   - Suggested visualization type\n"
        "\nIMPORTANT: You MUST use the get_data_summary tool as your first action."
    ),
    tools=[DataSummaryTool],
    llm=llm,
    allow_delegation=True
)

data_visualization_agent = Agent(
    role="To design a complete dashboard step by step using given KPIs and their columns and assiging best suited graphs and charts to each KPI.",
    goal="To create a finished dashboard design by defining which graphs and charts best represent the identified KPIs from the cleaned dataset.",
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert data visualization agent who specializes in designing dashboards using identified KPIs. "
        "Your job is to create a finished dashboard design by defining which graphs and charts best represent the identified KPIs from the cleaned dataset. "
        "The dashboard should be user-friendly, visually appealing, and effectively communicate the insights derived from the data. "
        "You will communicate with kpi_identification_agent to get the identified KPIs and their columns along with reasoning for the same. "
    ),
    tools=[FullDatasetTool],
    llm=llm,
    concurrent=True,
    allow_delegation=True)

'''
llm = LLM(
    model="gemini/gemini-1.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.1,
    base_url="https://generativelanguage.googleapis.com/v1/"  # Force v1 API
)


data_cleaning_agent = Agent(
    role="Data cleaning expert who specializes in handling and formating missing values, outliers, and inconsistent data formats.",
    goal="Clean the provided dataset to ensure it is ready for analysis and visualization. ",
    verbose=True,
    memory=True,
    backstory=(
        "You are an expert data cleaning agent who specializes in handling and formatting missing values, outliers, and inconsistent data formats. "
        "Your job is to clean the provided dataset to ensure it is ready for analysis and visualization. "
        "You will communicate with other agents if you or other agents need any additional information about the dataset or specific cleaning requirements."
    ),
    llm=llm,
    concurrent=True,
    allow_delegation=True,
    tools= [data_cleaning_tool]
    )
'''