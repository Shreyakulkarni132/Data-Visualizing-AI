from agents import kpi_identification_agent, data_visualization_agent
from crewai import Task
import pandas as pd

'''
data_cleaning_task = Task(
    description=(
        "Clean the provided dataset to ensure it is ready for analysis and visualization. "
        "Perform necessary data cleaning steps such as handling missing values, outliers, inconsistent data formats, "
        "and any other issues that may affect the quality of the data."
        "Use tools given to you in the tools file to perform data cleaning."
    ),
    agent= data_cleaning_agent,
    expected_output="A refined cleaned dataset in CSV format along with a short summary of cleaning steps performed.",
)
'''

kpi_identification_task = Task(
    description=(
        "STEP 1: Use the 'get_data_summary' tool to retrieve dataset information.\n"
        "STEP 2: Analyze the dataset structure, columns, data types, and sample values.\n"
        "STEP 3: Identify 5-8 meaningful KPIs that would provide valuable business insights.\n"
        "STEP 4: For EACH KPI you identify, provide:\n"
        "   a) KPI Name (clear and descriptive)\n"
        "   b) Column(s) involved\n"
        "   c) Data type of the column(s)\n"
        "   d) Business reasoning (why this KPI matters)\n"
        "   e) Suggested visualization type (chart recommendation)\n"
        "\nFocus on KPIs that are:\n"
        "- Measurable from the available data\n"
        "- Relevant to business objectives\n"
        "- Suitable for visualization\n"
        "- Diverse (covering different aspects of the data)\n"
    ),
    expected_output=(
        "A structured list of 5-8 KPIs with complete details:\n"
        "KPI 1:\n"
        "  Name: [KPI Name]\n"
        "  Columns: [column names]\n"
        "  Data Type: [type]\n"
        "  Reasoning: [why this matters]\n"
        "  Suggested Chart: [chart type]\n"
        "\n[Repeat for each KPI]\n"
        "\nExample format:\n"
        "KPI 1:\n"
        "  Name: Total Sales Revenue\n"
        "  Columns: sales_amount\n"
        "  Data Type: numeric (float64)\n"
        "  Reasoning: Critical metric for tracking business performance and growth\n"
        "  Suggested Chart: Line chart (time series) or Bar chart (by category)\n"
    ),
    agent=kpi_identification_agent
)


data_visualization_task = Task(
    description=( 
        "**STEP 1: Load Full Dataset.** You must first use the **'In-Memory Full Dataset Loader'** tool "
        "with the input key `{'data_key'}`. This will provide the entire dataset as a JSON string, which is necessary "
        "for statistical summaries and calculation.\n\n"
        
        "**STEP 2: Analyze & Design.** Using the full dataset and the identified KPIs from the preceding task, "
        "create a finished dashboard design by defining which graphs and charts best represent the identified KPIs.\n"
        
        "**Key Requirements:**\n"
        "* **Statistical Summary:** Provide a summary you deduce from the statistics of the data and what the user should focus on further.\n"
        "* **Dashboard Layout:** The dashboard should be user-friendly and visually appealing. Define a layout that includes well-defined sections, **cards** for key metrics, and associated visualizations.\n"
        "* **Visualization Choices:** Select appropriate graphs and charts for each KPI (e.g., line charts for time series, bar/pie charts for categorical data, histograms for distributions), ensuring clarity and impact.\n"
        "* **Aesthetics:** Consider color schemes, chart types, and overall aesthetics to enhance user engagement and comprehension."
    ),
    agent=data_visualization_agent,
    expected_output=(
        "A comprehensive dashboard design description with well-defined and placed graphs, charts, and cards, along with a summary of key deductions for the user."
    ),
    context=[kpi_identification_task],
)