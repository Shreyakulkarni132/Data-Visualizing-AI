import os
import uuid
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List

app = FastAPI(title="Chart Generator Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve images
app.mount("/static", StaticFiles(directory="static"), name="static")

CLEANED_DATASET_DIR = "cleaned_datasets"


def get_latest_csv():
    files = [
        os.path.join(CLEANED_DATASET_DIR, f)
        for f in os.listdir(CLEANED_DATASET_DIR)
        if f.endswith(".csv")
    ]
    return max(files, key=os.path.getmtime)


class KPI(BaseModel):
    name: str
    columns: str
    suggested_chart: str


class KPIRequest(BaseModel):
    kpis: List[KPI]


@app.post("/generate-charts")
def generate_charts(req: KPIRequest):
    df = pd.read_csv(get_latest_csv())
    image_urls = []

    for kpi in req.kpis:
        col = kpi.columns.split(",")[0].strip()
        chart_type = kpi.suggested_chart.lower()

        filename = f"static/{uuid.uuid4().hex}.png"

        plt.figure(figsize=(6, 4))

        if "pie" in chart_type:
            df[col].value_counts().plot.pie(autopct="%1.1f%%")
        elif "line" in chart_type:
            df[col].plot()
        elif "bar" in chart_type:
            df[col].value_counts().plot.bar()
        elif "histogram" in chart_type:
            df[col].plot.hist()
        else:
            continue

        plt.title(kpi.name)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        image_urls.append(f"http://127.0.0.1:8003/{filename}")

    return {"images": image_urls}
