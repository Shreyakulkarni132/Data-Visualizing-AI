# chart_generator.py
import matplotlib
matplotlib.use("Agg")

import os
import uuid
import time
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader


# -------------------------------------------------
# App Config
# -------------------------------------------------
app = FastAPI(title="Advanced Chart Generator + Dashboard PDF Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = "static"
CLEANED_DATASET_DIR = "cleaned_datasets"
PDF_DIR = "pdfs"

IMAGE_TTL_SECONDS = 300  # 5 minutes

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CLEANED_DATASET_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def clear_static_folder():
    """Delete all images so new dashboard never mixes with old ones."""
    for file in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(path) and file.endswith(".png"):
            os.remove(path)
            print(f"🧹 Removed old image: {file}")


def cleanup_old_images():
    """Safety cleanup: remove images older than TTL."""
    now = time.time()
    for file in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(path) and file.endswith(".png"):
            if now - os.path.getmtime(path) > IMAGE_TTL_SECONDS:
                os.remove(path)
                print(f"🗑️ TTL Deleted: {file}")


def get_latest_csv():
    files = [
        os.path.join(CLEANED_DATASET_DIR, f)
        for f in os.listdir(CLEANED_DATASET_DIR)
        if f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No CSV file found in cleaned_datasets folder.")
    return max(files, key=os.path.getmtime)


def get_all_current_images():
    images = [
        os.path.join(STATIC_DIR, f)
        for f in os.listdir(STATIC_DIR)
        if f.endswith(".png")
    ]
    return sorted(images, key=os.path.getmtime)


# -------------------------------------------------
# PDF Generation
# -------------------------------------------------
def generate_pdf_from_images(images):
    pdf_name = f"dashboard_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_width, page_height = A4

    cols = 2
    rows = 3
    margin_x = 30
    margin_y = 40
    spacing_x = 20
    spacing_y = 20

    usable_width = page_width - 2 * margin_x
    usable_height = page_height - 2 * margin_y

    img_width = (usable_width - (cols - 1) * spacing_x) / cols
    img_height = (usable_height - (rows - 1) * spacing_y) / rows

    x_positions = [margin_x + i * (img_width + spacing_x) for i in range(cols)]
    y_positions = [
        page_height - margin_y - img_height - i * (img_height + spacing_y)
        for i in range(rows)
    ]

    img_index = 0

    for img_path in images:
        col = img_index % cols
        row = (img_index // cols) % rows

        x = x_positions[col]
        y = y_positions[row]

        c.drawImage(
            ImageReader(img_path),
            x, y,
            width=img_width,
            height=img_height,
            preserveAspectRatio=True,
            mask="auto"
        )

        img_index += 1

        if img_index % (cols * rows) == 0:
            c.showPage()

    c.save()
    return pdf_path


# -------------------------------------------------
# Plotting Functions
# -------------------------------------------------

def plot_bar(df, cols, title):
    x = cols[0]
    y = cols[1] if len(cols) > 1 else None

    if y:
        data = df.groupby(x)[y].mean()
        data.plot(kind="bar")
    else:
        df[x].value_counts().plot(kind="bar")

    plt.title(title)


def plot_line(df, cols, title):
    x = cols[0]
    y = cols[1] if len(cols) > 1 else cols[0]
    df.groupby(x)[y].mean().plot(kind="line")
    plt.title(title)


def plot_stacked_bar(df, cols, title):
    group_col = cols[0]
    stack_col = cols[1]

    data = pd.crosstab(df[group_col], df[stack_col])
    data.plot(kind="bar", stacked=True)
    plt.title(title)


def plot_heatmap(df, cols, title):
    x = cols[0]
    y = cols[1]

    pivot = pd.crosstab(df[x], df[y])
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues")
    plt.title(title)


def plot_area(df, cols, title):
    group1 = cols[0]
    group2 = cols[1]
    value = cols[2]

    data = df.groupby([group1, group2])[value].mean().unstack()
    data.plot(kind="area", stacked=True)
    plt.title(title)


def plot_boxplot(df, cols, title):
    x = cols[0]
    y = cols[1]
    df.boxplot(column=y, by=x)
    plt.title(title)
    plt.suptitle("")


def plot_funnel(df, cols, title):
    stage = cols[0]
    cancel = cols[1]

    data = df.groupby(stage)[cancel].mean().sort_values(ascending=False)
    data.plot(kind="barh")
    plt.title(title)


def plot_radar(df, cols, title):
    cat = cols[0]
    metrics = cols[1:]

    grouped = df.groupby(cat)[metrics].mean()
    labels = grouped.index
    values = grouped.values

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    for i, row in enumerate(values):
        ax.plot(angles, row, label=labels[i])
        ax.fill(angles, row, alpha=0.1)

    ax.set_thetagrids(angles * 180 / np.pi, metrics)
    plt.title(title)
    ax.legend(loc="upper right")

def plot_scatter(df, cols, title):
    x = cols[0]
    y = cols[1]

    plt.scatter(df[x], df[y], alpha=0.6)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)



def plot_pie(df, cols, title):
    col = cols[0]
    df[col].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title(title)


# -------------------------------------------------
# Chart Dispatcher
# -------------------------------------------------
def plot_chart(df, kpi, filepath):
    chart = kpi.suggested_chart.lower()
    cols = ast.literal_eval(kpi.columns) if isinstance(kpi.columns, str) else kpi.columns

    plt.figure(figsize=(7, 5))

    if "stacked bar" in chart:
        plot_stacked_bar(df, cols, kpi.name)

    elif "bar" in chart:
        plot_bar(df, cols, kpi.name)

    elif "line" in chart:
        plot_line(df, cols, kpi.name)

    elif "area" in chart:
        plot_area(df, cols, kpi.name)

    elif "scatter" in chart:
        plot_scatter(df, cols, kpi.name)

    elif "heatmap" in chart:
        plot_heatmap(df, cols, kpi.name)

    elif "box" in chart:
        plot_boxplot(df, cols, kpi.name)

    elif "radar" in chart:
        plot_radar(df, cols, kpi.name)

    elif "funnel" in chart:
        plot_funnel(df, cols, kpi.name)

    elif "pie" in chart:
        plot_pie(df, cols, kpi.name)

    else:
        raise ValueError(f"Unsupported chart type: {chart}")

    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()


# -------------------------------------------------
# Request Models
# -------------------------------------------------
class KPI(BaseModel):
    name: str
    columns: str
    suggested_chart: str


class KPIRequest(BaseModel):
    kpis: List[KPI]


# -------------------------------------------------
# API Endpoints
# -------------------------------------------------
@app.post("/generate-charts")
def generate_charts(req: KPIRequest):
    clear_static_folder()
    cleanup_old_images()

    df = pd.read_csv(get_latest_csv())
    image_urls = []

    for kpi in req.kpis:
        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(STATIC_DIR, filename)

        try:
            plot_chart(df, kpi, filepath)
            image_urls.append(f"http://127.0.0.1:8003/static/{filename}")
            print(f"✅ Generated chart: {kpi.name}")
        except Exception as e:
            print(f"❌ Error generating {kpi.name}: {e}")

    return {"images": image_urls}


@app.get("/download-dashboard")
def download_dashboard():
    images = get_all_current_images()
    if not images:
        return {"error": "No charts available to generate PDF"}

    pdf_path = generate_pdf_from_images(images)

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename="Dashboard_Report.pdf"
    )


@app.get("/")
def root():
    return {"status": "Advanced KPI Chart Generator + Dashboard PDF Service Running"}


