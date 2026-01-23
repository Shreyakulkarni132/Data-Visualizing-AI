
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

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# -------------------------------------------------
# Router
# -------------------------------------------------
router = APIRouter(
    prefix="/chart",
    tags=["Chart Generator"]
)

# -------------------------------------------------
# Paths
# -------------------------------------------------
STATIC_DIR = "static"
CLEANED_DATASET_DIR = "cleaned_datasets"
PDF_DIR = "pdfs"

IMAGE_TTL_SECONDS = 300  # 5 minutes

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(CLEANED_DATASET_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

# -------------------------------------------------
# Utility Functions
# -------------------------------------------------
def clear_static_folder():
    for file in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(path) and file.endswith(".png"):
            os.remove(path)

def cleanup_old_images():
    now = time.time()
    for file in os.listdir(STATIC_DIR):
        path = os.path.join(STATIC_DIR, file)
        if os.path.isfile(path) and file.endswith(".png"):
            if now - os.path.getmtime(path) > IMAGE_TTL_SECONDS:
                os.remove(path)

def get_latest_csv():
    files = [
        os.path.join(CLEANED_DATASET_DIR, f)
        for f in os.listdir(CLEANED_DATASET_DIR)
        if f.endswith(".csv")
    ]
    if not files:
        raise FileNotFoundError("No cleaned CSV found")
    return max(files, key=os.path.getmtime)

def get_all_current_images():
    return sorted(
        [os.path.join(STATIC_DIR, f) for f in os.listdir(STATIC_DIR) if f.endswith(".png")],
        key=os.path.getmtime
    )

# -------------------------------------------------
# PDF Generation
# -------------------------------------------------
def generate_pdf_from_images(images):
    pdf_name = f"dashboard_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_width, page_height = A4

    cols, rows = 2, 3
    margin_x, margin_y = 30, 40
    spacing_x, spacing_y = 20, 20

    usable_width = page_width - 2 * margin_x
    usable_height = page_height - 2 * margin_y

    img_width = (usable_width - (cols - 1) * spacing_x) / cols
    img_height = (usable_height - (rows - 1) * spacing_y) / rows

    x_positions = [margin_x + i * (img_width + spacing_x) for i in range(cols)]
    y_positions = [
        page_height - margin_y - img_height - i * (img_height + spacing_y)
        for i in range(rows)
    ]

    for idx, img_path in enumerate(images):
        col = idx % cols
        row = (idx // cols) % rows
        c.drawImage(
            ImageReader(img_path),
            x_positions[col],
            y_positions[row],
            img_width,
            img_height,
            preserveAspectRatio=True
        )
        if (idx + 1) % (cols * rows) == 0:
            c.showPage()

    c.save()
    return pdf_path

# -------------------------------------------------
# Chart Dispatcher
# -------------------------------------------------
def normalize_columns(col_str):
    """
    Safely convert KPI columns to a Python list.
    Handles:
      - Simple string: "Age" -> ["Age"]
      - List string: '["Age", "Income"]' -> ["Age", "Income"]
      - Already a list -> return as is
    """
    if isinstance(col_str, list):
        return col_str
    if isinstance(col_str, str):
        col_str = col_str.strip()
        if col_str.startswith("[") and col_str.endswith("]"):
            try:
                return ast.literal_eval(col_str)
            except:
                pass
        # fallback: single column string
        return [col_str]
    return []

def plot_bar(df, cols, title):
    x = cols[0]
    y = cols[1] if len(cols) > 1 else None
    if y:
        df.groupby(x)[y].mean().plot(kind="bar")
    else:
        df[x].value_counts().plot(kind="bar")
    plt.title(title)

def plot_line(df, cols, title):
    x = cols[0]
    y = cols[1] if len(cols) > 1 else cols[0]
    df.groupby(x)[y].mean().plot(kind="line")
    plt.title(title)

def plot_stacked_bar(df, cols, title):
    data = pd.crosstab(df[cols[0]], df[cols[1]])
    data.plot(kind="bar", stacked=True)
    plt.title(title)

def plot_heatmap(df, cols, title):
    pivot = pd.crosstab(df[cols[0]], df[cols[1]])
    sns.heatmap(pivot, annot=True, fmt="d", cmap="Blues")
    plt.title(title)

def plot_area(df, cols, title):
    data = df.groupby([cols[0], cols[1]])[cols[2]].mean().unstack()
    data.plot(kind="area", stacked=True)
    plt.title(title)

def plot_boxplot(df, cols, title):
    df.boxplot(column=cols[1], by=cols[0])
    plt.title(title)
    plt.suptitle("")

def plot_funnel(df, cols, title):
    data = df.groupby(cols[0])[cols[1]].mean().sort_values(ascending=False)
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
    plt.scatter(df[cols[0]], df[cols[1]], alpha=0.6)
    plt.xlabel(cols[0])
    plt.ylabel(cols[1])
    plt.title(title)

def plot_pie(df, cols, title):
    df[cols[0]].value_counts().plot.pie(autopct="%1.1f%%")
    plt.title(title)

def plot_chart(df, kpi, filepath):
    chart = kpi.suggested_chart.lower()
    cols = normalize_columns(kpi.columns)
    plt.figure(figsize=(7, 5))

    try:
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
    except Exception as e:
        plt.close()
        raise RuntimeError(f"Chart failed for KPI '{kpi.name}': {str(e)}")
    plt.close()

# -------------------------------------------------
# Schemas
# -------------------------------------------------
class KPI(BaseModel):
    name: str
    columns: str
    suggested_chart: str

class KPIRequest(BaseModel):
    kpis: List[KPI]

# -------------------------------------------------
# API: Generate Charts
# -------------------------------------------------
@router.post("/generate")
def generate_charts(req: KPIRequest):
    clear_static_folder()
    cleanup_old_images()

    df = pd.read_csv(get_latest_csv())
    images = []

    for kpi in req.kpis:
        filename = f"{uuid.uuid4().hex}.png"
        path = os.path.join(STATIC_DIR, filename)
        try:
            plot_chart(df, kpi, path)
            images.append(f"/static/{filename}")
        except Exception as e:
            print(f"❌ Error generating {kpi.name}: {e}")

    return {"images": images}

# -------------------------------------------------
# API: Download Dashboard PDF
# -------------------------------------------------
@router.get("/download-dashboard")
def download_dashboard():
    images = get_all_current_images()
    if not images:
        return {"error": "No charts available"}

    pdf_path = generate_pdf_from_images(images)
    return FileResponse(pdf_path, media_type="application/pdf", filename="Dashboard_Report.pdf")
