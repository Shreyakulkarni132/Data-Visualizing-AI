import os
import uuid
import time
import pandas as pd
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# --------------------------------
# App Config
# --------------------------------
app = FastAPI(title="Chart Generator + Dashboard PDF Service")

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

# --------------------------------
# Utility Functions
# --------------------------------
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

# --------------------------------
# PDF Generation (Single Page Grid)
# --------------------------------
def generate_pdf_from_images(images):
    pdf_name = f"dashboard_{uuid.uuid4().hex}.pdf"
    pdf_path = os.path.join(PDF_DIR, pdf_name)

    c = canvas.Canvas(pdf_path, pagesize=A4)
    page_width, page_height = A4

    # 2 columns × 3 rows grid
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

        # Move to new page if grid is full
        if img_index % (cols * rows) == 0:
            c.showPage()

    c.save()
    return pdf_path

# --------------------------------
# Request Models
# --------------------------------
class KPI(BaseModel):
    name: str
    columns: str
    suggested_chart: str

class KPIRequest(BaseModel):
    kpis: List[KPI]

# --------------------------------
# API Endpoints
# --------------------------------
@app.post("/generate-charts")
def generate_charts(req: KPIRequest):
    # 🔥 Ensure no repetition
    clear_static_folder()

    # 🧹 Safety TTL cleanup
    cleanup_old_images()

    df = pd.read_csv(get_latest_csv())
    image_urls = []

    for kpi in req.kpis:
        col = kpi.columns.split(",")[0].strip()
        chart_type = kpi.suggested_chart.lower()

        filename = f"{uuid.uuid4().hex}.png"
        filepath = os.path.join(STATIC_DIR, filename)

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
            print(f"⚠ Unsupported chart type: {chart_type}")
            plt.close()
            continue

        plt.title(kpi.name)
        plt.tight_layout()
        plt.savefig(filepath)
        plt.close()

        image_urls.append(f"http://127.0.0.1:8003/static/{filename}")

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
    return {"status": "Chart Generator + Single-Page Dashboard PDF Service Running"}
