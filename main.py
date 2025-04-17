### RUN THIS IN COMMAND PROMPT ###
'''
python main.py
uvicorn main:app --reload
'''


from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import shutil
import os
from model_utils import process_image

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "processed": False})

@app.post("/", response_class=HTMLResponse)
async def handle_upload(request: Request, file: UploadFile = File(...)):
    upload_path = f"static/uploads/{file.filename}"
    os.makedirs("static/uploads", exist_ok=True)
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    process_image(upload_path)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "processed": True,
        "original": "static/outputs/original.jpg",
        "heatmap": "static/outputs/heatmap.jpg",
        "overlay": "static/outputs/overlay.jpg"
    })
