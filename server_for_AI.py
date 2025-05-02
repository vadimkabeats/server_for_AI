from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI()
@app.get("/")
async def read_root():
    return {"message": "Server is running"}
# Загружаем модель один раз
model = YOLO(r"C:\Users\vadke\PycharmProjects\Welding-Defects-Detection-master\welding_v2_aug\standart\weights\best.pt")

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "results/test_predictions"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Сохраняем входной файл
    file_id = str(uuid.uuid4())
    input_path = f"{UPLOAD_FOLDER}/{file_id}.jpg"
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Прогоняем через модель с визуализацией
    model.predict(source=input_path, save=True, project="results", name="test_predictions", exist_ok=True)

    # Путь к сохраненному изображению с разметкой
    output_path = f"{OUTPUT_FOLDER}/{os.path.basename(input_path)}"

    # Возвращаем изображение клиенту
    return FileResponse(output_path, media_type="image/jpeg")
