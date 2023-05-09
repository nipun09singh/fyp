import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from io import BytesIO
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic import BaseModel


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def home():
    return templates.TemplateResponse("index.html", {"request": None})


LABELS = ['None', 'Meningioma', 'Glioma', 'Pituitary']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_model = models.resnet50(pretrained=True)
for param in resnet_model.parameters():
    param.requires_grad = True

n_inputs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Linear(n_inputs, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 2048),
    nn.SELU(),
    nn.Dropout(p=0.4),
    nn.Linear(2048, 4),
    nn.LogSigmoid()
)

resnet_model.to(device)
resnet_model.load_state_dict(torch.load(
    './models/modelFinal.pt', map_location=device))
resnet_model.eval()


def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = Image.open(BytesIO(image_bytes))
    return transform(img).unsqueeze(0)


def predict_image(image_bytes):
    tensor = preprocess_image(image_bytes=image_bytes)
    y_hat = resnet_model(tensor.to(device))
    class_id = torch.argmax(y_hat.data, dim=1)
    return str(int(class_id)), LABELS[int(class_id)]


class PredictionResult(BaseModel):
    class_id: str
    class_name: str


@app.post('/predict', response_model=PredictionResult)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    class_id, class_name = predict_image(image_bytes)
    return {"class_id": class_id, "class_name": class_name}
