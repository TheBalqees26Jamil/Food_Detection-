from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io

# ----- FastAPI setup -----
app = FastAPI(title="Food Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- إعداد الموديل -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "models/final_efficientnet_b0.pth"

num_classes = 10
classes = ['apple_pie', 'bibimbap', 'cannoli', 'chicken_curry', 'falafel',
           'french_toast', 'ice_cream', 'ramen', 'sushi', 'tiramisu']

calories_dict = {
    'apple_pie': 300,
    'bibimbap': 450,
    'cannoli': 350,
    'chicken_curry': 500,
    'falafel': 350,
    'french_toast': 400,
    'ice_cream': 200,
    'ramen': 550,
    'sushi': 300,
    'tiramisu': 450
}

# ----- Load the trained model -----
def load_model():
    model = models.efficientnet_b0(weights=None)  # لا نحمّل ImageNet weights
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

# ----- Image transforms -----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ----- API endpoint -----
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, dim=1)

    class_name = classes[pred_idx.item()]
    confidence = conf.item() * 100
    calories = calories_dict[class_name]

    return {
        "class_name": class_name,
        "confidence": round(confidence, 2),
        "calories": calories
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
