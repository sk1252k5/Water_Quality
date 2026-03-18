import torch
import torch.nn as nn
from torchvision import models,transforms
from PIL import     Image
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the trained model

model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, 3)

# Get the directory where this script (model_recommendation.py) is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build the path to the model relative to this script
# We go up one level to the project root, then into the models folder
model_path = os.path.join(BASE_DIR, "..", "models", "water_cnn_model.pth")


model.load_state_dict(torch.load(model_path,map_location=device))
model.to(device)
model.eval()

#class labels
class_names = ['Bersih','Keruh','Kotor']

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    return class_names[predicted.item()], confidence.item()

def image_based_recommendation(predicted_class, confidence):

    # If model is not confident → treat as Keruh
    if confidence < 0.7:
        predicted_class = "Keruh"

    if predicted_class == "Bersih":
        return {
            "Water_Quality": "Clean",
            "Recommendation": "Safe to drink"
        }

    elif predicted_class == "Keruh":
        return {
            "Water_Quality": "Turbid",
            "Recommendation": "Sediment Filtration + Boiling Recommended"
        }

    elif predicted_class == "Kotor":
        return {
            "Water_Quality": "Highly Contaminated",
            "Recommendation": "RO + UV + Sediment Filter"
        }
    
def analyze_water_image(image_path):

    predicted_class, confidence = predict_image(image_path)

    recommendation = image_based_recommendation(predicted_class, confidence)

    return {
        "Predicted_Class": predicted_class,
        "Water_Quality": recommendation["Water_Quality"],
        "Recommendation": recommendation["Recommendation"]
    }