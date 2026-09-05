import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model_files():
    """Find all .pth model files in the current folder"""
    return sorted([f for f in os.listdir() if f.endswith(".pth")])

def load_model(model_path):
    checkpoint = torch.load(model_path, map_location=device)
    
    # Support both old and new saving styles
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        class_names = checkpoint["classes"]
        state_dict = checkpoint["model_state_dict"]
    else:
        # Old style (just the weights)
        raise Exception("This model was saved in an old format and is missing class names.")
    
    num_classes = len(class_names)
    
    # Check the file name to build the correct model architecture
    if "efficient" in model_path.lower() or "b0" in model_path.lower():
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    else:
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model, class_names

def predict(model, class_names, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)  # Fixed dimension error here
    
    return class_names[pred.item()], conf.item()

# ================= MAIN LOOP =================
while True:
    print("\n" + "="*50)
    print("Available Models:")
    print("="*50)
    
    models_list = get_model_files()
    
    if not models_list:
        print("No .pth model files found in this folder.")
        break
    
    for i, name in enumerate(models_list, 1):
        print(f"{i}. {name}")
    
    choice = input("\nType the number of the model you want to use: ").strip()
    
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(models_list):
        print("Invalid number. Please try again.")
        continue
    
    model_path = models_list[int(choice) - 1]
    print(f"\nLoading model: {model_path}")
    
    model, class_names = load_model(model_path)
    print("Model loaded successfully!")
    print("Classes:", class_names)
    
    image_path = input("\nEnter the path to the picture: ").strip()
    
    species, confidence = predict(model, class_names, image_path)
    
    print("\n" + "-"*40)
    print(f"Prediction : {species}")
    print(f"Confidence : {confidence:.1%}")
    print("-"*40)
    
    again = input("\nDo you want to test another image? (y/n): ").strip().lower()
    if again != "y":
        print("Goodbye!")
        break
