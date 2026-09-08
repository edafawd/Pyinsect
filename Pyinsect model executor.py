import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

def find_models():
    return sorted([f for f in os.listdir(".") if f.lower().endswith(".pth")])

def guess_arch(state_dict):
    keys = list(state_dict.keys())
    if any(k.startswith("features.") for k in keys) or any("classifier." in k for k in keys):
        return "efficientnet_b0"
    return "resnet18"

def build_model(arch, num_classes):
    if arch == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        class_names = ckpt["classes"]
        arch = ckpt.get("arch") or guess_arch(state_dict)
    else:
        raise ValueError("This .pth has no class names. Retrain and save with model_state_dict + classes.")

    model = build_model(arch, len(class_names))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, class_names, arch

def predict(model, class_names, device, image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        conf, idx = torch.max(probs, 0)
    return class_names[idx.item()], conf.item()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

while True:
    try:
        print("\n" + "=" * 50)
        print("Available Models:")
        print("=" * 50)

        models_list = find_models()
        if not models_list:
            print("No .pth files found in this folder.")
            break

        for i, name in enumerate(models_list, 1):
            print(f"{i}. {name}")

        choice = input("\nType the number of the model you want to use: ").strip()
        if not choice.isdigit() or not (1 <= int(choice) <= len(models_list)):
            print("Invalid number. Try again.")
            continue

        model_path = models_list[int(choice) - 1]
        print(f"Loading model: {model_path}")
        model, class_names, arch = load_model(model_path, device)
        print(f"Loaded {arch} with {len(class_names)} classes.")

        while True:
            image_path = input("\nEnter the path to the picture (or 'back' / 'exit'): ").strip().strip("'\"")
            if image_path.lower() in {"exit", "quit"}:
                print("Goodbye!")
                raise SystemExit
            if image_path.lower() == "back":
                break
            if not image_path:
                continue
            if not os.path.exists(image_path):
                print("File not found.")
                continue

            species, confidence = predict(model, class_names, device, image_path)
            print("-" * 40)
            print(f"Prediction : {species}")
            print(f"Confidence : {confidence:.1%}")
            print("-" * 40)

    except SystemExit:
        break
    except Exception:
        print("\nOops something went wrong we are sorry about that :)")
        print("(You can try again)\n")
