import os
# Force hardware acceleration directly to your RTX 4070 Ti
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"

import json
import struct
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

def get_selected_model(model_choice, num_classes):
    """Initializes the requested skeleton architecture graph and remaps output dimensions"""
    if model_choice == "1":
        print("[*] Configuring EfficientNet-B0 backbone...")
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        arch_name = "EfficientNetB0_Custom"
        
    elif model_choice == "2":
        print("[*] Configuring EfficientNet-B1 backbone...")
        model = models.efficientnet_b1(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, num_classes)
        )
        arch_name = "EfficientNetB1_Custom"
        
    elif model_choice == "3":
        print("[*] Configuring ConvNeXt-Base backbone...")
        model = models.convnext_base(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)
        arch_name = "ConvNeXtBase_Custom"
    else:
        raise ValueError("Invalid architecture choice index.")
        
    return model, arch_name

def train_and_compile_pipeline():
    print("═"*60)
    print("         PROPRIETARY .IBOT MODEL BUILD ENGINE")
    print("═"*60)
    
    # 1. Interactive Setup Prompts
    print("Select Core Backbone Architecture:")
    print("  1. EfficientNet-B0 (Lightweight & Fast)")
    print("  2. EfficientNet-B1 (Balanced & Efficient)")
    print("  3. ConvNeXt-Base   (Heavy & High-Accuracy)")
    
    while True:
        model_choice = input("Enter choice (1-3): ").strip()
        if model_choice in ["1", "2", "3"]: break
        print("[-] Invalid selection. Enter 1, 2, or 3.")
        
    while True:
        epochs_input = input("\nEnter total training epochs loops (e.g., 30 or 120): ").strip()
        if epochs_input.isdigit() and int(epochs_input) > 0:
            epochs = int(epochs_input)
            break
        print("[-] Please enter a valid positive integer number.")

    # 2. Hardware Allocation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[+] Hardware Locked to: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 3. Path Verification
    train_dir = "data/train"
    val_dir = "data/val"
    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        print(f"[-] Error: Could not locate directory tracking folders at '{train_dir}' or '{val_dir}'")
        return

    # 4. Transformations and Data Binding
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    
    # num_workers=0 blocks multi-threaded process locks on Linux systems
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    print(f"[+] Loaded Dataset: {num_classes} Classes found.")
    print(f"    Training Images  : {len(train_dataset)}")
    print(f"    Validation Images: {len(val_dataset)}")

    # 5. Model Initialization
    model, architecture_tag = get_selected_model(model_choice, num_classes)
    model = model.to(device)

    # 6. Loss Optimization Infrastructure
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    # 7. Core Optimization Execution Loop
    print(f"\n[*] Starting training sequence on the GPU...")
    for epoch in range(1, epochs + 1):
        # --- TRAINING PHASING ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100

        # --- VALIDATION PHASING ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        print(f"    Epoch [{epoch:03d}/{epochs:03d}] -> "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

    # ==================== AUTO-COMPILING TO PROPRIETARY .IBOT FORMAT ====================
    output_filename = f"{architecture_tag.lower()}_model.ibot"
    print(f"\n[*] Exporting multi-dimensional tensors directly to proprietary layout file...")
    
    meta_dict = {
        "architecture": architecture_tag,
        "num_classes": num_classes,
        "classes": class_names,
        "engine_requirement": "Multi_Model_IBot_Runtime_v1"
    }
    meta_json = json.dumps(meta_dict).encode('utf-8')

    # Serialize matrices deterministically
    state_dict = model.state_dict()
    flattened_weights = []
    
    for key in sorted(state_dict.keys()):
        tensor_data = state_dict[key].detach().cpu().numpy().flatten().tolist()
        flattened_weights.extend(tensor_data)

    print(f"[+] Serialized {len(flattened_weights)} parameters directly out of hardware memory.")
    weight_bytes = struct.pack(f"{len(flattened_weights)}f", *flattened_weights)

    with open(output_filename, "wb") as f:
        f.write(b"IBOT")                           # Magic bytes header identifier
        f.write(struct.pack(">I", 1))              # File system layout version 
        f.write(struct.pack(">I", len(meta_json))) # Size tracking signature
        f.write(meta_json)                         # Decoded internal configuration settings string
        f.write(weight_bytes)                      # Flat binary weights data array

    print(f"[✔] Success! Asset compiled directly to your file system format: '{output_filename}'\n")

if __name__ == "__main__":
    train_and_compile_pipeline()
