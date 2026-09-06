import struct
import json
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

class IBotLoader:
    def __init__(self, ibot_file_path):
        self.file_path = ibot_file_path
        self.metadata = None
        self.class_names = []
        self.flat_weights = []
        
        self.unpack_binary_file()

    def unpack_binary_file(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Missing proprietary model file: {self.file_path}")
            
        print(f"[*] Opening file stream: {self.file_path}")
        with open(self.file_path, "rb") as f:
            magic = f.read(4)
            if magic != b"IBOT":
                raise ValueError("Security Rejection: Invalid file system signature.")
                
            _ = struct.unpack(">I", f.read(4)) 
            meta_len = struct.unpack(">I", f.read(4))[0]
            
            meta_json = f.read(meta_len).decode('utf-8')
            self.metadata = json.loads(meta_json)
            self.class_names = self.metadata["classes"]
            
            print(f"[✔] Authentication Verified.")
            print(f"    Architecture : {self.metadata['architecture']}")
            print(f"    Total Classes: {self.metadata['num_classes']}")
            
            raw_weight_bytes = f.read()
            num_floats = len(raw_weight_bytes) // 4
            self.flat_weights = struct.unpack(f"{num_floats}f", raw_weight_bytes)
            print(f"    Loaded {num_floats} parameters into memory block array.")

    def build_executable_model(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[*] Reconstructing network architecture graph on: {device}...")
        
        model = models.efficientnet_b0(weights=None)
        
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(in_features, len(self.class_names))
        )
        
        state_dict = model.state_dict()
        pointer = 0
        
        for key in sorted(state_dict.keys()):
            num_elements = state_dict[key].numel()
            layer_shape = state_dict[key].shape
            
            chunk = self.flat_weights[pointer : pointer + num_elements]
            state_dict[key] = torch.tensor(chunk).reshape(layer_shape)
            pointer += num_elements
            
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print("[+] Model loaded successfully and ready for GPU inference passes!\n")
        return model, self.class_names, device

def run_inference(model, class_names, device, image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert("RGB")
    img_t = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, class_idx = torch.max(probs, 1)
        
    return class_names[class_idx.item()], confidence.item()

def get_local_ibot_files():
    """Scans current execution path directory for all .ibot system targets"""
    return sorted([f for f in os.listdir(".") if f.endswith(".ibot")])


# ==================== INTERACTIVE SELECTION LOOP ====================
if __name__ == "__main__":
    try:
        print("═"*60)
        print("         PROPRIETARY .IBOT MODEL SELECTION CENTER")
        print("═"*60)
        
        # 1. Scan and detect all compiled models
        ibot_files = get_local_ibot_files()
        
        if not ibot_files:
            print("[-] Error: No compiled .ibot files found in this directory folder.")
            print("    Please run 'PthToIBot.py' first to compile a model file.")
            exit()
            
        print("Available Proprietary Models:")
        for idx, filename in enumerate(ibot_files, 1):
            print(f"  {idx}. {filename}")
            
        # 2. Model selector interface logic
        while True:
            choice = input("\nSelect a model number to load: ").strip()
            if choice.isdigit() and 1 <= int(choice) <= len(ibot_files):
                target_file = ibot_files[int(choice) - 1]
                break
            print("[-] Invalid input choice. Please enter a number from the menu list.")

        print(f"\n[*] Deploying core for asset file: '{target_file}'")
        
        # 3. Load and assemble the chosen engine file
        loader = IBotLoader(target_file)
        model, classes, compute_device = loader.build_executable_model()
        
        print("═"*60)
        print("         PROPRIETARY .IBOT INTERACTIVE RUNTIME CORE")
        print("═"*60)
        print("Instructions: Paste your image path directory layout below.")
        print("              Type 'exit' or 'quit' to terminate the app.")
        print("═"*60)

        # 4. Continuous prediction interface prompt loop
        while True:
            image_to_test = input("\nEnter the full path directory to your picture: ").strip()
            
            if image_to_test.lower() in ["exit", "quit"]:
                print("Shutting down proprietary runtime. Goodbye!")
                break
                
            if not image_to_test:
                continue
                
            image_to_test = image_to_test.replace("'", "").replace('"', '')
            
            if os.path.exists(image_to_test):
                print(f"[*] Processing visual matrices on GPU...")
                try:
                    species, confidence = run_inference(model, classes, compute_device, image_to_test)
                    
                    print("\n" + "  " + "★"*20 + " MATCH RESULTS " + "★"*20)
                    print(f"    PROPRIETARY IDENTIFICATION: {species.upper().replace('_', ' ')}")
                    print(f"    CLASSIFIER CONFIDENCE CORE: {confidence:.2%}")
                    print("  " + "★"*55)
                except Exception as eval_err:
                    print(f"[-] Critical failure during computation pass: {eval_err}")
            else:
                print(f"[-] Directory Invalid: Cannot locate image asset file at '{image_to_test}'")

    except Exception as e:
        print(f"\n[Loader Error Status]: {e}")
