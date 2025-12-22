import json
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# For torchvision EfficientNet-B3, the default weights expect 300x300 + ImageNet norm.
IMG_SIZE = 300

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

def load_label_map(path="idx_to_style.json"):
    with open(path, "r") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}

def load_model(weights_path: str, num_classes: int):
    # Create torchvision EfficientNet-B3
    model = efficientnet_b3(weights=None)

    # Replace classifier to match your 27 classes
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)

    state = torch.load(weights_path, map_location=DEVICE)

    # Common cases:
    # 1) state is a raw state_dict
    # 2) state is a dict like {"model": state_dict} or {"state_dict": state_dict}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # If trained with DataParallel, keys may be prefixed with "module."
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model

@torch.inference_mode()
def predict_pil(model, img: Image.Image, top_k: int, idx_to_style: dict[int, str]):
    if img.mode != "RGB":
        img = img.convert("RGB")

    x = preprocess(img).unsqueeze(0).to(DEVICE)
    logits = model(x)
    probs = F.softmax(logits, dim=1)[0]

    top_k = min(top_k, probs.shape[0])
    vals, idxs = torch.topk(probs, top_k)

    results = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        results.append({"index": int(i), "style": idx_to_style[int(i)], "prob": float(v)})

    return {"predicted": results[0], "top_k": results}
