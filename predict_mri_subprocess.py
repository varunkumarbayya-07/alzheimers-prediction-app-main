"""
predict_mri_subprocess.py - PyTorch MRI inference subprocess
Called: python predict_mri_subprocess.py <img_path> <model_path> <indices_path>
Output: JSON on stdout
"""
import os, sys, json
os.environ["PYTHONIOENCODING"] = "utf-8"

if len(sys.argv) < 4:
    print(json.dumps({"error": "Usage: predict_mri_subprocess.py <img> <model> <indices>"}))
    sys.exit(1)

img_path, model_path, indices_path = sys.argv[1], sys.argv[2], sys.argv[3]

try:
    import numpy as np
    from PIL import Image as PILImage
    import torch
    import torchvision.models as tv_models
    import torchvision.transforms as T
    import torch.nn as nn

    # Load class indices
    with open(indices_path) as f:
        class_to_idx = json.load(f)
    n_cls   = len(class_to_idx)
    inv_idx = {str(v): k for k, v in class_to_idx.items()}

    # Load checkpoint
    ckpt    = torch.load(model_path, map_location="cpu")
    img_sz  = ckpt.get("img_size", 128)

    # Rebuild model (EfficientNet-B0)
    try:
        model = tv_models.efficientnet_b0(weights=None)
        in_f  = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, n_cls),
        )
    except Exception:
        model = tv_models.mobilenet_v3_small(weights=None)
        in_f  = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_f, n_cls)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Preprocess image
    transform = T.Compose([
        T.Resize((img_sz, img_sz)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    with PILImage.open(img_path) as img:
        img.load()
        tensor = transform(img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        out   = model(tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy().tolist()

    class_idx  = int(np.argmax(probs))
    class_name = inv_idx[str(class_idx)]
    confidence = float(probs[class_idx]) * 100

    print(json.dumps({
        "class_name": class_name,
        "confidence": confidence,
        "probs":      probs,
        "inv_idx":    inv_idx,
    }))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
