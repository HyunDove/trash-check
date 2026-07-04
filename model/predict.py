"""CNN 추론 — Colab에서 학습한 best_model.pt 로드 후 재질 분류 (CPU 추론).

입력은 YOLO 게이트가 잘라준 crop 이미지. 출력은 (영문 라벨, 확신도).
"""
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = Path(__file__).resolve().parent / "best_model.pt"

LABELS_KO = {"plastic": "플라스틱", "can": "캔", "glass": "유리", "paper": "종이", "trash": "일반쓰레기"}

_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None
_classes = None


def _get_model():
    global _model, _classes
    if _model is None:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        _classes = ckpt["classes"]
        m = models.efficientnet_b0()
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(_classes))
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        _model = m
    return _model, _classes


def classify(image: Image.Image) -> tuple[str, float]:
    """crop 이미지를 4클래스로 분류. (영문 라벨, 확신도 0~1) 반환."""
    model, classes = _get_model()
    x = _tf(image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx])
