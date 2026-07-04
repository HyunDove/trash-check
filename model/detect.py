"""YOLO 게이트 — 물체 개수 판정 + 단일 객체 crop.

COCO 사전학습 YOLOv8n을 그대로 사용한다(파인튜닝 없음).
역할: 업로드 이미지에서 물체 개수를 세고, 정확히 1개일 때만 박스 영역을 잘라 반환.
"""
from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO

CONF_THRESHOLD = 0.35

_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # 최초 호출 시 자동 다운로드(~6MB)
    return _model


@dataclass
class GateResult:
    status: str  # "ok" | "none" | "multiple"
    count: int
    crop: Image.Image | None = None


def detect_single_object(image: Image.Image) -> GateResult:
    """물체 개수를 판정하고, 1개일 때 crop 이미지를 함께 반환한다."""
    results = _get_model()(image, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return GateResult(status="none", count=0)
    if len(boxes) > 1:
        return GateResult(status="multiple", count=len(boxes))

    x1, y1, x2, y2 = boxes.xyxy[0].tolist()
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    return GateResult(status="ok", count=1, crop=crop)
