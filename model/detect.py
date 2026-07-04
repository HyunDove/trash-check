"""YOLO 게이트 — 물체 개수 판정 + 단일 객체 crop.

COCO 사전학습 YOLOv8n을 그대로 사용한다(파인튜닝 없음).
COCO 80개 클래스에는 캔·종이박스·일반쓰레기 등 대부분의 쓰레기 카테고리가
없어서, 실제로 물체가 하나 있어도 0개로 판정되는 경우가 흔하다. 앱은
"한 장에 하나만 촬영"을 UX로 강제하므로, 0개 감지 시 실패 처리하지 않고
전체 이미지를 그대로 crop으로 사용한다.
"""
from dataclasses import dataclass

from PIL import Image
from ultralytics import YOLO

CONF_THRESHOLD = 0.25

_model = None


def _get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")  # 최초 호출 시 자동 다운로드(~6MB)
    return _model


@dataclass
class GateResult:
    status: str  # "ok" | "multiple"
    count: int
    crop: Image.Image | None = None


def detect_single_object(image: Image.Image) -> GateResult:
    """물체 개수를 판정하고, crop 이미지를 함께 반환한다."""
    results = _get_model()(image, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return GateResult(status="ok", count=1, crop=image)  # COCO 미인식 물체 폴백
    if len(boxes) > 1:
        return GateResult(status="multiple", count=len(boxes))

    x1, y1, x2, y2 = boxes.xyxy[0].tolist()
    crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
    return GateResult(status="ok", count=1, crop=crop)
