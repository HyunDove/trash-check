"""YOLO 게이트 — 가장 큰 물체 하나를 찾아 crop.

COCO 사전학습 YOLOv8n을 그대로 사용한다(파인튜닝 없음).
COCO 80개 클래스에는 캔·종이박스·일반쓰레기 등 대부분의 쓰레기 카테고리가
없어서, 실제로 물체가 하나 있어도 0개로 판정되는 경우가 흔하다. 또한
배경의 리모컨·컵 등 관련 없는 물체까지 함께 잡히는 경우도 많다. 앱은
"한 장에 하나만 촬영"을 UX로 전제하므로, 감지된 물체가 여러 개여도
경고하지 않고 바운딩 박스 면적이 가장 큰 물체(=사진에서 가장 비중이
큰 대상) 하나만 골라 crop한다. 0개 감지 시에는 전체 이미지를 그대로
crop으로 사용한다.
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
    status: str  # "ok"
    count: int
    crop: Image.Image | None = None


def detect_single_object(image: Image.Image) -> GateResult:
    """감지된 물체 중 바운딩 박스 면적이 가장 큰 것을 crop해 반환한다."""
    results = _get_model()(image, conf=CONF_THRESHOLD, verbose=False)
    boxes = results[0].boxes

    if len(boxes) == 0:
        return GateResult(status="ok", count=1, crop=image)  # COCO 미인식 물체 폴백

    crop = image
    try:
        xyxy = boxes.xyxy.tolist()
        areas = [(x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in xyxy]
        x1, y1, x2, y2 = xyxy[areas.index(max(areas))]
        candidate = image.crop((int(x1), int(y1), int(x2), int(y2)))
        if candidate.width > 0 and candidate.height > 0:
            crop = candidate  # 유효한 크롭일 때만 사용, 아니면 원본 이미지 폴백
    except Exception:
        pass  # 좌표 계산이 실패해도 원본 이미지로 진행 (crop이 None이 되는 상황 방지)

    return GateResult(status="ok", count=len(boxes), crop=crop)
