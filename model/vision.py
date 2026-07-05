"""VLM 최종 판정 — HF Inference API의 비전 모델로 재질+이물질을 함께 판단.

CNN(EfficientNet-B0)은 TrashNet(흰 배경 스튜디오 사진)으로 학습되어
실제 촬영 사진(다양한 배경·조명·오염)에서는 재질을 혼동하는 경우가 잦다
(예: 기름때 묻은 플라스틱 그릇을 캔으로 오분류). 범용 비전-언어 모델인
Qwen2.5-VL은 이런 실제 사진에 대한 일반화 능력이 CNN보다 뛰어나므로,
사용 가능할 때는 VLM의 재질+이물질 판단을 최종 결과로 우선한다.
HF_TOKEN이 없거나 API 실패 시 None을 반환해 CNN 판정으로 자연스럽게
폴백한다(app.py의 judge()가 처리).
"""
import base64
import io
import os

from PIL import Image

VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

MATERIALS = {"plastic", "can", "glass", "paper", "trash"}

JUDGE_PROMPT = (
    "이 사진 속 쓰레기 재질을 분류해줘. 다음 중 하나만 골라: "
    "plastic(플라스틱), can(캔/금속), glass(유리), paper(종이/골판지), "
    "trash(음식물 찌꺼기·기름때 등 이물질이 심해 재활용이 불가능한 상태). "
    "이물질(음식물 찌꺼기, 기름때, 흙 등)이 묻어 있는지도 판단해서, "
    "반드시 '재질,이물질여부' 형식으로 한 줄만 답해 (이물질여부는 yes 또는 no). "
    "예시: plastic,yes"
)


def vlm_judge(image: Image.Image) -> dict | None:
    """재질과 이물질 여부를 함께 판단. 사용 불가/실패 시 None(CNN 판정으로 폴백)."""
    token = os.getenv("HF_TOKEN")
    if not token:
        return None

    try:
        from huggingface_hub import InferenceClient

        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()

        client = InferenceClient(api_key=token)
        resp = client.chat.completions.create(
            model=VLM_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": JUDGE_PROMPT},
                ],
            }],
            max_tokens=20,
        )
        text = resp.choices[0].message.content.strip().lower().replace(" ", "")
        parts = text.split(",")
        material = parts[0] if parts[0] in MATERIALS else None
        if material is None:
            return None

        contaminated = len(parts) > 1 and parts[1] in ("yes", "y", "예", "true")
        return {"material": material, "contaminated": contaminated}
    except Exception:
        return None  # API 오류·모델 미지원 시 판정 생략 (CNN 결과로 폴백)
