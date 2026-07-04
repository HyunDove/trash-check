"""VLM 이물질 판정 — HF Inference API의 비전 모델로 오염 여부를 확인.

재질 분류(CNN)와 별개로, 이물질(음식물·오염물)이 묻어 있으면 재활용이
불가하므로 일반쓰레기로 안내한다. HF_TOKEN이 없거나 API 실패 시 None을
반환해 판정을 건너뛴다(앱은 신뢰도 임계값 휴리스틱으로만 동작).
"""
import base64
import io
import os

from PIL import Image

VLM_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "이 사진은 재활용 분리배출하려는 쓰레기입니다. "
    "음식물 찌꺼기, 기름때, 흙 등 이물질이 눈에 띄게 묻어 있습니까? "
    "반드시 '예' 또는 '아니오' 한 단어로만 답하세요."
)


def check_contamination(image: Image.Image) -> bool | None:
    """이물질이 묻어 있으면 True, 깨끗하면 False, 판정 불가면 None."""
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
                    {"type": "text", "text": PROMPT},
                ],
            }],
            max_tokens=10,
        )
        answer = resp.choices[0].message.content.strip()
        if answer.startswith("예"):
            return True
        if answer.startswith("아니"):
            return False
        return None
    except Exception:
        return None  # API 오류·모델 미지원 시 판정 생략 (앱 흐름은 계속)
