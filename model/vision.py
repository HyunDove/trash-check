"""VLM 최종 판정 — 비전 모델로 재질+이물질을 함께 판단.

CNN(EfficientNet-B0)은 TrashNet(흰 배경 스튜디오 사진)으로 학습되어
실제 촬영 사진(다양한 배경·조명·오염)에서는 재질을 혼동하는 경우가 잦다
(예: 기름때 묻은 플라스틱 그릇을 캔으로 오분류). 범용 비전-언어 모델은
이런 실제 사진에 대한 일반화 능력이 CNN보다 뛰어나므로, 사용 가능할
때는 VLM의 재질+이물질 판단을 최종 결과로 우선한다.

LLM_BACKEND 환경변수로 로컬(Ollama)/배포(HF API) 이중화 — rag/chain.py의
get_llm()과 동일한 스위치. 로컬은 Ollama의 비전 지원 모델을 그대로 쓰고,
배포는 HF Inference Providers를 여러 모델 순서대로 시도한다(계정에 연결된
provider에 따라 특정 모델이 막힐 수 있어서). 어느 쪽이든 실패하면
material=None을 반환해 CNN 판정으로 자연스럽게 폴백한다(app.py의 judge()가 처리).
"""
import base64
import io
import os

from PIL import Image

OLLAMA_VLM_MODEL = "gemma4:e2b"

# HF Inference Providers는 계정에 연결된 provider에 따라 지원 모델이 달라
# 특정 모델이 "not supported by any provider" 오류를 낼 수 있다. 널리
# 서빙되는 비전 모델을 순서대로 시도해 하나가 막혀도 자동으로 다음으로 넘어간다.
VLM_MODELS = [
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "Qwen/Qwen2.5-VL-72B-Instruct",
    "Qwen/Qwen2-VL-7B-Instruct",
]

MATERIALS = {"plastic", "can", "glass", "paper", "trash"}

JUDGE_PROMPT = (
    "이 사진 속 쓰레기 재질을 분류해줘. 다음 중 하나만 골라: "
    "plastic(플라스틱), can(캔/금속), glass(유리), paper(종이/골판지), "
    "trash(음식물 찌꺼기·기름때 등 이물질이 심해 재활용이 불가능한 상태). "
    "이물질(음식물 찌꺼기, 기름때, 흙 등)이 묻어 있는지도 판단해서, "
    "반드시 '재질,이물질여부' 형식으로 한 줄만 답해 (이물질여부는 yes 또는 no). "
    "예시: plastic,yes"
)


def _parse_verdict(text: str) -> tuple:
    parts = text.lower().replace(" ", "").split(",")
    material = parts[0] if parts[0] in MATERIALS else None
    contaminated = len(parts) > 1 and parts[1] in ("yes", "y", "예", "true")
    return material, contaminated


def _vlm_judge_ollama(image: Image.Image) -> dict:
    import ollama

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    try:
        resp = ollama.chat(
            model=OLLAMA_VLM_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT, "images": [buf.getvalue()]}],
        )
        text = resp["message"]["content"].strip()
        material, contaminated = _parse_verdict(text)
        return {"material": material, "contaminated": contaminated, "raw": f"[ollama:{OLLAMA_VLM_MODEL}] {text}"}
    except Exception as e:
        return {"material": None, "contaminated": False, "raw": f"Ollama VLM 호출 실패: {e}"}


def _vlm_judge_hf(image: Image.Image) -> dict:
    token = os.getenv("HF_TOKEN")
    if not token:
        return {"material": None, "contaminated": False, "raw": "HF_TOKEN이 설정되지 않았습니다."}

    from huggingface_hub import InferenceClient

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    client = InferenceClient(api_key=token)

    errors = []
    for model_id in VLM_MODELS:
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        {"type": "text", "text": JUDGE_PROMPT},
                    ],
                }],
                max_tokens=20,
            )
            text = resp.choices[0].message.content.strip()
            material, contaminated = _parse_verdict(text)
            return {"material": material, "contaminated": contaminated, "raw": f"[{model_id}] {text}"}
        except Exception as e:
            errors.append(f"[{model_id}] {e}")

    return {"material": None, "contaminated": False, "raw": "모든 모델 실패:\n" + "\n".join(errors)}


def vlm_judge(image: Image.Image) -> dict:
    """재질과 이물질 여부를 함께 판단.

    반환: {"material": str|None, "contaminated": bool, "raw": str}.
    "material"이 None이면 VLM 판정 실패/사용 불가를 뜻하며, 호출측(app.py)이
    CNN 판정으로 폴백한다. "raw"는 원본 응답 또는 실패 사유를 담아 화면에
    디버그 표시할 수 있게 한다 — 실패가 조용히 묻히는 것을 방지.
    """
    if os.getenv("LLM_BACKEND", "ollama") == "hf":
        return _vlm_judge_hf(image)
    return _vlm_judge_ollama(image)
