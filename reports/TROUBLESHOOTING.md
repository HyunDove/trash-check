# 🔧 트러블슈팅

과제 진행 중 겪은 주요 이슈와 해결 과정을 정리했습니다. 각 사례는 "증상 → 원인 → 해결" 순서로 기록합니다.

---

## 1. Streamlit Cloud 배포 시 `cv2` import 실패

**증상**

```
File "/mount/src/trash-check/model/detect.py", line 9, in <module>
    from ultralytics import YOLO
File ".../ultralytics/utils/__init__.py", line 24, in <module>
    import cv2
...
ImportError: libGL.so.1: cannot open shared object file
```

**원인**

`ultralytics`가 기본으로 끌어오는 `opencv-python`은 GUI(X11) 빌드라서, 로컬에서는 문제없이 동작하지만 GUI가 없는 Streamlit Cloud의 최소 리눅스 이미지에는 `libGL.so.1` 등 X11 관련 공유 라이브러리가 아예 없어 import 단계에서 죽는다.

1차로 `packages.txt`에 `libgl1`, `libglib2.0-0`을 추가했으나, Streamlit Cloud의 실제 배포판(Debian trixie)에서 `libglib2.0-0`이 요구하는 구버전 `libffi7`을 찾지 못해 **apt 의존성 충돌**(`unmet dependencies`)로 설치 자체가 실패하는 2차 문제가 이어졌다.

**해결**

- `packages.txt`는 실제로 필요한 `libgl1` 하나만 남김 (glib은 base 이미지에 이미 호환 버전 존재)
- `requirements.txt`에 `ultralytics` 다음 줄로 `opencv-python-headless`를 명시 — GUI 의존성이 없는 서버용 빌드로, pip 설치 순서상 나중에 설치되며 같은 `cv2` 모듈 경로를 덮어써 X11 요구 자체를 없앰

```
# packages.txt
libgl1

# requirements.txt (일부)
ultralytics
opencv-python-headless
```

---

## 2. YOLO 게이트가 실제 쓰레기 사진을 자꾸 오판

**증상**

- 캔·종이·일반쓰레기 등 흔한 사진을 올려도 "쓰레기를 찾지 못했어요"가 뜸 (0개 감지)
- 반대로 배경에 리모컨·컵 등이 함께 찍힌 사진은 "쓰레기가 4개 감지됐어요"로 거부됨

**원인**

YOLOv8n은 COCO 80개 클래스(사람·자동차·병·컵 등 일반 사물)로 학습된 사전학습 모델이다. 그런데 COCO 80개 클래스에는 **캔·종이박스·일반쓰레기 같은 카테고리가 아예 없다.** 그래서:

1. 실제로 물체가 하나 있어도 COCO 클래스에 해당하지 않으면 0개로 판정
2. 배경의 리모컨·컵처럼 COCO에 있는 물체까지 함께 잡히면 "여러 개"로 오인

앱은 이미 "한 장에 하나만 촬영" UX를 전제하고 있었으므로, YOLO를 물체 분류기가 아니라 순수한 **개수/위치 힌트**로만 쓰는 방향으로 정리했다.

**해결**

- 0개 감지 시 실패 처리 대신 **전체 이미지를 그대로 CNN 입력**으로 사용 (`model/detect.py`)
- 여러 개 감지돼도 재업로드를 요구하지 않고, **바운딩 박스 면적이 가장 큰 물체 하나만 자동으로 선택**해 crop
- crop 좌표 계산이 실패하거나 가로/세로가 0인 예외 상황에도 원본 이미지로 안전하게 폴백하도록 `try/except` 방어 추가

```python
if len(boxes) == 0:
    return GateResult(status="ok", count=1, crop=image)  # 폴백

crop = image
try:
    areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in boxes.xyxy.tolist()]
    x1, y1, x2, y2 = boxes.xyxy.tolist()[areas.index(max(areas))]
    candidate = image.crop((int(x1), int(y1), int(x2), int(y2)))
    if candidate.width > 0 and candidate.height > 0:
        crop = candidate
except Exception:
    pass  # 원본 이미지로 안전 폴백
```

---

## 3. VLM(비전-언어 모델) 도입과 HF Inference Provider 제약

**배경**

CNN(EfficientNet-B0)은 TrashNet(흰 배경 스튜디오 사진)으로 학습되어, 실제 촬영 사진(다양한 배경·조명·오염)에서 재질을 혼동하는 사례가 반복됐다 (예: 기름때 묻은 플라스틱 그릇을 캔으로 오분류). 데이터 증강을 강화해 재학습해봤지만 검증셋 기준 개선이 뚜렷하지 않았다. 그래서 범용 비전-언어 모델(VLM)을 재질+이물질 판정의 **최종 판단자**로 승격하는 방향으로 전환했다.

**증상 1 — 특정 모델이 항상 실패**

```
{'message': "The requested model 'Qwen/Qwen2.5-VL-7B-Instruct' is not
supported by any provider you have enabled.", 'code': 'model_not_supported'}
```

**원인 1**: HF Inference Providers는 사용자 계정에 **활성화된 provider**(Together, Fireworks, Novita 등)에 따라 실제로 호출 가능한 모델이 달라진다. 특정 모델을 서빙하는 provider가 하나도 활성화되어 있지 않으면 모델 자체가 존재해도 이 오류가 난다.

**증상 2 — 다른 모델로 바꿔도 결제 오류**

```
Client error '402 Payment Required' ...
You have depleted your monthly included credits.
```

**원인 2**: 무료 요금제의 월간 포함 크레딧을 소진한 상태. 이건 코드 문제가 아니라 계정 과금 이슈라 provider를 더 활성화하거나 다음 달 크레딧이 초기화될 때까지 기다려야 한다.

**해결**

1. **여러 후보 모델을 순서대로 시도**하도록 `vlm_judge()`를 구성 (`Llama-3.2-11B-Vision-Instruct` → `Qwen2.5-VL-72B-Instruct` → `Qwen2-VL-7B-Instruct`). 하나가 막혀도 자동으로 다음 모델로 넘어감
2. **실패를 조용히 숨기지 않고 원본 오류를 그대로 노출** — `vlm_judge()`가 항상 `{"material", "contaminated", "raw"}` 형태의 dict를 반환하도록 하고, `raw` 필드에 성공 응답 또는 실패 사유(어떤 모델이 왜 실패했는지)를 담아 앱의 "🔧 VLM 디버그 정보" expander에서 바로 확인 가능하게 함
3. **VLM 전면 실패 시에도 앱은 정상 동작** — `material`이 `None`이면 CNN 분류 + 확신도 임계값(70%) 방식으로 자연스럽게 폴백하도록 설계해, HF 계정의 provider/크레딧 상태와 무관하게 앱 자체는 항상 동작

```python
# 실패해도 dict 반환, raw에 원인 기록 → 디버깅 가능 + CNN 폴백 안전
{"material": None, "contaminated": False, "raw": "API 오류: ..."}
```

> **교훈**: 외부 API를 폴백 설계에 넣을 때는 실패 사유를 화면(또는 로그)에서 바로 확인할 수 있어야 한다. `except Exception: return None`처럼 조용히 삼키면, "왜 안 되지?"를 알아내는 데만 여러 차례의 시행착오가 필요해진다.

---

## 4. RAG 검색이 플라스틱/캔 질의에서 무관한 문서만 반환

**증상**

챗봇이 플라스틱·캔 재질에 대해 "이 재질의 기본 분리배출 방법을 알려줘"라고 물으면, `rag/docs/재질별_분리배출_요령_생활법령.md`의 전용 섹션(`## 플라스틱류 분리배출 요령`, `## 캔·금속류 분리배출 요령`)이 검색 결과에 전혀 나오지 않고, 매번 PDF의 도자기류·코르크따개·돋보기 같은 알파벳순 잡화 안내만 반환됐다. 유리·종이 질의는 정상적으로 관련 문서가 검색됐다.

**원인**

`similarity_search_with_score`로 실제 유사도 점수를 확인해보니, 플라스틱 관련 청크가 전체 82개 청크 중 **77위**(거리 9.58, 1위는 4.19)로 사실상 최하위권이었다. 원인은 환경부 가이드라인 PDF의 26~33페이지("분리 배출 품목 사전" 부록 — 나무젓가락부터 항아리까지 A-Z로 나열된 잡화 안내표)에 있었다. 이 페이지들은 "배출/종량제" 같은 단어를 청크 하나당 10회 이상 반복하는 구조라, 다국어 MiniLM 임베딩이 이 반복 키워드 밀도에 강하게 끌려가면서 실제 재질별 요령 문서를 항상 밀어냈다.

**해결**

`rag/ingest.py`의 `load_documents()`에서 PDF 로드 시 26~33페이지(0-index)를 `EXCLUDED_PDF_PAGES`로 제외하고 벡터DB를 재구축했다. 그 결과 82청크 → 54청크로 줄었고, 플라스틱 질의에서도 PDF의 정식 "합성수지류 PET/PVC/PE..." 재질별 요령 챕터가 top-8에 정상적으로 검색됐다. 검색 결과 개수를 늘리는 `k=4 → k=8`(`rag/chain.py`) 조정도 함께 적용했지만, 근본 원인은 페이지 제외였다 — k만 올렸을 때는 개선되지 않았다.

```python
# rag/ingest.py
EXCLUDED_PDF_PAGES = set(range(26, 34))

def load_documents():
    ...
    if path.suffix == ".pdf":
        pdf_docs = PyPDFLoader(str(path)).load()
        docs.extend(d for d in pdf_docs if d.metadata.get("page") not in EXCLUDED_PDF_PAGES)
```

> **교훈**: RAG 검색 품질이 나쁠 때 k값부터 늘리고 싶어지지만, 원인이 특정 문서의 "구조적 노이즈"(반복 키워드가 많은 색인·표·부록 등)라면 k를 아무리 늘려도 근본적으로 해결되지 않는다. `similarity_search_with_score`로 실제 순위와 거리 값을 까보는 진단이 먼저다.

---

## 5. 로컬 실행 시 VLM이 항상 "HF_TOKEN이 설정되지 않았습니다"로 실패

**증상**

로컬에서 `streamlit run app.py`로 실행하면 챗봇(LLM)은 `LLM_BACKEND` 기본값이 `ollama`라 로컬 Ollama로 잘 동작하는데, VLM 재질 판정만 "🔧 VLM 디버그 정보"에 `HF_TOKEN이 설정되지 않았습니다`가 뜨며 항상 CNN 폴백으로만 동작했다.

**원인**

`model/vision.py`의 `vlm_judge()`가 애초에 HF Inference API 전용으로만 구현돼 있었다. `rag/chain.py`의 `get_llm()`은 `LLM_BACKEND` 환경변수로 로컬/배포를 이중화했지만, VLM 쪽은 그 패턴이 적용되지 않은 상태였다.

**해결**

`vlm_judge()`를 `_vlm_judge_ollama()` / `_vlm_judge_hf()`로 분리하고, `rag/chain.py`와 동일하게 `LLM_BACKEND` 환경변수(기본값 `ollama`)로 분기하도록 변경했다. 로컬은 `ollama` 파이썬 클라이언트로 vision capability가 있는 로컬 모델(`gemma4:e2b`)에 이미지를 직접 전달하고, 배포(`LLM_BACKEND=hf`)는 기존 HF 다중 모델 폴백 로직을 그대로 유지한다.

```python
def vlm_judge(image: Image.Image) -> dict:
    if os.getenv("LLM_BACKEND", "ollama") == "hf":
        return _vlm_judge_hf(image)
    return _vlm_judge_ollama(image)
```

> **교훈**: 로컬/배포 이중화 패턴을 한 모듈(`rag/chain.py`)에 도입했다면, 같은 환경변수에 의존하는 다른 모듈(`model/vision.py`)에도 일관되게 적용해야 한다. 그렇지 않으면 배포 기준으로만 잘 동작하고 로컬 데모는 항상 폴백 경로만 타는 상황이 생긴다.
