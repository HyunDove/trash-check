# ♻️ 분리수거 판별 어시스턴트 (trash-check)

쓰레기 사진을 업로드하면 재질(플라스틱/캔/유리/종이)을 분류하고,
RAG 챗봇이 공식 가이드 문서를 근거로 올바른 분리배출 방법을 안내합니다.

> AI 응용 심화 소프트웨어 개발자 과정 개인 과제 (2026-07-02 ~ 07-06)

## 아키텍처

```
사진 업로드 → [YOLO 게이트] 물체 개수 판정 (0개/2개+ → 재업로드 안내)
                 └ 1개 → crop → [CNN] 재질 분류 → [RAG] 분리배출 방법 안내
```

- **YOLO 게이트**: COCO 사전학습 YOLOv8n 그대로 사용 (파인튜닝 없음)
- **CNN**: EfficientNet-B0 전이학습, TrashNet 4클래스 (Colab GPU 학습)
- **RAG**: LangChain + Chroma + bge-m3 임베딩, LLM은 Qwen2.5 7B
  - 로컬 데모: Ollama (`LLM_BACKEND=ollama`, 기본값)
  - 배포: HF Inference API (`LLM_BACKEND=hf` + `HF_TOKEN`)

## 실행 방법

```bash
# 1. 의존성 설치
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2. 데이터셋 다운로드 (CNN 학습용)
.venv\Scripts\python scripts\download_dataset.py

# 3. CNN 학습: model/train_colab.ipynb 를 Colab(GPU)에서 실행
#    → best_model.pt 다운로드 후 model/ 에 배치

# 4. RAG 벡터DB 구축 (1회)
.venv\Scripts\python rag\ingest.py

# 5. Ollama 모델 준비 (로컬 데모)
ollama pull qwen2.5:7b

# 6. 앱 실행
.venv\Scripts\streamlit run app.py
```

## 폴더 구조

```
app.py                  # Streamlit 메인 (업로드 → 게이트 → 분류 → 챗봇)
model/
  train_colab.ipynb     # Colab 전이학습 노트북
  detect.py             # YOLO 게이트 (개수 판정 + crop)
  predict.py            # CNN 추론 (best_model.pt)
  best_model.pt         # 학습된 가중치 (Colab 산출물, gitignore)
rag/
  ingest.py             # 문서 → Chroma 벡터DB 구축
  chain.py              # RAG 체인 + LLM 이중화(get_llm)
  docs/                 # 분리배출 가이드 원문
  chroma_db/            # 벡터DB (gitignore)
scripts/
  download_dataset.py   # TrashNet 다운로드 + 4클래스 재구성
data/trashnet/          # 학습 데이터 (gitignore)
docs/PROJECT.md         # 프로젝트 계획 문서
```

## 데이터 출처

- **TrashNet**: Gary Thung & Mindy Yang, Stanford CS229
  (Hugging Face: https://huggingface.co/datasets/garythung/trashnet · GitHub: https://github.com/garythung/trashnet)
- **환경부 「재활용품 분리배출 가이드라인」** (구로구청 게시본 PDF)
- **찾기쉬운 생활법령정보 — 자원재활용** (https://easylaw.go.kr, 기준일 2026-06-15)
