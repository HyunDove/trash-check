# ♻️ 분리수거 판별기 + 환경 정책 안내 봇

> AI 응용 심화 소프트웨어 개발자 과정 — 딥러닝 + LLM(LangChain + RAG) 과제
> **기간**: 2026-07-02(목) ~ 2026-07-06(월) · **인원**: 1인

---

## 1. 프로젝트 개요

쓰레기 사진을 업로드하면 CNN 모델이 재질(플라스틱/캔/유리/종이 등)을 분류하고,
LangChain RAG 챗봇이 분리배출 가이드 문서를 근거로 올바른 배출 방법을 안내하는 서비스.

**예시 흐름**: 페트병 사진 업로드 → "플라스틱(PET)" 분류 →
"라벨을 제거하고 압착 후 투명 페트 전용 수거함에 배출하세요" (근거 문서 기반 답변)

## 2. 아키텍처

```
[사진 업로드]
      ↓
[YOLO 게이트] — COCO 사전학습 YOLOv8n 그대로 사용(학습 X), 물체 개수 판정
      ├─ 0개  → "쓰레기를 찾지 못했어요. 다시 업로드해주세요."
      ├─ 2개+ → "쓰레기가 여러 개 감지됐어요. 하나만 나오게 다시 업로드해주세요."
      └─ 1개  → 박스 영역 crop
      ↓
[CNN 분류 모델] — crop 이미지를 플라스틱/캔/유리/종이 분류 (직접 전이학습)
      ↓
분류 결과 + 사용자 질문
      ↓
[LangChain RAG] ← 벡터DB(Chroma) ← 분리배출 가이드 문서
      ↓
배출 방법 답변 생성 (Ollama LLM)
      ↓
[Streamlit UI: 이미지 업로드 + 챗봇]
```

## 3. 기술 스택 (확정)

| 파트 | 선택 | 비고 |
|---|---|---|
| 딥러닝 학습 | PyTorch + 전이학습(ResNet18 or EfficientNet-B0) | **Google Colab GPU**에서 학습 → 가중치(.pt) 다운로드 |
| 추론 | 로컬 CPU 추론 (학습된 .pt 로드) | 추론은 CPU로 충분 |
| 데이터셋 | TrashNet(6클래스, ~2,500장) 기본 / 부족 시 Kaggle Garbage Classification(12클래스) | **출처**: Hugging Face `garythung/trashnet`의 `dataset-resized.zip` (https://huggingface.co/datasets/garythung/trashnet). 원저작: Gary Thung & Mindy Yang, Stanford CS229 프로젝트 (원본 저장소: https://github.com/garythung/trashnet) |
| LLM | **이중화**: 로컬 데모=Ollama `qwen2.5:7b` / 배포=HF Inference API `Qwen/Qwen2.5-7B-Instruct` | `LLM_BACKEND` 환경변수로 스위칭. HF는 `langchain-huggingface`의 `HuggingFaceEndpoint`+`ChatHuggingFace`, `HF_TOKEN` 필요(Streamlit Secrets 등록, 하드코딩 금지) |
| RAG | LangChain + Chroma(로컬 벡터DB) + 임베딩 모델 | 임베딩도 로컬(예: BAAI/bge-m3 or ko-sbert) |
| RAG 문서 | `rag/docs/` 수집 완료: ① 환경부 「재활용품 분리배출 가이드라인」 PDF(4.8MB, 구로구청 게시본 https://www.guro.go.kr/raonkeditordata/2020/10/20201007_171440860_48470.pdf) ② 찾기쉬운 생활법령정보 재질별 분리배출 요령 md(easylaw.go.kr, 기준일 2026-06-15) | 필요 시 지자체 자료 추가 |
| UI / 배포 | **Streamlit** (`st.file_uploader` + `st.chat_input`/`st.chat_message`) | Streamlit Community Cloud 배포 목표 |

### 주요 결정 사항
- **YOLO는 파인튜닝하지 않음** → COCO 사전학습 모델을 물체 개수 게이트로만 사용 (박스 라벨링 데이터 불필요)
- **단일 객체 정책** → 물체가 2개 이상 탐지되면 재업로드 요청, 1개일 때만 crop 후 CNN 분류 → RAG 진행
- **CNN 입력은 YOLO crop 이미지** → 배경 제거로 분류 정확도 향상, 탐지/분류 역할 분리
- **CNN 전이학습이 과제의 딥러닝 학습 파트** → TrashNet(폴더 분류 데이터셋) 사용, 라벨링 작업 0
- **Gradio 미사용** → Streamlit 단독 (이미지 업로드·챗봇 모두 Streamlit 내장 기능으로 구현 가능, 배포 계획과 일치)
- **GPU 없음** → 학습은 Colab, 로컬은 추론 전용
- **LLM 이중화 (Qwen 고정)** → 로컬 데모=Ollama(`qwen2.5:7b`, 무료·무제한), 배포=HF Inference API(다운로드 0, Streamlit Cloud 동작). RAG 체인은 공유하고 `get_llm()`에서 `LLM_BACKEND` 환경변수로 분기
- **임베딩은 양쪽 모두 로컬** → `HuggingFaceEmbeddings`(sentence-transformers)는 경량이라 이원화 불필요

## 4. 일정

| 날짜 | 작업 |
|---|---|
| 목 (07-02) | 프로젝트 세팅, 데이터셋 다운로드·EDA, RAG용 문서 수집 |
| 금 (07-03) | Colab에서 CNN 전이학습·평가·튜닝 → 가중치 다운로드 |
| 토 (07-04) | LangChain RAG 파이프라인 (문서 로드 → 청킹 → 벡터DB → 체인) |
| 일 (07-05) | Streamlit 통합 (분류 결과를 RAG 컨텍스트로 주입), 엣지케이스 정리 |
| 월 (07-06) | 발표자료, 데모 시나리오, README, (가능 시) 배포 |

## 5. 예상 폴더 구조

```
trash-check/              # D:\AiWorkspace\individual\trash-check
  app.py                  # Streamlit 메인
  model/
    train_colab.ipynb     # Colab 학습 노트북
    best_model.pt         # 학습된 가중치
    detect.py             # YOLO 게이트 (개수 판정 + crop)
    predict.py            # CNN 추론 함수
  rag/
    ingest.py             # 문서 → 벡터DB 구축
    chain.py              # LangChain RAG 체인
    docs/                 # 분리배출 가이드 원문(PDF 등)
    chroma_db/            # 벡터DB (gitignore)
  docs/
    PROJECT.md            # 이 문서
  requirements.txt
  README.md
```

## 6. 진행 현황 (2026-07-02 기준)

### 완료
- [x] 프로젝트 세팅: venv, `requirements.txt`, `README.md`, `.gitignore`
- [x] 데이터셋 구축: `scripts/download_dataset.py` → `data/trashnet/` 4클래스 2,391장
- [x] RAG 문서 수집: `rag/docs/` 환경부 가이드라인 PDF + 생활법령정보 md
- [x] 전체 코드 스캐폴딩 (구문 검증 통과):
  - `model/train_colab.ipynb` — EfficientNet-B0 전이학습(클래스 가중치·체크포인트·평가 포함)
  - `model/detect.py` — YOLO 게이트(개수 판정: 0개/2개+ 재업로드, 1개 crop)
  - `model/predict.py` — CNN CPU 추론(`best_model.pt` 로드, 클래스명 내장)
  - `rag/ingest.py` — docs → Chroma 벡터DB(bge-m3 임베딩)
  - `rag/chain.py` — RAG 체인 + `get_llm()` 이중화(`LLM_BACKEND`=ollama/hf)
  - `app.py` — Streamlit 업로드 → 게이트 → 분류 → 챗봇

### 남은 작업 (실행 순서)
- [ ] `data\trashnet` zip 압축 → Colab에서 `train_colab.ipynb` 실행 → `best_model.pt`를 `model/`에 배치
- [ ] `pip install -r requirements.txt` (torch·ultralytics·langchain 계열 추가 설치)
- [ ] `python rag\ingest.py` 벡터DB 구축 (1회)
- [ ] `ollama pull qwen2.5:7b`
- [ ] `streamlit run app.py` 통합 테스트 (게이트 0개/1개/여러 개 시나리오)
- [ ] 발표 자료·데모 시나리오, (선택) HF API 배포

## 7. 남은 확인/리스크

- [x] LLM 모델 선정 → Qwen2.5 7B 확정 (로컬 `ollama pull qwen2.5:7b` ~4.7GB, D드라이브 용량 확인)
- [x] ⚠️ Streamlit Cloud 배포 시 Ollama 불가 → HF Inference API 이중화로 해소
- [ ] HF 토큰 발급 + Streamlit Cloud Secrets에 `HF_TOKEN` 등록
- [ ] HF 무료 티어 호출 제한 확인 (배포본은 시연 용도, 상시 서비스 아님)
- [x] TrashNet 클래스 매핑 확정 → plastic/metal→can/glass/paper+cardboard→paper, trash 제외. `scripts/download_dataset.py`로 `data/trashnet/` 구축 완료 (paper 998 · glass 501 · plastic 482 · can 410, 총 2,391장)
- [ ] paper 클래스 불균형(타 클래스 2배) → Colab 학습 시 클래스 가중치 or 언더샘플링 반영
- [ ] Colab 무료 GPU 세션 제한(런타임 끊김) 대비 — 체크포인트 저장 습관화
