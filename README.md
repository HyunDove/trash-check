# ♻️ 분리수거 판별 어시스턴트

> **YOLO 게이트 + CNN 분류 + LangChain RAG** 로 쓰레기 재질을 판별하고 분리배출 방법을 안내하는 딥러닝 + LLM 과제입니다.

<table>
  <tr>
    <td>📅 <b>기간</b></td>
    <td>2026년 07월 02일 ~ 07월 06일 (5일)</td>
    <td>👤 <b>팀원</b></td>
    <td>1인 · 김승현</td>
  </tr>
</table>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EfficientNet--B0-EE4C2C?logo=pytorch&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?logo=googlecolab&logoColor=white)
![YOLO](https://img.shields.io/badge/Ultralytics-YOLOv8n-00FFFF?logo=yolo&logoColor=black)
![LangChain](https://img.shields.io/badge/LangChain-RAG-1C3C3C?logo=langchain&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-Qwen2.5_7B-000000?logo=ollama&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)

---

## 📋 과제 개요

| 항목 | 내용 |
|---|---|
| 🎯 **주제** | 쓰레기 사진 업로드 → 재질 판별 → 분리배출 방법 RAG 안내 |
| 🤖 **탐지** | YOLOv8n (COCO 사전학습, 파인튜닝 없음) — 물체 개수 게이트 |
| 🧠 **분류** | CNN 전이학습 (EfficientNet-B0) — 플라스틱/캔/유리/종이 4클래스 |
| 💬 **LLM** | LangChain RAG + Qwen2.5 7B (로컬 Ollama / 배포 HF Inference API 이중화) |
| 🖥️ **학습 환경** | Google Colab T4 GPU (로컬 GPU 없음, 추론은 CPU) |
| 🚀 **결과물** | Streamlit 데모 앱 (이미지 업로드 + 챗봇) |

---

## 🗺️ 전체 파이프라인

```
📷 사진 업로드
     ↓
🔍 YOLO 게이트 (model/detect.py) — COCO 사전학습 YOLOv8n, 학습 없이 그대로 사용
   물체 개수 판정
     ├─ 0개  → "쓰레기를 찾지 못했어요. 다시 업로드해주세요."
     ├─ 2개+ → "쓰레기가 여러 개 감지됐어요. 하나만 나오게 다시 업로드해주세요."
     └─ 1개  → 박스 영역 crop
     ↓
🧠 CNN 분류 (model/predict.py) — EfficientNet-B0 전이학습, TrashNet 4클래스
   plastic / can / glass / paper
     ↓
📚 LangChain RAG (rag/chain.py) ← Chroma 벡터DB ← 분리배출 가이드 문서
   분류 결과 + 사용자 질문 → 근거 문서 검색 → 배출 방법 답변 생성
     ↓
🌐 Streamlit UI (app.py) — 이미지 업로드 + 챗봇 한 화면에서 제공
```

### 왜 이 구조인가

| 결정 | 이유 |
|---|---|
| YOLO는 파인튜닝하지 않음 | 박스 라벨링 데이터가 필요 없어 일정(5일) 내 완주 가능. COCO 사전학습만으로 "개수 세기"는 충분 |
| CNN 입력은 YOLO crop 이미지 | 배경 제거로 분류 정확도 향상, 탐지·분류 역할을 분리해 설계 의도가 명확 |
| 여러 개 탐지 시 재업로드 요청 | 한 장에 하나의 쓰레기만 정확히 판별하는 UX 정책 |
| CNN 전이학습이 딥러닝 학습 파트 | TrashNet(폴더 분류 데이터셋)이라 라벨링 작업 0으로 학습까지 진행 |
| LLM 이중화 (Qwen2.5 7B 고정) | 로컬 데모=Ollama(무료·무제한), 배포=HF Inference API(다운로드 0, Streamlit Cloud 등 배포 환경에서도 동작) |
| Gradio 대신 Streamlit | 이미지 업로드·챗봇 모두 Streamlit 내장 기능으로 구현 가능해 배포 계획과 일치 |

---

## 📁 프로젝트 구조

```
trash-check/
│
├── 📄 app.py                        # Streamlit 메인 (업로드 → 게이트 → 분류 → 챗봇)
├── 📄 requirements.txt              # 로컬 추론/서비스 의존성
│
├── 📂 model/
│   ├── train_colab.ipynb            # Colab 학습 노트북 (클래스 가중치·체크포인트·평가 포함)
│   ├── detect.py                    # YOLO 게이트 (개수 판정 + crop)
│   ├── predict.py                   # CNN 추론 (best_model.pt 로드, CPU)
│   └── best_model.pt                # 학습된 가중치 (Colab 산출물, git 제외)
│
├── 📂 rag/
│   ├── ingest.py                    # 문서 → Chroma 벡터DB 구축
│   ├── chain.py                     # RAG 체인 + get_llm() 이중화(LLM_BACKEND)
│   ├── docs/                        # 분리배출 가이드 원문 (PDF·md)
│   └── chroma_db/                   # 벡터DB (git 제외)
│
├── 📂 scripts/
│   └── download_dataset.py          # TrashNet 다운로드 + 4클래스 재구성
│
├── 📂 data/trashnet/                 # 학습 데이터 (git 제외)
└── 📂 docs/
    └── PROJECT.md                   # 프로젝트 계획·진행 현황 문서
```

---

## ⚙️ 실행 방법

### 1️⃣ 의존성 설치

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### 2️⃣ 데이터셋 다운로드 (CNN 학습용)

```bash
.venv\Scripts\python scripts\download_dataset.py
```

### 3️⃣ Google Colab에서 CNN 학습

1. `model/train_colab.ipynb` 을 Colab에 업로드
2. 런타임 → **T4 GPU** 선택
3. `data/trashnet/` 를 zip으로 압축해 노트북 첫 셀에서 업로드
4. 전체 셀 실행 → 학습 완료 후 `best_model.pt` 다운로드
5. 다운로드한 `best_model.pt` 를 로컬 `model/` 폴더에 배치

### 4️⃣ RAG 벡터DB 구축 (1회)

```bash
.venv\Scripts\python rag\ingest.py
```

### 5️⃣ Ollama 모델 준비 (로컬 데모)

```bash
ollama pull qwen2.5:7b
```

### 6️⃣ 앱 실행

```bash
.venv\Scripts\streamlit run app.py
```

> 배포 환경(Ollama 미지원)에서는 `LLM_BACKEND=hf` + `HF_TOKEN` 환경변수로 HF Inference API를 사용합니다.

---

## 📊 데이터셋

TrashNet 6클래스를 과제 4클래스로 매핑해 `data/trashnet/`에 구축했습니다.

| 클래스 | 장수 | 원본 TrashNet 클래스 |
|---|---:|---|
| paper (종이) | 998 | paper + cardboard |
| glass (유리) | 501 | glass |
| plastic (플라스틱) | 482 | plastic |
| can (캔) | 410 | metal |
| **총합** | **2,391** | (trash 클래스는 분류 대상 제외) |

> paper 클래스가 다른 클래스의 약 2배라 Colab 학습 시 클래스 가중치(class weight)를 적용합니다 (`train_colab.ipynb` 3번 셀).

---

## ✅ 진행 현황

- [x] 프로젝트 세팅 (venv, requirements.txt, .gitignore)
- [x] 데이터셋 구축 (`scripts/download_dataset.py` → 4클래스 2,391장)
- [x] RAG 문서 수집 (`rag/docs/` 환경부 가이드라인 PDF + 생활법령정보 md)
- [x] 전체 코드 스캐폴딩 (YOLO 게이트 · CNN 추론 · RAG 체인 · Streamlit 앱)
- [ ] Colab CNN 전이학습 실행 → `best_model.pt` 확보
- [ ] RAG 벡터DB 구축 및 검색 품질 확인
- [ ] Ollama 로컬 데모 통합 테스트 (게이트 0개/1개/여러 개 시나리오)
- [ ] HF 토큰 발급 및 배포 이중화 테스트
- [ ] 발표 자료·데모 시나리오 정리

세부 계획과 리스크는 [`docs/PROJECT.md`](docs/PROJECT.md) 참고.

---

## 📚 데이터 출처

- **TrashNet**: Gary Thung & Mindy Yang, Stanford CS229
  (Hugging Face: https://huggingface.co/datasets/garythung/trashnet · GitHub: https://github.com/garythung/trashnet)
- **환경부 「재활용품 분리배출 가이드라인」** (구로구청 게시본 PDF)
- **찾기쉬운 생활법령정보 — 자원재활용** (https://easylaw.go.kr, 기준일 2026-06-15)
