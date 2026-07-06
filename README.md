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
![VLM](https://img.shields.io/badge/VLM-재질%2B이물질_최종판정-6A1B9A)

---

## 📄 과제 보고서

[![과제 보고서](https://img.shields.io/badge/과제_보고서-v1.0-2E7D32?logo=github&logoColor=white)](https://github.com/HyunDove/trash-check/releases/tag/v1.0.0)

YOLO 게이트·CNN 전이학습·VLM 최종 판정·LangChain RAG 설계·구현·결과를 정리한 수행내역서입니다. 📄 [수행내역서.md 다운로드 →](reports/수행내역서.md)

---

## 🔗 바로가기

| 구분 | 링크 |
|---|---|
| 🌐 **Streamlit 시연 앱** | [trash-check.streamlit.app](https://trash-check.streamlit.app/) |
| 📓 **Colab 학습 노트북** | [Google Drive](https://drive.google.com/file/d/1qO4Hd_4_81Lnw8OfeLuaXswqD9xJ07ZB/view?usp=sharing) |
| 🧠 **학습된 모델** | [model/best_model.pt](model/best_model.pt) |
| 📦 **Release (발표자료 PDF·수행내역서)** | [v1.0.0](https://github.com/HyunDove/trash-check/releases/tag/v1.0.0) |

---

## 📋 과제 개요

| 항목 | 내용 |
|---|---|
| 🎯 **주제** | 쓰레기통에 사진을 던져 넣으면 재질을 판별해 알맞은 분리수거함으로 안내 |
| 🤖 **탐지** | YOLOv8n (COCO 사전학습, 파인튜닝 없음) — 물체 개수 게이트 |
| 🧠 **분류** | CNN 전이학습 (EfficientNet-B0) — 플라스틱/캔/유리/종이/일반쓰레기 5클래스 |
| 👁️ **최종 판정** | VLM(로컬 Ollama / 배포 HF API 이중화, HF는 다중 모델 폴백)이 재질+이물질을 함께 판단해 CNN 결과를 보정, 실패 시 CNN+확신도 임계값 폴백 |
| 💬 **LLM** | LangChain RAG + Qwen2.5 7B (로컬 Ollama / 배포 HF Inference API 이중화) |
| 🖥️ **학습 환경** | Google Colab T4 GPU (로컬 GPU 없음, 추론은 CPU) |
| 🚀 **결과물** | Streamlit 데모 앱 — 쓰레기통 드래그드롭 업로드 + 투입 애니메이션 + 챗봇 |

---

## 🗺️ 전체 파이프라인

```
📷 사진 업로드 (큰 쓰레기통 투입구 — 클릭 또는 드래그드롭)
     ↓
🔍 YOLO 게이트 (model/detect.py) — COCO 사전학습 YOLOv8n, 학습 없이 그대로 사용
   물체 개수와 무관하게 항상 crop 하나를 만들어냄 (conf=0.25)
     ├─ 0개 감지 → COCO 미포함 물체(캔·종이 등)로 흔함 → 전체 이미지를 그대로 사용
     └─ 1개+ 감지 → 바운딩 박스 면적이 가장 큰 물체 하나만 자동 선택해 crop
     ↓
🧠 CNN 분류 (model/predict.py) — EfficientNet-B0 전이학습, TrashNet 5클래스
   plastic / can / glass / paper / trash
     ↓
👁️ VLM 최종 판정 (model/vision.py) — 재질+이물질을 한 번에 판단 (로컬 Ollama / 배포 HF API 이중화)
   VLM 사용 가능 시 그 결과를 최종 채택, 실패 시에만 CNN+확신도 70% 임계값 폴백
     ↓
🗑️ 쓰레기통 투입 애니메이션 — crop 이미지가 알맞은 분리수거함으로 날아가 들어감
     ↓
📚 LangChain RAG (rag/chain.py) ← Chroma 벡터DB ← 분리배출 가이드 문서
   분류 결과 + 사용자 질문 → 근거 문서 검색 → 배출 방법 답변 생성
     ↓
🌐 Streamlit UI (app.py) — 판별 데모 / 학습 성과 / 모델 구조 3탭
```

### 왜 이 구조인가

| 결정 | 이유 |
|---|---|
| YOLO는 파인튜닝하지 않음 | 박스 라벨링 데이터가 필요 없어 일정(5일) 내 완주 가능. COCO 사전학습만으로 "개수 세기"는 충분 |
| CNN 입력은 YOLO crop 이미지 | 배경 제거로 분류 정확도 향상, 탐지·분류 역할을 분리해 설계 의도가 명확 |
| 여러 개 감지돼도 가장 큰 물체만 자동 선택 | COCO 미포함 배경 물체(리모컨·컵 등)까지 잡혀 재업로드를 반복 요구하던 문제 해결 — [트러블슈팅](reports/TROUBLESHOOTING.md) 참고 |
| CNN 전이학습이 딥러닝 학습 파트 | TrashNet(폴더 분류 데이터셋)이라 라벨링 작업 0으로 학습까지 진행 |
| VLM을 최종 판단자로 승격 | CNN은 스튜디오 사진 학습이라 실사진에서 재질을 혼동(예: 기름때 묻은 플라스틱→캔). VLM 사용 가능 시 그 판단으로 확정, 불가 시 CNN+임계값 폴백 — [트러블슈팅](reports/TROUBLESHOOTING.md) 참고 |
| 쓰레기통 투입 UX | file_uploader를 쓰레기통 모양으로 리스타일 + SVG 분리수거함 5종 애니메이션으로 실제 분리배출 행위를 은유 |
| LLM 이중화 (Qwen2.5 7B 고정) | 로컬 데모=Ollama(무료·무제한), 배포=HF Inference API(다운로드 0, Streamlit Cloud 등 배포 환경에서도 동작) |
| Gradio 대신 Streamlit | 이미지 업로드·챗봇 모두 Streamlit 내장 기능으로 구현 가능해 배포 계획과 일치 |

---

## 📁 프로젝트 구조

```
trash-check/
│
├── 📄 app.py                        # Streamlit 메인 (쓰레기통 UI → 게이트 → 분류 → 판정 → 챗봇)
├── 📄 requirements.txt              # 로컬 추론/서비스 의존성
├── 📄 packages.txt                  # Streamlit Cloud apt 의존성 (libgl1 등, cv2 실행용)
│
├── 📂 .streamlit/
│   └── config.toml                  # 에코 그린 테마
│
├── 📂 model/
│   ├── train_colab.ipynb            # Colab 학습 노트북 (클래스 가중치·체크포인트·평가·그래프 저장)
│   ├── detect.py                    # YOLO 게이트 (개수 판정 + crop)
│   ├── predict.py                   # CNN 추론 (best_model.pt 로드, CPU)
│   ├── vision.py                    # VLM 재질+이물질 판정 (LLM_BACKEND 이중화: 로컬 Ollama / 배포 HF API)
│   └── best_model.pt                # 학습된 가중치 (5클래스, Colab 산출물)
│
├── 📂 rag/
│   ├── ingest.py                    # 문서 → Chroma 벡터DB 구축
│   ├── chain.py                     # RAG 체인 + get_llm() 이중화(LLM_BACKEND)
│   ├── docs/                        # 분리배출 가이드 원문 (PDF·md)
│   └── chroma_db/                   # 벡터DB (git 제외)
│
├── 📂 scripts/
│   └── download_dataset.py          # TrashNet 다운로드 + 5클래스 재구성
│
├── 📂 reports/                       # Colab 산출 발표용 그래프 + TROUBLESHOOTING.md
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

> 실제 학습이 진행된 노트북(실행 결과·로그 포함)은 [Google Drive](https://drive.google.com/file/d/1qO4Hd_4_81Lnw8OfeLuaXswqD9xJ07ZB/view?usp=sharing)에서도 확인할 수 있습니다.

1. `data/trashnet/`의 5개 클래스 폴더(`can`, `glass`, `paper`, `plastic`, `trash`)를 그대로
   Google Drive의 `MyDrive/trash-check/dataset/` 아래에 업로드 (zip 압축 불필요)
2. `model/train_colab.ipynb` 을 Colab에 업로드, 런타임 → **T4 GPU** 선택
3. 전체 셀 실행 (Drive 마운트 → 데이터를 로컬 디스크로 복사 → 학습 → 평가 → 발표용 그래프 저장)
4. 학습 완료 후 `MyDrive/trash-check/best_model.pt`를 다운로드해 로컬 `model/best_model.pt`에 배치
5. `MyDrive/trash-check/reports/`의 학습 곡선·혼동행렬·클래스별 지표 PNG도 발표 자료용으로 함께 다운로드

### 4️⃣ RAG 벡터DB 구축 (1회)

```bash
.venv\Scripts\python rag\ingest.py
```

### 5️⃣ Ollama 모델 준비 (로컬 데모)

```bash
ollama pull qwen2.5:7b   # 챗봇 LLM
ollama pull gemma4:e2b   # VLM 재질+이물질 판정 (vision capability 필요)
```

### 6️⃣ 앱 실행

```bash
.venv\Scripts\streamlit run app.py
```

> 배포 환경(Ollama 미지원)에서는 `LLM_BACKEND=hf` + `HF_TOKEN` 환경변수로 HF Inference API를 사용합니다.
> 챗봇 LLM과 VLM 재질 판정 모두 이 환경변수 하나로 함께 전환됩니다(`rag/chain.py`, `model/vision.py` 동일 패턴).
> HF 크레딧 소진 등으로 VLM 호출이 실패해도 CNN 판정으로 자동 폴백되어 앱은 정상 동작합니다.

### 7️⃣ Streamlit Cloud 배포

1. share.streamlit.io에서 저장소 연결 후 **Settings → Secrets**에 등록 (필수 — 없으면 챗봇이 로컬 Ollama로 연결을 시도하다 실패):
   ```toml
   LLM_BACKEND = "hf"
   HF_TOKEN = "hf_..."
   ```
2. `packages.txt`(`libgl1`)와 `requirements.txt`의 `opencv-python-headless`가 `ultralytics`의 OpenCV 의존성(libGL 등 X11 라이브러리) 문제를 해결합니다.
3. 첫 부팅 시 벡터DB가 없으면 자동으로 구축됩니다 (`app.py`의 `load_rag()`).
4. LLM 호출이 실패해도(Secrets 누락, API 장애 등) `safe_ask()`가 감싸서 앱이 죽지 않고 챗봇에 안내 메시지만 표시됩니다.

---

## 📊 데이터셋

TrashNet 6클래스를 과제 5클래스로 매핑해 `data/trashnet/`에 구축했습니다 (일반쓰레기 판정을 위해 `trash` 클래스 포함).

| 클래스 | 장수 | 원본 TrashNet 클래스 |
|---|---:|---|
| paper (종이) | 997 | paper + cardboard |
| glass (유리) | 501 | glass |
| plastic (플라스틱) | 482 | plastic |
| can (캔) | 410 | metal |
| trash (일반쓰레기) | 137 | trash |
| **총합** | **2,527** | |

> 클래스 불균형(paper가 trash의 약 7배)이 커서 Colab 학습 시 클래스 가중치(class weight)를 적용합니다 (`train_colab.ipynb` 3번 셀).

---

## 📈 학습 성과 (EfficientNet-B0 · Colab T4 · 15 epoch)

**전체 검증 정확도 93.66%**

| 그래프 | 내용 |
|---|---|
| ![학습 곡선](reports/training_curve.png) | Train Loss는 꾸준히 감소, Val Accuracy는 8 epoch 근처부터 93% 안팎으로 수렴 |
| ![혼동행렬](reports/confusion_matrix.png) | paper·glass·can은 대각선에 집중되지만 plastic↔can 오분류가 일부 남아있음 (아래 참고) |
| ![클래스별 지표](reports/class_metrics.png) | paper·glass·can 90%대, trash는 데이터가 가장 적은 클래스(137장)임에도 F1 80%대 확보 |

> **plastic↔can 혼동에 대해**: 증강을 강화(RandomResizedCrop·강한 ColorJitter·RandomErasing 등)해 재학습했지만, TrashNet(흰 배경 스튜디오 사진)과 실제 촬영 사진의 도메인 차이 때문에 검증셋 기준으로도 뚜렷한 개선은 없었다. 그래서 실제 서비스에서는 **VLM을 최종 판단자로 승격**해 이 한계를 보완한다 — 자세한 경위는 [트러블슈팅 #3](reports/TROUBLESHOOTING.md) 참고.

---

## 🔧 트러블슈팅

과제 진행 중 겪은 주요 이슈 5건을 정리했습니다. 전체 내용은 [`reports/TROUBLESHOOTING.md`](reports/TROUBLESHOOTING.md) 참고.

| # | 이슈 | 한 줄 요약 |
|---|---|---|
| 1 | Streamlit Cloud `cv2` import 실패 | `ultralytics`의 GUI용 `opencv-python`이 서버 환경에서 `libGL.so.1` 부재로 실패 → `opencv-python-headless` + `packages.txt(libgl1)`로 해결 |
| 2 | YOLO 게이트 오판 | COCO 80클래스에 캔·종이 등이 없어 0개/여러 개로 자꾸 오판 → 0개는 전체 이미지 폴백, 여러 개는 최대 박스 자동 선택으로 전환 |
| 3 | VLM 도입과 HF Provider 제약 | 특정 모델이 계정에 미지원(model_not_supported)이거나 월 크레딧 소진(402)으로 실패 → 여러 모델 순차 시도 + 실패 사유를 화면에 노출 + CNN 폴백으로 항상 동작 보장 |
| 4 | RAG 검색이 플라스틱/캔 질의에서 무관한 문서만 반환 | PDF의 A-Z 잡화 사전 부록(26~33p)이 "배출/종량제" 단어를 과반복해 임베딩 유사도를 왜곡 → 해당 페이지를 ingest 대상에서 제외해 해결 |
| 5 | 로컬 실행 시 VLM이 항상 "HF_TOKEN 없음"으로 실패 | VLM이 HF API 전용으로만 구현돼 있어 로컬 데모에서 항상 CNN 폴백만 탐 → `LLM_BACKEND` 스위치로 로컬은 Ollama 비전 모델(`gemma4:e2b`) 사용하도록 이중화 |

---

## ✅ 진행 현황

- [x] 프로젝트 세팅 (venv, requirements.txt, .gitignore, packages.txt)
- [x] 데이터셋 구축 (`scripts/download_dataset.py` → 5클래스 2,527장, trash 포함)
- [x] RAG 문서 수집 (`rag/docs/` 환경부 가이드라인 PDF + 생활법령정보 md)
- [x] 전체 코드 스캐폴딩 (YOLO 게이트 · CNN 추론 · RAG 체인 · Streamlit 앱)
- [x] Colab CNN 5클래스 전이학습 완료 (검증 정확도 93.66%) → `best_model.pt`·`reports/` 그래프 확보
- [x] 데이터 증강 강화 후 재학습 (RandomResizedCrop·ColorJitter·RandomErasing 등, 20 epoch) — 검증 정확도는 동일, plastic↔can 실사진 오분류는 VLM으로 보완하기로 결정
- [x] 쓰레기통 투입 UI + SVG 분리수거함 5종 애니메이션 구현
- [x] 챗봇을 모달 대신 업로드 영역 우측 상시 패널로 배치, 새 이미지 업로드 시 대화 자동 초기화
- [x] VLM을 재질+이물질 최종 판단자로 승격 (다중 모델 폴백 + CNN 폴백 + 디버그 노출)
- [x] Streamlit Cloud 배포 트러블슈팅 3건 해결 (`reports/TROUBLESHOOTING.md` 참고)
- [x] 챗봇 안전장치 (`safe_ask`) — LLM 호출 실패 시 트레이스백 대신 안내 메시지
- [x] RAG 벡터DB 구축 및 검색 품질 최종 확인 (`rag/ingest.py`) — PDF 잡화표 페이지 제외로 플라스틱/캔 검색 품질 문제 해결
- [x] VLM을 로컬 Ollama로도 사용 가능하도록 이중화 (`model/vision.py`, `LLM_BACKEND` 스위치)
- [x] Ollama 로컬 데모 통합 테스트 (게이트·판정·챗봇 전체 시나리오)
- [x] HF 계정 Inference Provider 추가 활성화 또는 크레딧 확보 후 VLM 최종 판정 재검증
- [ ] 발표 자료·데모 시나리오 정리

세부 계획과 리스크는 [`docs/PROJECT.md`](docs/PROJECT.md) 참고.

---

## 📚 데이터 출처

- **TrashNet**: Gary Thung & Mindy Yang, Stanford CS229
  (Hugging Face: https://huggingface.co/datasets/garythung/trashnet · GitHub: https://github.com/garythung/trashnet)
- **환경부 「재활용품 분리배출 가이드라인」** (구로구청 게시본 PDF)
- **찾기쉬운 생활법령정보 — 자원재활용** (https://easylaw.go.kr, 기준일 2026-06-15)
