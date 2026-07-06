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

## 6. 진행 현황 (2026-07-06 기준 — 프로젝트 완료)

### 완료
- [x] 프로젝트 세팅: venv, `requirements.txt`, `README.md`, `.gitignore`, `packages.txt`
- [x] 데이터셋 구축: `scripts/download_dataset.py` → `data/trashnet/` **5클래스** 2,527장 (trash 137장 포함)
- [x] RAG 문서 수집: `rag/docs/` 환경부 가이드라인 PDF + 생활법령정보 md
- [x] Colab CNN 5클래스 전이학습 완료 (검증 정확도 93.66%) — `model/best_model.pt`, `reports/*.png` 확보
- [x] 전체 코드 스캐폴딩 + UI 고도화:
  - `model/train_colab.ipynb` — EfficientNet-B0 전이학습(클래스 가중치·체크포인트·평가·그래프·한글폰트 포함)
  - `model/detect.py` — YOLO 게이트(0개 감지 시 전체 이미지 폴백, 2개+는 최대 박스 자동 선택)
  - `model/predict.py` — CNN CPU 추론(`best_model.pt` 로드, 5클래스 라벨)
  - `model/vision.py` — VLM 재질+이물질 최종 판정, `LLM_BACKEND` 이중화(로컬 Ollama `gemma4:e2b` / 배포 HF 다중 모델 폴백)
  - `rag/ingest.py` — docs → Chroma 벡터DB(경량 다국어 MiniLM 임베딩, PDF 잡화표 페이지 제외)
  - `rag/chain.py` — RAG 체인(k=8) + `get_llm()` 이중화(`LLM_BACKEND`=ollama/hf)
  - `app.py` — 쓰레기통 투입 UI + SVG 분리수거함 5종 애니메이션 + 좌측 사이드바 nav 3탭(데모/성과/구조) + `safe_ask` 에러 핸들링
- [x] Streamlit Cloud 배포 완료(https://trash-check.streamlit.app/) + 트러블슈팅 5건 해결 (`reports/TROUBLESHOOTING.md` 참고)
- [x] HF Inference Provider 재검증 및 Streamlit Cloud Secrets(`LLM_BACKEND=hf`, `HF_TOKEN`) 등록 완료
- [x] RAG 벡터DB 검색 품질 확인 및 개선 (PDF 잡화 부록 인덱싱 제외로 플라스틱/캔 검색 정상화)
- [x] Ollama 로컬 통합 테스트 완료 (게이트·CNN·VLM·챗봇 전체 흐름)
- [x] Streamlit 디자인 개편 — 크림+세이지그린 자연 친화 테마, 좌측 nav, 챗봇 아이콘 방식 전환
- [x] AI개발 수행내역서 작성(`reports/수행내역서.md`) + 발표자료 PDF 작성
- [x] GitHub Release `v1.0.0` 게시 (발표자료 PDF·수행내역서 첨부)
- [x] README에 Release·Colab 노트북(Drive)·바로가기 섹션 추가, GitHub About 설명 등록
- [x] GitHub Pages 포트폴리오(`hyundove.github.io`)에 본 프로젝트 카드 추가

## 7. 남은 확인/리스크

- [x] LLM 모델 선정 → Qwen2.5 7B 확정 (로컬 `ollama pull qwen2.5:7b` ~4.7GB)
- [x] Streamlit Cloud 배포 시 Ollama 불가 → HF Inference API 이중화로 해소, Secrets 등록 완료
- [x] TrashNet 클래스 매핑 확정 → 5클래스(can/glass/paper/plastic/trash), `data/trashnet/` 구축 완료 (paper 997·glass 501·plastic 482·can 410·trash 137, 총 2,527장)
- [x] 클래스 불균형 → Colab 학습 시 클래스 가중치 반영 완료 (검증 정확도 93.66%)
- [x] Colab 그래프 한글 폰트 깨짐 → NanumGothic 설치로 해결
- [x] Streamlit Cloud OpenCV/apt 의존성 문제 → opencv-python-headless + libgl1로 해결
- [x] plastic↔can CNN 오분류 → VLM을 재질 최종 판단자로 승격해 아키텍처 차원에서 보완
- [x] RAG가 플라스틱/캔 질의에 무관한 문서만 반환 → PDF 잡화표 페이지 인덱싱 제외로 해결
- [x] 로컬에서 VLM이 HF_TOKEN 없이 항상 실패 → `LLM_BACKEND` 스위치로 Ollama 이중화
- [x] Streamlit Cloud에서 간헐적 healthz 연결 리셋(앱 재시작) 관찰 — 무료 티어 메모리 한도(1GB)로 YOLO+CNN+임베딩+LLM을 동시 로드하는 구조 특성상 OOM 가능성 있음, 재현 시 로그(Manage app → Logs) 추가 확인 필요
- [x] 발표 자료·데모 시나리오 최종 리허설 (README 진행 현황의 마지막 남은 항목)
