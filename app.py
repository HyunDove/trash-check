"""분리수거 판별 어시스턴트 — Streamlit 메인.

흐름: 이미지 업로드 → YOLO 게이트(개수 판정) → CNN 재질 분류 → RAG 챗봇 안내.
실행: streamlit run app.py
배포: Streamlit Cloud에서는 Secrets에 LLM_BACKEND=hf, HF_TOKEN을 등록하면
      st.secrets → 환경변수로 주입되어 HF Inference API로 동작한다.
"""
import os
from pathlib import Path

import streamlit as st
from PIL import Image

# Streamlit Cloud Secrets → 환경변수 주입 (rag.chain import 전에 수행)
try:
    for key in ("LLM_BACKEND", "HF_TOKEN"):
        if key not in os.environ and key in st.secrets:
            os.environ[key] = st.secrets[key]
except FileNotFoundError:
    pass  # 로컬에 secrets.toml이 없으면 환경변수/기본값 사용

ROOT = Path(__file__).resolve().parent
REPORTS_DIR = ROOT / "reports"

MATERIAL_META = {
    "플라스틱": {"emoji": "🥤", "color": "#1976D2", "tip": "내용물을 비우고 라벨을 제거한 뒤 압착해서 배출"},
    "캔": {"emoji": "🥫", "color": "#F57C00", "tip": "내용물을 비우고 물로 헹군 뒤 배출"},
    "유리": {"emoji": "🍾", "color": "#00897B", "tip": "깨지지 않게 주의, 보증금 병은 소매점 반납"},
    "종이": {"emoji": "📦", "color": "#8D6E63", "tip": "테이프·철핀 제거 후 접어서 배출"},
}

st.set_page_config(page_title="분리수거 판별 어시스턴트", page_icon="♻️", layout="wide")

st.markdown(
    """
    <style>
    .eco-header {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 60%, #A5D6A7 100%);
        border-radius: 16px; padding: 28px 32px; color: white; margin-bottom: 8px;
    }
    .eco-header h1 { margin: 0; font-size: 2rem; }
    .eco-header p { margin: 6px 0 0; opacity: 0.92; }
    .step-card {
        background: #F1F8E9; border: 1px solid #C5E1A5; border-radius: 12px;
        padding: 14px 16px; text-align: center; height: 100%;
    }
    .step-card b { color: #2E7D32; }
    .result-card {
        border-radius: 14px; padding: 20px 24px; color: white; margin: 10px 0;
    }
    .result-card h2 { margin: 0; }
    .result-card p { margin: 6px 0 0; opacity: 0.95; }
    </style>
    <div class="eco-header">
        <h1>♻️ 분리수거 판별 어시스턴트</h1>
        <p>쓰레기 사진을 올리면 AI가 재질을 판별하고, 공식 가이드 근거로 올바른 배출 방법을 알려드려요.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 파이프라인 단계 카드
c1, c2, c3, c4 = st.columns(4)
for col, (icon, title, desc) in zip(
    (c1, c2, c3, c4),
    [
        ("📷", "1. 업로드", "쓰레기 사진 한 장"),
        ("🔍", "2. YOLO 게이트", "물체 개수 판정 · crop"),
        ("🧠", "3. CNN 분류", "플라스틱·캔·유리·종이"),
        ("💬", "4. RAG 안내", "가이드 문서 기반 답변"),
    ],
):
    col.markdown(
        f'<div class="step-card">{icon}<br><b>{title}</b><br><small>{desc}</small></div>',
        unsafe_allow_html=True,
    )

st.write("")

tab_demo, tab_metrics, tab_arch = st.tabs(["♻️ 판별 데모", "📊 학습 성과", "🧠 모델 구조"])


@st.cache_resource(show_spinner="AI 모델을 준비하는 중...")
def load_pipeline():
    from model.detect import detect_single_object
    from model.predict import LABELS_KO, classify

    return detect_single_object, classify, LABELS_KO


@st.cache_resource(show_spinner="분리배출 가이드 지식베이스를 준비하는 중...")
def load_rag():
    from rag import ingest
    from rag.chain import ask

    if not ingest.DB_DIR.exists():
        ingest.main()  # 배포 첫 부팅 시 벡터DB 자동 구축
    return ask


# ── 탭 1: 판별 데모 ──────────────────────────────────────────
with tab_demo:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "material" not in st.session_state:
        st.session_state.material = None

    left, right = st.columns([1, 1])

    with left:
        uploaded = st.file_uploader(
            "쓰레기 사진 업로드 (하나만 나오게 찍어주세요)", type=["jpg", "jpeg", "png"]
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, width=340)

            detect_single_object, classify, LABELS_KO = load_pipeline()
            with st.spinner("물체를 확인하는 중..."):
                gate = detect_single_object(image)

            if gate.status == "none":
                st.warning("🔍 쓰레기를 찾지 못했어요. 물체가 잘 보이게 다시 촬영해주세요.")
                st.session_state.material = None
            elif gate.status == "multiple":
                st.warning(f"🔍 쓰레기가 {gate.count}개 감지됐어요. **하나만** 나오게 다시 업로드해주세요.")
                st.session_state.material = None
            else:
                with st.spinner("재질을 분류하는 중..."):
                    label, conf = classify(gate.crop)
                material = LABELS_KO.get(label, label)
                st.session_state.material = material

                meta = MATERIAL_META.get(material, {"emoji": "♻️", "color": "#2E7D32", "tip": ""})
                st.markdown(
                    f"""
                    <div class="result-card" style="background:{meta['color']}">
                        <h2>{meta['emoji']} {material}</h2>
                        <p>확신도 {conf:.0%} · {meta['tip']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(gate.crop, caption="분류에 사용된 영역 (YOLO crop)", width=240)

    with right:
        if st.session_state.material:
            st.subheader(f"💬 {st.session_state.material} 분리배출 문의")

            if not st.session_state.messages:
                ask = load_rag()
                with st.chat_message("assistant"):
                    with st.spinner("기본 배출 방법을 안내하는 중..."):
                        intro = ask(st.session_state.material, "이 재질의 기본 분리배출 방법을 알려줘")
                    st.write(intro)
                st.session_state.messages.append({"role": "assistant", "content": intro})
            else:
                for msg in st.session_state.messages:
                    with st.chat_message(msg["role"]):
                        st.write(msg["content"])

            if question := st.chat_input("배출 방법을 물어보세요 (예: 라벨은 어떻게 해요?)"):
                ask = load_rag()
                st.session_state.messages.append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.write(question)
                with st.chat_message("assistant"):
                    with st.spinner("답변 생성 중..."):
                        answer = ask(st.session_state.material, question)
                    st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            st.info("👈 왼쪽에서 사진을 업로드하면 재질 판별 후 챗봇이 열립니다.")

# ── 탭 2: 학습 성과 ──────────────────────────────────────────
with tab_metrics:
    st.subheader("📊 CNN 전이학습 성과 (EfficientNet-B0 · Colab T4)")

    st.markdown("**데이터셋 — TrashNet 4클래스 재구성 (총 2,391장)**")
    st.table(
        {
            "클래스": ["paper (종이)", "glass (유리)", "plastic (플라스틱)", "can (캔)"],
            "장수": [998, 501, 482, 410],
            "원본 TrashNet 클래스": ["paper + cardboard", "glass", "plastic", "metal"],
        }
    )
    st.caption("paper 클래스 불균형(약 2배) → CrossEntropyLoss 클래스 가중치로 보정")

    graphs = [
        ("training_curve.png", "학습 곡선 (Train Loss / Val Accuracy)"),
        ("confusion_matrix.png", "혼동행렬 (Confusion Matrix)"),
        ("class_metrics.png", "클래스별 Precision / Recall / F1"),
    ]
    available = [(REPORTS_DIR / f, cap) for f, cap in graphs if (REPORTS_DIR / f).exists()]
    if available:
        cols = st.columns(len(available))
        for col, (path, cap) in zip(cols, available):
            col.image(str(path), caption=cap, use_container_width=True)
    else:
        st.info("학습 그래프(reports/*.png)가 아직 저장소에 없습니다. Colab 학습 산출물을 `reports/` 폴더에 배치하세요.")

# ── 탭 3: 모델 구조 ──────────────────────────────────────────
with tab_arch:
    st.subheader("🧠 아키텍처")
    st.code(
        """📷 사진 업로드
     ↓
🔍 YOLO 게이트 (YOLOv8n · COCO 사전학습 그대로, 파인튜닝 없음)
   물체 개수 판정 → 0개/2개+ 는 재업로드 안내, 1개면 박스 crop
     ↓
🧠 CNN 분류 (EfficientNet-B0 전이학습 · TrashNet 4클래스)
   plastic / can / glass / paper
     ↓
📚 LangChain RAG (Chroma 벡터DB ← 환경부 가이드라인 + 생활법령정보)
   분류 결과 + 질문 → 근거 문서 검색 → 답변 생성
     ↓
🌐 Streamlit UI""",
        language=None,
    )

    st.subheader("설계 결정")
    st.table(
        {
            "결정": [
                "YOLO 파인튜닝 없음",
                "CNN 입력 = YOLO crop",
                "단일 객체 정책",
                "LLM 이중화 (Qwen2.5 7B)",
            ],
            "이유": [
                "박스 라벨링 데이터 불필요 — COCO 사전학습만으로 개수 판정 충분",
                "배경 제거로 분류 정확도 향상, 탐지·분류 역할 분리",
                "여러 개 감지 시 재업로드 요청 — 한 장에 하나만 정확히 판별",
                "로컬 데모 = Ollama, 배포 = HF Inference API (LLM_BACKEND 스위칭)",
            ],
        }
    )

    backend = os.getenv("LLM_BACKEND", "ollama")
    st.caption(f"현재 LLM 백엔드: **{backend}** · 데이터 출처: TrashNet(Stanford CS229) · 환경부 분리배출 가이드라인 · 생활법령정보")
