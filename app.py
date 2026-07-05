"""분리수거 판별 어시스턴트 — Streamlit 메인.

흐름: 큰 쓰레기통(업로더)에 사진 투입 → YOLO 게이트 → CNN 재질 분류
     → VLM 최종 판정(재질+이물질, 사용 불가 시 CNN 폴백)
     → 알맞은 쓰레기통으로 들어가는 애니메이션 → RAG 챗봇.
실행: streamlit run app.py
배포: Streamlit Cloud Secrets에 LLM_BACKEND=hf, HF_TOKEN 등록.
"""
import base64
import hashlib
import html
import io
import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
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

CONF_THRESHOLD = 0.70  # 확신도가 이보다 낮으면 일반쓰레기로 안내 (판별 불확실 휴리스틱)

# 쓰레기통 5종 (표시 순서 고정)
BINS = [
    {"key": "plastic", "label": "플라스틱", "emoji": "🥤", "color": "#1976D2", "dark": "#0D47A1"},
    {"key": "can", "label": "캔", "emoji": "🥫", "color": "#F57C00", "dark": "#E65100"},
    {"key": "glass", "label": "유리", "emoji": "🍾", "color": "#00897B", "dark": "#004D40"},
    {"key": "paper", "label": "종이", "emoji": "📦", "color": "#8D6E63", "dark": "#4E342E"},
    {"key": "trash", "label": "일반쓰레기", "emoji": "🗑️", "color": "#616161", "dark": "#212121"},
]
BIN_INDEX = {b["key"]: i for i, b in enumerate(BINS)}
BIN_META = {b["key"]: b for b in BINS}

TIPS = {
    "plastic": "내용물을 비우고 라벨을 제거한 뒤 압착해서 배출하세요",
    "can": "내용물을 비우고 물로 헹군 뒤 배출하세요",
    "glass": "깨지지 않게 주의하고, 보증금 병은 소매점에 반납하세요",
    "paper": "테이프·철핀을 제거하고 접어서 배출하세요",
    "trash": "재활용이 어려운 상태예요. 종량제봉투에 배출하세요",
}

# 업로드 영역 중앙에 놓을 큰 쓰레기통 일러스트 (분리수거함 카드와 같은 스타일)
_MAIN_BIN_SVG = """<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 140 170'>
  <rect x='10' y='24' width='120' height='16' rx='7' fill='#1B5E20'/>
  <rect x='54' y='10' width='32' height='16' rx='5' fill='#1B5E20'/>
  <path d='M20 46 L120 46 L112 158 Q110 166 102 166 L38 166 Q30 166 28 158 Z' fill='#2E7D32'/>
  <rect x='40' y='64' width='10' height='84' rx='5' fill='rgba(255,255,255,0.32)'/>
  <rect x='65' y='64' width='10' height='84' rx='5' fill='rgba(255,255,255,0.32)'/>
  <rect x='90' y='64' width='10' height='84' rx='5' fill='rgba(255,255,255,0.32)'/>
</svg>"""
_MAIN_BIN_B64 = base64.b64encode(_MAIN_BIN_SVG.encode()).decode()

st.set_page_config(page_title="분리수거 판별 어시스턴트", page_icon="♻️", layout="wide")

_HEADER_HTML = """
    <style>
    .eco-header {
        background: linear-gradient(135deg, #2E7D32 0%, #66BB6A 60%, #A5D6A7 100%);
        border-radius: 16px; padding: 24px 32px; color: white; margin-bottom: 8px;
    }
    .eco-header h1 { margin: 0; font-size: 1.9rem; }
    .eco-header p { margin: 6px 0 0; opacity: 0.92; }

    /* 업로더 자리에 큰 쓰레기통 일러스트만 표시 (설명 문구·테두리 없음) */
    [data-testid="stFileUploaderDropzone"] {
        background: transparent;
        border: none !important;
        min-height: 260px;
        width: 100%;
        display: flex !important;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
    }
    [data-testid="stFileUploaderDropzone"]::before {
        content: "";
        display: block;
        width: 200px;
        height: 240px;
        background-image: url("data:image/svg+xml;base64,__MAIN_BIN_B64__");
        background-repeat: no-repeat;
        background-position: center;
        background-size: contain;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] { display: none; }

    /* 카카오톡 느낌 채팅창 (업로드 영역 우측에 상시 노출) */
    .chat-container {
        max-height: 420px;
        overflow-y: auto;
        background: #b2c7da;
        border-radius: 12px;
        padding: 16px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .chat-row { display: flex; }
    .chat-row.user { justify-content: flex-end; }
    .chat-row.assistant { justify-content: flex-start; }
    .chat-bubble {
        max-width: 75%;
        padding: 10px 14px;
        border-radius: 16px;
        font-size: 0.92rem;
        line-height: 1.45;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    .chat-bubble.user {
        background: #FEE500; color: #3C1E1E; border-top-right-radius: 4px;
    }
    .chat-bubble.assistant {
        background: #FFFFFF; color: #222; border-top-left-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.15);
    }

    .result-card { border-radius: 14px; padding: 18px 22px; color: white; margin: 10px 0; }
    .result-card h2 { margin: 0; }
    .result-card p { margin: 6px 0 0; opacity: 0.95; }
    .judge-badge {
        display: inline-block; background: rgba(255,255,255,0.22);
        border-radius: 999px; padding: 2px 12px; font-size: 0.82rem; margin-top: 8px;
    }
    </style>
    <div class="eco-header">
        <h1>♻️ 분리수거 판별 어시스턴트</h1>
        <p>쓰레기통에 사진을 던져 넣으면 AI가 재질을 판별해 알맞은 분리수거함에 넣어드려요.</p>
    </div>
    """
_HEADER_HTML = _HEADER_HTML.replace("__MAIN_BIN_B64__", _MAIN_BIN_B64)
st.markdown(_HEADER_HTML, unsafe_allow_html=True)

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


def svg_bin(meta: dict, idx: int) -> str:
    """분리수거함 SVG 일러스트 (뚜껑·몸통·투입구·라벨)."""
    c, d = meta["color"], meta["dark"]
    return f"""
    <div class="bin" id="bin-{idx}">
      <svg viewBox="0 0 110 130" width="96" height="112">
        <g class="lid" id="lid-{idx}">
          <rect x="8" y="20" width="94" height="14" rx="6" fill="{d}"/>
          <rect x="42" y="10" width="26" height="12" rx="4" fill="{d}"/>
        </g>
        <path d="M16 38 L94 38 L88 122 Q87 128 81 128 L29 128 Q23 128 22 122 Z" fill="{c}"/>
        <rect x="30" y="52" width="8" height="58" rx="4" fill="rgba(255,255,255,0.35)"/>
        <rect x="51" y="52" width="8" height="58" rx="4" fill="rgba(255,255,255,0.35)"/>
        <rect x="72" y="52" width="8" height="58" rx="4" fill="rgba(255,255,255,0.35)"/>
        <text x="55" y="49" text-anchor="middle" font-size="15">{meta['emoji']}</text>
      </svg>
      <div class="bin-label" style="color:{d}">{meta['label']}</div>
    </div>
    """


def static_bins_html() -> str:
    """업로드 전 기본 화면에 보여줄 정적인 분리수거함 5종 (애니메이션 없음)."""
    bins_html = "".join(svg_bin(b, i) for i, b in enumerate(BINS))
    return f"""
    <style>
      .stage {{ position: relative; width: 100%; height: 160px; font-family: sans-serif; }}
      .bins {{ position: absolute; bottom: 0; width: 100%;
               display: flex; justify-content: space-around; align-items: flex-end; }}
      .bin {{ text-align: center; }}
      .bin-label {{ font-size: 13px; font-weight: 700; margin-top: 2px; }}
    </style>
    <div class="stage"><div class="bins">{bins_html}</div></div>
    """


def bin_animation_html(item_b64: str, target_key: str, recycled: bool) -> str:
    """crop 이미지가 목표 쓰레기통으로 날아가 들어가는 애니메이션 HTML."""
    idx = BIN_INDEX[target_key]
    n = len(BINS)
    target_pct = (idx + 0.5) / n * 100  # 목표 통의 가로 중심(%)
    bins_html = "".join(svg_bin(b, i) for i, b in enumerate(BINS))
    effect = (
        f'<div class="particles" style="left:{target_pct}%">♻️ ♻️ ♻️</div>'
        if recycled
        else f'<div class="particles sad" style="left:{target_pct}%">💨</div>'
    )
    shake = "" if recycled else f"#bin-{idx} svg {{ animation: shake 0.5s ease 1.5s 2; }}"
    return f"""
    <style>
      .stage {{ position: relative; width: 100%; height: 300px; font-family: sans-serif; }}
      .bins {{ position: absolute; bottom: 0; width: 100%;
               display: flex; justify-content: space-around; align-items: flex-end; }}
      .bin {{ text-align: center; }}
      .bin-label {{ font-size: 13px; font-weight: 700; margin-top: 2px; }}
      .item {{
        position: absolute; top: 0; left: 50%; width: 74px; height: 74px;
        margin-left: -37px; border-radius: 12px; object-fit: cover;
        border: 3px solid white; box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        animation: fly 1.6s cubic-bezier(0.45, 0, 0.6, 1) forwards;
        z-index: 5;
      }}
      #lid-{idx} {{ transform-origin: 15px 27px; animation: lid 1.6s ease forwards; }}
      #bin-{idx} svg {{ animation: bounce 0.45s ease 1.55s 1; }}
      {shake}
      .particles {{
        position: absolute; bottom: 120px; transform: translateX(-50%);
        font-size: 22px; opacity: 0; animation: pop 1.2s ease 1.6s forwards;
      }}
      .particles.sad {{ font-size: 26px; }}
      @keyframes fly {{
        0%   {{ top: -6px; left: 50%; transform: scale(1) rotate(0deg); opacity: 1; }}
        55%  {{ top: 40px; left: {target_pct}%; transform: scale(0.9) rotate(14deg); opacity: 1; }}
        100% {{ top: 168px; left: {target_pct}%; transform: scale(0.28) rotate(32deg); opacity: 0; }}
      }}
      @keyframes lid {{
        0%, 35% {{ transform: rotate(0deg); }}
        55%     {{ transform: rotate(-24deg); }}
        80%     {{ transform: rotate(-24deg); }}
        100%    {{ transform: rotate(0deg); }}
      }}
      @keyframes bounce {{
        0% {{ transform: scale(1); }} 40% {{ transform: scale(1.08, 0.92); }} 100% {{ transform: scale(1); }}
      }}
      @keyframes shake {{
        0%, 100% {{ transform: translateX(0); }}
        25% {{ transform: translateX(-4px); }} 75% {{ transform: translateX(4px); }}
      }}
      @keyframes pop {{
        0% {{ opacity: 0; transform: translateX(-50%) translateY(10px) scale(0.6); }}
        50% {{ opacity: 1; transform: translateX(-50%) translateY(-16px) scale(1.15); }}
        100% {{ opacity: 0; transform: translateX(-50%) translateY(-34px) scale(1); }}
      }}
    </style>
    <div class="stage">
      <img class="item" src="data:image/jpeg;base64,{item_b64}"/>
      {effect}
      <div class="bins">{bins_html}</div>
    </div>
    """


def judge(cnn_label: str, cnn_conf: float, vlm_result: dict | None):
    """최종 목적지와 사유를 결정한다.

    VLM(Qwen2.5-VL)이 사용 가능하면 재질·이물질 판단을 VLM 결과로 최종 확정한다
    (CNN은 TrashNet 스튜디오 사진으로 학습돼 실제 사진에서 재질을 혼동하는
    경우가 있어, 일반 비전-언어 모델의 판단을 더 신뢰한다). VLM을 쓸 수 없을
    때만 CNN 분류 + 확신도 임계값으로 폴백한다.
    """
    if vlm_result:
        if vlm_result["contaminated"]:
            return "trash", "이물질이 감지됐어요 — 깨끗이 씻으면 재활용할 수 있어요! (VLM 판정)"
        return vlm_result["material"], "재질 판별 완료! (VLM 판정)"

    if cnn_label == "trash":
        return "trash", "CNN이 일반쓰레기로 분류했어요"
    if cnn_conf < CONF_THRESHOLD:
        return "trash", f"확신도가 낮아({cnn_conf:.0%}) 일반쓰레기로 안내해요. 다른 각도로 다시 찍어보세요"
    return cnn_label, "재질 판별 완료! (CNN 판정)"


def safe_ask(ask, material: str, question: str) -> str:
    """LLM 호출 실패(연결 오류·API 장애 등) 시 트레이스백 대신 안내 메시지 반환."""
    try:
        return ask(material, question)
    except Exception as e:
        return (
            "⚠️ 챗봇 응답을 가져오지 못했어요. LLM 설정을 확인해주세요 "
            "(Secrets의 LLM_BACKEND/HF_TOKEN, 또는 로컬 Ollama 실행 여부).\n\n"
            f"오류 내용: `{e}`"
        )


def render_chat_bubbles(messages: list[dict]) -> str:
    """카카오톡 스타일 말풍선 HTML — 대화가 길어지면 컨테이너 내부에 Y축 스크롤 생성."""
    rows = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "assistant"
        text = html.escape(msg["content"]).replace("\n", "<br>")
        rows.append(f'<div class="chat-row {role}"><div class="chat-bubble {role}">{text}</div></div>')
    return '<div class="chat-container">' + "".join(rows) + "</div>"


def render_chat_panel():
    """업로드 영역 우측에 상시 노출되는 카카오톡 스타일 챗봇 패널."""
    material = st.session_state.material
    if not material:
        st.info("👈 왼쪽에서 사진을 업로드하면 재질 판별 후 챗봇이 활성화돼요.")
        return

    st.subheader(f"💬 {material} 분리배출 문의")

    if not st.session_state.messages:
        ask = load_rag()
        with st.spinner("기본 배출 방법을 안내하는 중..."):
            intro = safe_ask(ask, material, "이 재질의 기본 분리배출 방법을 알려줘")
        st.session_state.messages.append({"role": "assistant", "content": intro})

    question = st.chat_input("배출 방법을 물어보세요 (예: 라벨은 어떻게 해요?)")
    if question:
        ask = load_rag()
        st.session_state.messages.append({"role": "user", "content": question})
        with st.spinner("답변 생성 중..."):
            answer = safe_ask(ask, material, question)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    st.markdown(render_chat_bubbles(st.session_state.messages), unsafe_allow_html=True)


# ── 탭 1: 판별 데모 ──────────────────────────────────────────
with tab_demo:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "material" not in st.session_state:
        st.session_state.material = None

    col_left, col_right = st.columns([3, 2])

    with col_left:
        uploaded = st.file_uploader(
            "쓰레기 사진 업로드", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        show_static_bins = True  # 업로드 전(또는 판정 불가 시)에는 정적인 쓰레기통 5종을 그대로 보여줌

        if uploaded:
            upload_id = hashlib.md5(uploaded.getvalue()).hexdigest()
            if st.session_state.get("last_upload_id") != upload_id:
                st.session_state.last_upload_id = upload_id
                st.session_state.messages = []  # 새 사진 업로드 시 이전 대화 초기화
                st.session_state.material = None

            image = Image.open(uploaded).convert("RGB")
            detect_single_object, classify, LABELS_KO = load_pipeline()

            with st.spinner("물체를 확인하는 중..."):
                gate = detect_single_object(image)

            with st.spinner("재질을 분류하는 중..."):
                label, conf = classify(gate.crop)

            from model.vision import vlm_judge

            with st.spinner("AI가 최종 재질·이물질을 판단하는 중..."):
                vlm_result = vlm_judge(gate.crop)

            dest_key, reason = judge(label, conf, vlm_result)
            dest = BIN_META[dest_key]
            material_ko = LABELS_KO.get(dest_key, dest_key)

            # 투입 애니메이션 (정적 쓰레기통 대신 표시)
            buf = io.BytesIO()
            gate.crop.save(buf, format="JPEG", quality=88)
            item_b64 = base64.b64encode(buf.getvalue()).decode()
            components.html(
                bin_animation_html(item_b64, dest_key, recycled=dest_key != "trash"),
                height=310,
            )
            show_static_bins = False

            badge = []
            if vlm_result:
                if not vlm_result["contaminated"]:
                    badge.append(f"재질: {material_ko} (VLM 판정)")
                else:
                    badge.append("이물질 감지됨 (VLM 판정)")
                badge.append(f"CNN 추정: {LABELS_KO.get(label, label)} · {conf:.0%}")
            elif label != "trash":
                badge.append(f"재질: {material_ko} · 확신도 {conf:.0%} (CNN, VLM 미사용)")
            st.markdown(
                f"""
                <div class="result-card" style="background:{dest['color']}">
                    <h2>{dest['emoji']} {dest['label']} 통으로!</h2>
                    <p>{reason} · {TIPS[dest_key]}</p>
                    <span class="judge-badge">{' · '.join(badge) if badge else 'CNN 분류 결과'}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.session_state.material = dest["label"] if dest_key == "trash" else material_ko

        if show_static_bins:
            components.html(static_bins_html(), height=170)

    with col_right:
        render_chat_panel()

# ── 탭 2: 학습 성과 ──────────────────────────────────────────
with tab_metrics:
    st.subheader("📊 CNN 전이학습 성과 (EfficientNet-B0 · Colab T4)")

    st.markdown("**데이터셋 — TrashNet 5클래스 재구성 (총 2,527장)**")
    st.table(
        {
            "클래스": ["paper (종이)", "glass (유리)", "plastic (플라스틱)", "can (캔)", "trash (일반쓰레기)"],
            "장수": [997, 501, 482, 410, 137],
            "원본 TrashNet 클래스": ["paper + cardboard", "glass", "plastic", "metal", "trash"],
        }
    )
    st.caption("클래스 불균형(paper ≈ trash의 7배) → CrossEntropyLoss 클래스 가중치로 보정")

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
        """📷 사진 업로드 (쓰레기통 투입)
     ↓
🔍 YOLO 게이트 (YOLOv8n · COCO 사전학습 그대로, 파인튜닝 없음)
   물체 개수 판정 → 0개/2개+ 는 재업로드 안내, 1개면 박스 crop
     ↓
🧠 CNN 분류 (EfficientNet-B0 전이학습 · TrashNet 5클래스)
   plastic / can / glass / paper / trash
     ↓
👁️ VLM 최종 판정 (Qwen2.5-VL · HF API) — 재질+이물질 함께 판단
   사용 가능하면 VLM 결과를 최종으로 확정 (CNN보다 실사진 일반화 우수)
   VLM 불가 시에만 CNN 분류 + 확신도 임계값(70%)으로 폴백
     ↓
📚 LangChain RAG (Chroma 벡터DB ← 환경부 가이드라인 + 생활법령정보)
   분류 결과 + 질문 → 근거 문서 검색 → 답변 생성
     ↓
🌐 Streamlit UI (쓰레기통 투입 애니메이션 + 챗봇)""",
        language=None,
    )

    st.subheader("설계 결정")
    st.table(
        {
            "결정": [
                "YOLO 파인튜닝 없음",
                "CNN 입력 = YOLO crop",
                "단일 객체 정책",
                "VLM을 최종 판단자로 승격",
                "LLM 이중화 (Qwen2.5 7B)",
            ],
            "이유": [
                "박스 라벨링 데이터 불필요 — COCO 사전학습만으로 개수 판정 충분",
                "배경 제거로 분류 정확도 향상, 탐지·분류 역할 분리",
                "여러 개 감지 시 재업로드 요청 — 한 장에 하나만 정확히 판별",
                "CNN은 스튜디오 사진 학습이라 실사진에서 재질 혼동 발생 — 사용 가능 시 VLM 결과로 확정, 불가 시 CNN+임계값 폴백",
                "로컬 데모 = Ollama, 배포 = HF Inference API (LLM_BACKEND 스위칭)",
            ],
        }
    )

    backend = os.getenv("LLM_BACKEND", "ollama")
    st.caption(f"현재 LLM 백엔드: **{backend}** · 데이터 출처: TrashNet(Stanford CS229) · 환경부 분리배출 가이드라인 · 생활법령정보")
