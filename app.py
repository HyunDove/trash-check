"""분리수거 판별 어시스턴트 — Streamlit 메인.

흐름: 이미지 업로드 → YOLO 게이트(개수 판정) → CNN 재질 분류 → RAG 챗봇 안내.
실행: streamlit run app.py
"""
import streamlit as st
from PIL import Image

from model.detect import detect_single_object
from model.predict import LABELS_KO, classify
from rag.chain import ask

st.set_page_config(page_title="분리수거 판별 어시스턴트", page_icon="♻️")
st.title("♻️ 분리수거 판별 어시스턴트")
st.caption("쓰레기 사진을 올리면 재질을 분류하고 올바른 분리배출 방법을 안내합니다.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "material" not in st.session_state:
    st.session_state.material = None

uploaded = st.file_uploader("쓰레기 사진 업로드 (하나만 나오게 찍어주세요)", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, width=350)

    with st.spinner("물체를 확인하는 중..."):
        gate = detect_single_object(image)

    if gate.status == "none":
        st.warning("쓰레기를 찾지 못했어요. 물체가 잘 보이게 다시 촬영해주세요.")
        st.session_state.material = None
    elif gate.status == "multiple":
        st.warning(f"쓰레기가 {gate.count}개 감지됐어요. 하나만 나오게 다시 업로드해주세요.")
        st.session_state.material = None
    else:
        with st.spinner("재질을 분류하는 중..."):
            label, conf = classify(gate.crop)
        material = LABELS_KO.get(label, label)
        st.session_state.material = material
        st.success(f"**{material}** (으)로 분류했어요. (확신도 {conf:.0%})")
        st.image(gate.crop, caption="분류에 사용된 영역", width=250)

if st.session_state.material:
    st.divider()
    st.subheader(f"💬 {st.session_state.material} 분리배출 문의")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("배출 방법을 물어보세요 (예: 라벨은 어떻게 해요?)"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                answer = ask(st.session_state.material, question)
            st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            with st.spinner("기본 배출 방법을 안내하는 중..."):
                intro = ask(st.session_state.material, "이 재질의 기본 분리배출 방법을 알려줘")
            st.write(intro)
        st.session_state.messages.append({"role": "assistant", "content": intro})
