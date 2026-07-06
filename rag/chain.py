"""LangChain RAG 체인 — 분류 결과 + 사용자 질문으로 분리배출 방법 안내.

LLM 이중화: LLM_BACKEND=ollama(기본, 로컬 데모) | hf(배포, HF Inference API).
RAG 체인 코드는 양쪽 공유, LLM 객체만 스위칭한다.
"""
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

DB_DIR = Path(__file__).resolve().parent / "chroma_db"
# ingest.py의 EMBEDDING_MODEL과 반드시 동일해야 벡터DB 호환 (경량 다국어 모델, 배포 메모리 고려)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

PROMPT = ChatPromptTemplate.from_template(
    """당신은 분리수거 안내 도우미입니다. 아래 참고 문서를 근거로 답변하세요.
문서에 없는 내용은 지어내지 말고 모른다고 답하세요. 한국어로 간결하게 답변하세요.

[참고 문서]
{context}

[감지된 쓰레기 재질]
{material}

[사용자 질문]
{question}

답변:"""
)


def get_llm():
    """LLM_BACKEND 환경변수로 로컬(Ollama)/배포(HF API) 스위칭."""
    if os.getenv("LLM_BACKEND", "ollama") == "hf":
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        return ChatHuggingFace(llm=HuggingFaceEndpoint(
            repo_id="Qwen/Qwen2.5-7B-Instruct",
            huggingfacehub_api_token=os.environ["HF_TOKEN"],
            max_new_tokens=512,
            temperature=0.3,
        ))
    from langchain_ollama import ChatOllama

    return ChatOllama(model="qwen2.5:7b", temperature=0.3)


_retriever = None
_chain = None


def _get_chain():
    global _retriever, _chain
    if _chain is None:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        db = Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)
        _retriever = db.as_retriever(search_kwargs={"k": 8})
        _chain = PROMPT | get_llm() | StrOutputParser()
    return _retriever, _chain


def ask(material_ko: str, question: str) -> str:
    """재질(한국어 라벨)과 질문을 받아 근거 문서 기반 답변을 생성한다."""
    retriever, chain = _get_chain()
    docs = retriever.invoke(f"{material_ko} 분리배출 방법 {question}")
    context = "\n\n".join(d.page_content for d in docs)
    return chain.invoke({"context": context, "material": material_ko, "question": question})
