"""RAG 인덱싱 — rag/docs의 분리배출 가이드 문서를 Chroma 벡터DB로 구축.

실행(1회성): .venv\\Scripts\\python.exe rag\\ingest.py
문서 갱신 시 재실행하면 기존 DB를 덮어쓴다.
"""
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

RAG_DIR = Path(__file__).resolve().parent
DOCS_DIR = RAG_DIR / "docs"
DB_DIR = RAG_DIR / "chroma_db"

# 한국어 지원 경량 임베딩 (~470MB) — Streamlit Cloud 메모리 제한 고려 (bge-m3는 ~2GB로 과중)
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# 환경부 가이드라인 PDF의 26~33페이지(0-index)는 "품목 A-Z 잡화 사전" 부록으로,
# "배출/종량제" 같은 단어를 반복 나열해 임베딩 유사도를 왜곡시켜 재질별 본문 검색을 밀어낸다 (검색 품질 저하 확인됨).
EXCLUDED_PDF_PAGES = set(range(26, 34))


def load_documents():
    docs = []
    for path in sorted(DOCS_DIR.iterdir()):
        if path.suffix == ".pdf":
            pdf_docs = PyPDFLoader(str(path)).load()
            docs.extend(d for d in pdf_docs if d.metadata.get("page") not in EXCLUDED_PDF_PAGES)
        elif path.suffix == ".md":
            docs.extend(TextLoader(str(path), encoding="utf-8").load())
    return docs


def main():
    docs = load_documents()
    print(f"문서 {len(docs)}개 로드")

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"청크 {len(chunks)}개 생성")

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
    print(f"벡터DB 구축 완료: {DB_DIR}")


if __name__ == "__main__":
    main()
