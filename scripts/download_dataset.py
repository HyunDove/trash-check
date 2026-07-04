"""TrashNet 데이터셋 다운로드 → 4클래스 폴더 구조로 저장.

garythung/trashnet은 datasets 자동 로더와 호환되지 않아(저장소 구조 문제)
dataset-resized.zip을 직접 받아 압축 해제 후 재구성한다.

출처: Hugging Face https://huggingface.co/datasets/garythung/trashnet
원저작: Gary Thung & Mindy Yang, Stanford CS229 (https://github.com/garythung/trashnet)

TrashNet 6클래스 → 과제 4클래스 매핑:
  plastic → plastic(플라스틱) / metal → can(캔) / glass → glass(유리)
  paper, cardboard → paper(종이) / trash → 제외
출력: data/trashnet/{plastic,can,glass,paper}/*.jpg
실행: .venv\\Scripts\\python.exe scripts\\download_dataset.py
"""
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

CLASS_MAP = {
    "plastic": "plastic",
    "metal": "can",
    "glass": "glass",
    "paper": "paper",
    "cardboard": "paper",
    "trash": "trash",  # 일반쓰레기 — 5클래스 학습용
}

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "trashnet"
TMP_DIR = ROOT / "data" / "_extract"


def main():
    zip_path = hf_hub_download(
        repo_id="garythung/trashnet",
        filename="dataset-resized.zip",
        repo_type="dataset",
    )
    print(f"다운로드 완료: {zip_path}")

    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(TMP_DIR)

    counts = {}
    for img in TMP_DIR.rglob("*.jpg"):
        if img.name.startswith("._"):
            continue  # macOS 리소스 포크 메타데이터 파일(AppleDouble) 제외
        src_label = img.parent.name
        dst_label = CLASS_MAP.get(src_label)
        if dst_label is None:
            continue
        out = OUT_DIR / dst_label
        out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img, out / img.name)
        counts[dst_label] = counts.get(dst_label, 0) + 1

    shutil.rmtree(TMP_DIR)
    print("저장 완료:", counts)


if __name__ == "__main__":
    main()
