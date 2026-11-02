"""Đường dẫn chuẩn của đồ án — một chỗ duy nhất, không hardcode rải rác.

Mọi script đều gọi `goc_du_an()`, nên chạy từ thư mục nào cũng ra cùng kết quả.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_MOC = "pyproject.toml"
#: Trên Colab không có cây repo, nên đặt biến môi trường này trỏ vào thư mục
#: đã giải nén gói (`coffeeguard_bench_*.zip`). Notebook 02 tự đặt nó.
BIEN_MOI_TRUONG = "COFFEEGUARD_ROOT"


def goc_du_an(bat_dau: Path | None = None) -> Path:
    """Thư mục gốc: biến môi trường nếu có, không thì đi ngược lên tìm `pyproject.toml`."""
    if (env := os.environ.get(BIEN_MOI_TRUONG)):
        p = Path(env).expanduser().resolve()
        if not p.is_dir():
            raise RuntimeError(f"{BIEN_MOI_TRUONG}={env} khong phai thu muc")
        return p
    p = (bat_dau or Path(__file__)).resolve()
    for cha in [p, *p.parents]:
        if (cha / _MOC).is_file():
            return cha
    raise RuntimeError(
        f"Khong tim thay {_MOC} tu {p}. Neu dang chay ngoai repo (vd Colab), "
        f"dat bien moi truong {BIEN_MOI_TRUONG}."
    )


ROOT = goc_du_an()

DATA = ROOT / "data"
RAW = DATA / "raw"
EXTERNAL = DATA / "external"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
SPLITS = DATA / "splits"          # CÓ COMMIT (xem .gitignore) — quy tắc 12
REPORTS = ROOT / "reports"
PREDICTIONS = REPORTS / "predictions"
#: Dự đoán của thí nghiệm rò rỉ (chia theo lá / theo ảnh). Thư mục RIÊNG để
#: `bench.py score` — chỉ đọc PREDICTIONS — không bao giờ trộn chúng vào bảng chính.
PREDICTIONS_RO_RI = REPORTS / "predictions_ro_ri"
#: Thí nghiệm rò rỉ theo CẤP ĐỘ model (ResNet18/50/152 × 3 cách chia, notebook
#: 02b_cap_do_*). Cả lượt theo cây cũng ghi ở đây, KHÔNG vào PREDICTIONS, để
#: `score` và bảng chính không bao giờ thấy chúng. Notebook 02b ghi thẳng chuỗi
#: "reports/predictions_cap_do" (gói Colab cũ chưa có hằng này) — đổi tên thì sửa cả hai.
PREDICTIONS_CAP_DO = REPORTS / "predictions_cap_do"
#: Dự đoán có/không TTA trên cùng trọng số (`bench/tta.py`). Thư mục RIÊNG vì
#: thường chỉ phủ một phần fold — trộn vào PREDICTIONS sẽ làm `score` báo thiếu ảnh.
PREDICTIONS_TTA = REPORTS / "predictions_tta"
MODELS = ROOT / "models"

# --- nguồn dữ liệu thô ---
ROCOLE_PHOTOS = RAW / "rocole" / "Photos"
ROCOLE_COCO = RAW / "rocole" / "Annotations" / "RoCoLE-coco.json"
ROCOLE_LABELS = EXTERNAL / "RoCoLe_annotations" / "rocole_labels.csv"
BRACOL_IMAGES = RAW / "bracol" / "images"
BRACOL_LABELS = EXTERNAL / "BRACOL_labels" / "dataset.csv"
SYMPTOM = RAW / "symptom"

# --- tạo phẩm đã đóng băng ---
INDEX_CSV = INTERIM / "rocole_index.csv"
FOLDS_JSON = SPLITS / "rocole_folds.json"


def bam_file(p: Path, chunk: int = 1 << 20) -> str:
    """sha256 của một file — dùng để buộc split vào đúng bản index sinh ra nó."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while blk := f.read(chunk):
            h.update(blk)
    return h.hexdigest()


def tao_thu_muc() -> None:
    for d in (INTERIM, PROCESSED, SPLITS, REPORTS, PREDICTIONS):
        d.mkdir(parents=True, exist_ok=True)
