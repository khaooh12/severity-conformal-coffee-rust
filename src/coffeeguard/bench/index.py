"""Dựng bảng chỉ mục ảnh RoCoLe: nhãn + nhóm + khung bao lá.

Đây là **mẫu số** của mọi thứ phía sau. Nó nối ba nguồn độc lập:

    rocole_labels.csv   nhãn 6 lớp, plant_id / leaf_id / face (từ .xlsx)
    RoCoLE-coco.json    đa giác viền lá, đúng 1 đa giác / ảnh
    Photos/*.jpg        file ảnh thật

Nếu ba nguồn lệch nhau thì hàm ở đây **ném lỗi**, không tự bỏ qua. Lặng lẽ bỏ
ảnh thiếu mặt nạ là cách chắc chắn nhất để bảng phân bố lớp trong báo cáo khác
với dữ liệu thực dùng, mà không có gì báo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import paths as P

#: COCO của RoCoLe ghi `file_name` là URL Labelbox, tên thật nằm sau dấu '-' cuối.
#: vd: ".../cjs5...%2Fbea93765-...-C5P27E2.jpg" -> "C5P27E2.jpg"
_CAT_LA = 1  # categories: 1=Leaf, 2=classification (không annotation nào dùng)


def ten_file_tu_url(url: str) -> str:
    return url.rsplit("-", 1)[-1]


@dataclass(frozen=True)
class KetQuaIndex:
    df: pd.DataFrame
    bo_qua: dict[str, list[str]]


def _doc_mat_na(coco_path: Path) -> dict[str, dict]:
    """file ảnh -> {w, h, leaf_x, leaf_y, leaf_w, leaf_h, leaf_area_px}."""
    doc = json.loads(coco_path.read_text(encoding="utf-8"))
    kich_thuoc = {
        img["id"]: (ten_file_tu_url(img["file_name"]), img["width"], img["height"])
        for img in doc["images"]
    }

    ra: dict[str, dict] = {}
    trung: list[str] = []
    for ann in doc["annotations"]:
        if ann.get("category_id") != _CAT_LA:
            continue
        ten, w, h = kich_thuoc[ann["image_id"]]
        x, y, bw, bh = ann["bbox"]
        if ten in ra:
            # RoCoLe có đúng 1 đa giác/ảnh. Nhiều hơn = giả định của ta sai,
            # và "lấy cái đầu tiên" sẽ cắt nhầm lá ở một số ảnh.
            trung.append(ten)
            continue
        seg = ann.get("segmentation") or []
        # Giữ luôn đa giác viền lá: cần nó để đo CHÍNH XÁC phần lá nằm trong cửa
        # sổ đã cắt (cửa sổ neo cạnh ngắn cắt mất hai đầu lá, nên `area` của cả
        # lá không còn là tử số đúng), và sau này để thay nền ngẫu nhiên.
        ra[ten] = {
            "img_w": int(w),
            "img_h": int(h),
            "leaf_x": float(x),
            "leaf_y": float(y),
            "leaf_w": float(bw),
            "leaf_h": float(bh),
            "leaf_area_px": float(ann.get("area") or 0.0),
            "leaf_poly": json.dumps(
                [round(float(v), 1) for v in seg[0]] if seg else [], separators=(",", ":")
            ),
        }
    if trung:
        raise ValueError(
            f"{len(trung)} anh co NHIEU hon mot da giac la (vd {trung[:5]}). "
            "Gia dinh '1 da giac/anh' khong con dung -> xem lai buoc cat."
        )
    return ra


def dung_index(
    labels_csv: Path = P.ROCOLE_LABELS,
    coco_json: Path = P.ROCOLE_COCO,
    photos: Path = P.ROCOLE_PHOTOS,
) -> KetQuaIndex:
    for p in (labels_csv, coco_json, photos):
        if not p.exists():
            raise FileNotFoundError(f"Thieu nguon du lieu: {p}")

    lab = pd.read_csv(labels_csv)
    can = {"file", "multiclass_label", "plant_id", "leaf_id", "face"}
    if not can.issubset(lab.columns):
        raise ValueError(f"{labels_csv} thieu cot: {sorted(can - set(lab.columns))}")

    la = pd.Series(lab["multiclass_label"]).unique()
    la_dz = sorted(set(la) - set(ct.CLASSES_6))
    if la_dz:
        raise ValueError(
            f"Nhan la trong {labels_csv}: {la_dz}. Hop dong chi co {list(ct.CLASSES_6)}"
        )

    mat_na = _doc_mat_na(coco_json)
    tren_dia = {p.name for p in photos.glob("*.jpg")}

    bo_qua = {
        "thieu_anh": sorted(set(lab.file) - tren_dia),
        "thieu_mat_na": sorted(set(lab.file) - set(mat_na)),
        "co_anh_khong_nhan": sorted(tren_dia - set(lab.file)),
    }
    if bo_qua["thieu_anh"] or bo_qua["thieu_mat_na"]:
        raise ValueError(
            "Ba nguon khong khop nhau — KHONG tu bo qua:\n"
            f"  co nhan nhung thieu file anh : {len(bo_qua['thieu_anh'])} {bo_qua['thieu_anh'][:5]}\n"
            f"  co nhan nhung thieu mat na   : {len(bo_qua['thieu_mat_na'])} {bo_qua['thieu_mat_na'][:5]}"
        )

    df = lab.rename(columns={"multiclass_label": "label6"}).copy()
    df["dataset"] = "rocole"
    df["png"] = [Path(f).stem + ".png" for f in df.file]
    df["band"] = [ct.band_of(c) for c in df.label6]
    df["binary"] = [ct.BINARY_OF[c] for c in df.label6]

    mn = pd.DataFrame.from_dict(mat_na, orient="index")
    mn.index.name = "file"
    df = df.merge(mn.reset_index(), on="file", how="left", validate="one_to_one")

    # Hai con số nền để so với cửa sổ vuông. Chúng KHÁC nhau và hay bị lẫn:
    #   leaf_frac_khung     = lá / ảnh gốc 16:9
    #   leaf_frac_letterbox = lá / ô vuông sau khi chèn viền  <- cái model thực nhận
    # Ảnh 16:9 nhồi vào ô vuông sinh thêm ~44% diện tích toàn viền đen, nên con
    # số thứ hai thấp hơn hẳn con số thứ nhất.
    df["leaf_frac_khung"] = df.leaf_area_px / (df.img_w * df.img_h)
    df["leaf_frac_letterbox"] = df.leaf_area_px / (
        df[["img_w", "img_h"]].max(axis=1) ** 2
    )
    df["bbox_ti_le"] = df[["leaf_w", "leaf_h"]].max(axis=1) / df[
        ["leaf_w", "leaf_h"]
    ].min(axis=1)

    cot = [
        "png", "file", "dataset", "label6", "band", "binary",
        "plant_id", "leaf_id", "face",
        "img_w", "img_h", "leaf_x", "leaf_y", "leaf_w", "leaf_h",
        "leaf_area_px", "leaf_frac_khung", "leaf_frac_letterbox", "bbox_ti_le",
        "leaf_poly",
    ]
    df = df[cot].sort_values("png").reset_index(drop=True)

    if df.png.duplicated().any():
        raise ValueError("Ten PNG bi trung sau khi doi duoi file — se ghi de nhau")
    return KetQuaIndex(df=df, bo_qua=bo_qua)


def ghi_index(kq: KetQuaIndex, dich: Path = P.INDEX_CSV) -> Path:
    dich.parent.mkdir(parents=True, exist_ok=True)
    kq.df.to_csv(dich, index=False, encoding="utf-8")
    return dich


def doc_index(nguon: Path = P.INDEX_CSV) -> pd.DataFrame:
    if not nguon.exists():
        raise FileNotFoundError(
            f"Chua co {nguon}. Chay:  python scripts/bench.py index"
        )
    return pd.read_csv(nguon)
