"""Chia 5 fold theo CÂY — sinh một lần, đóng băng, commit vào git.

Vì sao không để mỗi notebook tự chia (quy tắc 12): bảng so sánh khi đó đang so
**cách chia** chứ không so kiến trúc, và mọi con số vẫn trông bình thường.

Vì sao chia theo `plant_id` chứ không theo ảnh: mỗi cây có 2 lá, mỗi lá 2 ảnh
(mặt trên + mặt dưới). Chia theo ảnh thì mặt còn lại của CHÍNH chiếc lá đó nằm ở
tập test — rò rỉ gần như tuyệt đối, và kết quả sẽ đẹp một cách vô nghĩa.

Vì sao 5 fold chứ không phải một lần 70/15/15: `rust_level_4` chỉ có 29 ảnh. Chia
một lần thì tập test có 3 ảnh cấp 4 — so 4 kiến trúc trên 3 ảnh là so nhiễu
thuần. Với 5 fold, mỗi ảnh vào test đúng một lần -> 29 dự đoán cho cấp 4, và
phương sai lấy từ chênh lệch giữa các fold nên không cần chạy nhiều seed.

Thí nghiệm rò rỉ (`Docs/Khoang_trong_bai_bao.md`, G1): hai cách chia CỐ Ý rò rỉ —
theo LÁ (lá kia của cùng cây có thể ở tập train) và theo ẢNH (cả mặt kia của
cùng lá). Chúng ghi ra file RIÊNG, mang cờ `thi_nghiem_ro_ri`, và:

* `ghi_split` từ chối ghi chúng đè lên `rocole_folds.json`;
* `doc_split` từ chối đọc chúng trừ khi gọi với `cho_phep_ro_ri=True`.

Không có hai chốt này, một notebook đọc nhầm file là train ra model "giỏi hơn"
hẳn, export lên app, và mọi bảng so sánh vẫn trông bình thường.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import paths as P

SCHEMA_VERSION = "2.0"

#: cách chia -> cột nhóm. Chỉ `theo_cay` là cách chia của đồ án; hai cách còn lại
#: chỉ dùng để ĐO mức phồng do rò rỉ.
CACH_CHIA: dict[str, str] = {
    "theo_cay": "plant_id",
    "theo_la": "leaf_id",
    "theo_anh": "png",
}
CACH_CHIA_RO_RI: tuple[str, ...] = ("theo_la", "theo_anh")


def duong_dan_split(cach_chia: str = "theo_cay") -> Path:
    if cach_chia not in CACH_CHIA:
        raise ValueError(f"cach_chia '{cach_chia}' khong hop le: {list(CACH_CHIA)}")
    if cach_chia == "theo_cay":
        return P.FOLDS_JSON
    return P.SPLITS / f"rocole_folds_{cach_chia}.json"


def _phan_fold(df: pd.DataFrame, n_folds: int, seed: int, group_col: str) -> dict[str, int]:
    """Gán mỗi nhóm (cây) vào đúng một fold, cân bằng theo lớp hiếm.

    Dùng StratifiedGroupKFold: chia ngẫu nhiên thuần theo cây dễ dồn hết
    `rust_level_4` vào 2-3 fold, khiến các fold còn lại không đo được gì cho lớp
    đó và độ lệch chuẩn giữa fold phồng lên vì lý do sai.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    y = df.label6.map({c: i for i, c in enumerate(ct.CLASSES_6)}).to_numpy()
    groups = df[group_col].to_numpy()
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_cua_nhom: dict[str, int] = {}
    for f, (_, idx_te) in enumerate(sgkf.split(df, y, groups)):
        for g in pd.unique(groups[idx_te]):
            fold_cua_nhom[str(g)] = f
    return fold_cua_nhom


def _kiem_chung(df: pd.DataFrame, n_folds: int, group_col: str) -> list[str]:
    """Assert những thứ phải đúng; trả về danh sách CẢNH BÁO (không chặn).

    Kiểm không rò rỉ ở mức nhóm đã chọn VÀ mọi mức nhỏ hơn: chia theo cây thì
    không lá nào, không cây nào nằm hai bên; chia theo lá thì không lá nào.
    """
    muc = {"plant_id": ("plant_id", "leaf_id", "png"),
           "leaf_id": ("leaf_id", "png"),
           "png": ("png",)}.get(group_col, (group_col,))
    for f in range(n_folds):
        tr, te = df[df.fold != f], df[df.fold == f]
        for cot in muc:
            chung = set(tr[cot]) & set(te[cot])
            assert not chung, f"RO RI fold {f}: {len(chung)} {cot} o ca hai ben"
    assert df.fold.notna().all(), "Co anh khong duoc gan fold"
    assert df.png.is_unique, "PNG trung nhau"

    canh_bao = []
    bang = pd.crosstab(df.label6, df.fold).reindex(ct.CLASSES_6).fillna(0).astype(int)
    for lop in ct.CLASSES_6:
        n = int(bang.loc[lop].sum())
        rong = [int(f) for f in bang.columns if bang.loc[lop, f] == 0]
        if rong:
            canh_bao.append(
                f"lop '{lop}' (n={n}) VANG MAT o fold {rong} -> macro-F1 cua "
                f"nhung fold do tinh tren it lop hon, khong so thang duoc"
            )
        elif n < 60:
            canh_bao.append(
                f"lop '{lop}' chi co n={n} tren toan bo -> bao cao chinh phai "
                f"dung 5 dai, 4 cap chi la phu va phai ghi kem n"
            )
    return canh_bao


def tao_split(
    df_index: pd.DataFrame,
    n_folds: int = ct.N_FOLDS,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
    index_sha256: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    import sklearn

    cach = {v: k for k, v in CACH_CHIA.items()}.get(group_col, group_col)
    ro_ri = cach in CACH_CHIA_RO_RI
    df = df_index.copy()
    fold_cua_nhom = _phan_fold(df, n_folds, seed, group_col)
    df["fold"] = df[group_col].astype(str).map(fold_cua_nhom).astype(int)

    canh_bao = _kiem_chung(df, n_folds, group_col)

    bang6 = pd.crosstab(df.label6, df.fold).reindex(ct.CLASSES_6).fillna(0).astype(int)
    bangB = pd.crosstab(df.band, df.fold).reindex(ct.CLASSES_BAND).fillna(0).astype(int)

    meta = {
        "schema_version": SCHEMA_VERSION,
        "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset": "rocole",
        "seed": seed,
        "n_folds": n_folds,
        "nhom_theo": group_col,
        "cach_chia": cach,
        "thi_nghiem_ro_ri": ro_ri,
        "thuat_toan": "sklearn.model_selection.StratifiedGroupKFold(shuffle=True)",
        "sklearn_version": sklearn.__version__,
        "index_csv": str(P.INDEX_CSV.relative_to(P.ROOT)).replace("\\", "/"),
        "index_sha256": index_sha256,
        "classes_6": list(ct.CLASSES_6),
        "classes_band": list(ct.CLASSES_BAND),
        "band_of": dict(ct.BAND_OF),
        "n_images": int(len(df)),
        "n_plants": int(df.plant_id.nunique()),
        "n_nhom": int(df[group_col].nunique()),
        "n_leaves": int(df.leaf_id.nunique()),
        "so_anh_moi_lop_theo_fold": {
            lop: [int(bang6.loc[lop, f]) for f in range(n_folds)] for lop in ct.CLASSES_6
        },
        "so_anh_moi_dai_theo_fold": {
            d: [int(bangB.loc[d, f]) for f in range(n_folds)] for d in ct.CLASSES_BAND
        },
        "canh_bao": canh_bao,
        "fold_cua_nhom": fold_cua_nhom,
        "ghi_chu": (
            "THI NGHIEM RO RI — CO Y de mat kia / la kia cua cung cay o tap train. "
            "CHI dung de do muc phong ket qua. KHONG train model cho app tu file nay."
            if ro_ri else
            "DA DONG BANG. Notebook train chi DOC file nay, khong bao gio chia lai. "
            "Thu tu classes_6 la BAT BIEN. Sinh lai file nay = moi ket qua truoc do "
            "khong con so sanh duoc voi ket qua sau do."
        ),
        "may_sinh": f"{platform.system()} {platform.release()} / py{platform.python_version()}",
    }
    return df, meta


def ghi_split(meta: dict, dich: Path = P.FOLDS_JSON) -> Path:
    if meta.get("thi_nghiem_ro_ri") and dich.resolve() == P.FOLDS_JSON.resolve():
        raise ValueError(
            f"Tu choi ghi split RO RI ({meta.get('cach_chia')}) de len {P.FOLDS_JSON.name} — "
            "do la file chia chinh cua do an. Dung duong_dan_split(cach_chia)."
        )
    dich.parent.mkdir(parents=True, exist_ok=True)
    dich.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return dich


def doc_split(nguon: Path = P.FOLDS_JSON, cho_phep_ro_ri: bool = False) -> dict:
    if not nguon.exists():
        raise FileNotFoundError(
            f"Chua co {nguon}. Chay:  python scripts/bench.py splits"
        )
    meta = json.loads(nguon.read_text(encoding="utf-8"))
    if list(meta.get("classes_6", [])) != list(ct.CLASSES_6):
        raise ValueError(
            "THU TU LOP trong file split khac contract.py.\n"
            f"  split   : {meta.get('classes_6')}\n"
            f"  contract: {list(ct.CLASSES_6)}\n"
            "Giai ma index->nhan se cho nhan SAI voi confidence VAN CAO."
        )
    if meta.get("thi_nghiem_ro_ri") and not cho_phep_ro_ri:
        raise ValueError(
            f"{nguon.name} la split THI NGHIEM RO RI ({meta.get('cach_chia')}). "
            "Model train tu no KHONG duoc dua vao bang chinh hay app. "
            "Neu dang chay thi nghiem ro ri that su: doc_split(..., cho_phep_ro_ri=True)."
        )
    return meta


def gan_fold(df_index: pd.DataFrame, meta: dict | None = None) -> pd.DataFrame:
    """Gắn cột `fold` từ file đã đóng băng — KHÔNG chia lại."""
    meta = meta or doc_split()
    gcol = meta["nhom_theo"]
    m = meta["fold_cua_nhom"]
    df = df_index.copy()
    thieu = sorted(set(df[gcol].astype(str)) - set(m))
    if thieu:
        raise ValueError(
            f"{len(thieu)} nhom co trong index nhung KHONG co trong file split "
            f"(vd {thieu[:5]}). Index da doi sau khi split duoc dong bang."
        )
    df["fold"] = df[gcol].astype(str).map(m).astype(int)
    return df
