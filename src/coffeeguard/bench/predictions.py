"""Đọc / ghi / **kiểm chứng** file CSV dự đoán.

Đây là chốt chặn quan trọng nhất của cả tầng benchmark.

`score.py` không biết framework nào, không nạp model nào — nó chỉ đọc CSV. Nghĩa
là mọi sai lệch phải bị bắt ở đây, nếu không nó sẽ đi thẳng vào bảng kết quả:

* notebook tự chia lại fold      -> cột `fold` lệch file split đã đóng băng
* giải mã index->nhãn sai thứ tự -> `y_pred` lệch hẳn phân bố, argmax(p) != y_pred
* thiếu/thừa ảnh                 -> n khác 1.559, macro-F1 không so thẳng được
* quên softmax                   -> tổng p != 1
* chạy lại đè lên file cũ        -> trùng `run_id`

Tất cả những lỗi trên đều KHÔNG gây exception ở phía train. Bảng vẫn ra số đẹp.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import paths as P


class LoiHopDong(ValueError):
    """CSV dự đoán vi phạm hợp đồng — dừng lại, đừng tính chỉ số trên nó."""


def ten_file(framework: str, arch: str, run_id: str) -> str:
    an_toan = lambda s: "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
    return f"pred__{an_toan(framework)}__{an_toan(arch)}__{an_toan(run_id)}.csv"


def ghi_du_doan(
    df: pd.DataFrame,
    framework: str,
    arch: str,
    run_id: str,
    thu_muc: Path = P.PREDICTIONS,
    ghi_de: bool = False,
) -> Path:
    """Ghi CSV đúng hợp đồng. Kiểm chứng TRƯỚC khi ghi, không phải sau."""
    d = df.copy()
    d["framework"], d["arch"], d["run_id"] = framework, arch, run_id
    thieu = [c for c in ct.PRED_COLUMNS if c not in d.columns]
    if thieu:
        raise LoiHopDong(f"Thieu cot bat buoc: {thieu}")
    d = d[list(ct.PRED_COLUMNS)]
    kiem_chung(d, kiem_split=False)

    thu_muc.mkdir(parents=True, exist_ok=True)
    dich = thu_muc / ten_file(framework, arch, run_id)
    if dich.exists() and not ghi_de:
        raise LoiHopDong(
            f"{dich.name} DA TON TAI. Doi `run_id` thay vi ghi de — ghi de lam "
            "mat ket qua cu va khong con truy nguoc duoc bang so sanh."
        )
    d.to_csv(dich, index=False, encoding="utf-8")
    return dich


def kiem_chung(
    pred: pd.DataFrame,
    index: pd.DataFrame | None = None,
    split_meta: dict | None = None,
    kiem_split: bool = True,
) -> list[str]:
    """Ném `LoiHopDong` nếu sai. Trả về danh sách cảnh báo (không chặn)."""
    loi: list[str] = []
    canh_bao: list[str] = []

    thieu = [c for c in ct.PRED_COLUMNS if c not in pred.columns]
    if thieu:
        raise LoiHopDong(f"Thieu cot: {thieu}\nHop dong: {list(ct.PRED_COLUMNS)}")

    # --- một file = một lần chạy ---
    for c in ct.PRED_META_COLUMNS:
        if pred[c].nunique() != 1:
            loi.append(f"Cot '{c}' co {pred[c].nunique()} gia tri — moi file chi mot lan chay")

    # --- nhãn nằm trong hợp đồng ---
    for c in ("y_true", "y_pred"):
        la = sorted(set(pred[c]) - set(ct.CLASSES_6))
        if la:
            loi.append(f"'{c}' chua nhan ngoai hop dong: {la}")

    # --- ảnh không trùng, không thiếu ---
    if pred.png.duplicated().any():
        n = int(pred.png.duplicated().sum())
        loi.append(f"{n} anh xuat hien nhieu hon mot lan — moi anh vao test dung 1 lan")

    # --- xác suất ---
    P_ = pred[list(ct.PRED_PROB_COLUMNS)].to_numpy(dtype=np.float64)
    if not np.isfinite(P_).all():
        loi.append("Cot xac suat co NaN/Inf")
    else:
        tong = P_.sum(axis=1)
        xau = int((np.abs(tong - 1.0) > ct.TOL_XAC_SUAT).sum())
        if xau:
            loi.append(
                f"{xau} dong co tong xac suat != 1 (vd {tong[np.abs(tong-1)>ct.TOL_XAC_SUAT][:3]}). "
                "Thuong la quen softmax, hoac xuat logit tho."
            )
        if (P_ < -1e-6).any():
            loi.append("Co xac suat am")

        # argmax(p) phải khớp y_pred — lệch nghĩa là thứ tự lớp bị giải mã sai
        # bằng danh sách của model khác (quy tắc 10): nhãn sai, confidence cao,
        # không exception.
        argmax = [ct.CLASSES_6[i] for i in P_.argmax(axis=1)]
        lech = int((np.asarray(argmax) != pred.y_pred.to_numpy()).sum())
        if lech:
            loi.append(
                f"{lech}/{len(pred)} dong co argmax(xac suat) KHAC y_pred. "
                "Gan nhu chac chan thu tu lop luc giai ma khong phai CLASSES_6."
            )

    # --- đối chiếu index + split đã đóng băng ---
    if kiem_split:
        from coffeeguard.bench.index import doc_index
        from coffeeguard.bench.splits import doc_split, gan_fold

        idx = index if index is not None else doc_index()
        meta = split_meta if split_meta is not None else doc_split()
        idx = gan_fold(idx, meta)

        thieu_anh = sorted(set(idx.png) - set(pred.png))
        thua_anh = sorted(set(pred.png) - set(idx.png))
        if thieu_anh:
            loi.append(f"Thieu {len(thieu_anh)} anh so voi index (vd {thieu_anh[:3]})")
        if thua_anh:
            loi.append(f"Thua {len(thua_anh)} anh khong co trong index (vd {thua_anh[:3]})")

        if not thieu_anh and not thua_anh:
            g = idx.set_index("png").loc[pred.png]
            lech_fold = int((g.fold.to_numpy() != pred.fold.to_numpy()).sum())
            if lech_fold:
                loi.append(
                    f"{lech_fold}/{len(pred)} anh co 'fold' KHAC file split da dong bang. "
                    "Notebook nay da TU CHIA LAI -> ket qua khong so sanh duoc voi "
                    "cac model khac (quy tac 12)."
                )
            lech_nhan = int((g.label6.to_numpy() != pred.y_true.to_numpy()).sum())
            if lech_nhan:
                loi.append(f"{lech_nhan} anh co 'y_true' khac nhan trong index")
            for c in ("plant_id", "leaf_id", "face"):
                n = int((g[c].astype(str).to_numpy() != pred[c].astype(str).to_numpy()).sum())
                if n:
                    canh_bao.append(f"{n} anh co '{c}' khac index (khong anh huong chi so)")

    # --- cảnh báo, không chặn ---
    dem = pred.y_pred.value_counts()
    if len(dem) == 1:
        canh_bao.append(
            f"Model chi doan DUY NHAT lop '{dem.index[0]}' — accuracy co the ~50,7% "
            "ma macro-F1 rat thap. Day la sup do, khong phai ket qua."
        )
    vang = sorted(set(ct.CLASSES_6) - set(pred.y_pred))
    if vang:
        canh_bao.append(f"Khong bao gio du doan cac lop: {vang}")

    if loi:
        raise LoiHopDong(
            "CSV du doan VI PHAM hop dong:\n  - " + "\n  - ".join(loi)
        )
    return canh_bao


def doc_du_doan(p: Path, kiem: bool = True) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(p)
    canh_bao = kiem_chung(df) if kiem else []
    return df, canh_bao


def liet_ke(thu_muc: Path = P.PREDICTIONS) -> list[Path]:
    return sorted(thu_muc.glob("pred__*.csv")) if thu_muc.exists() else []
