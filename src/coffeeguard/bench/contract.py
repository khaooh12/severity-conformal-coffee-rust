"""HỢP ĐỒNG của tầng benchmark — bất biến, mọi framework phải tuân theo.

Ba thứ được đóng băng ở đây:

1. **Thứ tự lớp** (`CLASSES_6`). Một model train với thứ tự alphabet mà giải mã
   bằng danh sách của model khác sẽ cho nhãn SAI với confidence VẪN CAO và
   không có exception nào (quy tắc 10). Notebook train phải lấy thứ tự từ đây,
   không tự viết lại list.

2. **Ánh xạ 6 lớp -> 5 dải báo cáo** (`BAND_OF`). `rust_level_4` chỉ có 29 ảnh
   nên báo cáo CHÍNH dùng dải; 4 cấp là phụ và phải ghi kèm n.

3. **Định dạng file CSV dự đoán** (`PRED_COLUMNS`). Mọi framework xuất đúng một
   file này; `score.py` không biết framework nào, chỉ biết CSV.

File `web/public/models/registry.json` mang bản sao của (2) cho phía trình duyệt.
`tests/test_bench_contract.py` so hai bản — lệch nhau thì PWA sẽ gộp dải sai mà
vẫn hiện kết quả bình thường.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------- lớp

#: Thứ tự BẤT BIẾN. Index i <-> CLASSES_6[i] ở mọi nơi: nhãn, logits, ONNX.
CLASSES_6: Final[tuple[str, ...]] = (
    "healthy", 
    "red_spider_mite",
    "rust_level_1",
    "rust_level_2",
    "rust_level_3",
    "rust_level_4",
)

#: 5 dải dùng cho BẢNG BÁO CÁO CHÍNH (gộp cấp 3+4 vì cấp 4 chỉ 29 ảnh).
CLASSES_BAND: Final[tuple[str, ...]] = (
    "healthy",
    "red_spider_mite",
    "rust_nhe",
    "rust_trung_binh",
    "rust_nang",
)

BAND_OF: Final[dict[str, str]] = {
    "healthy": "healthy",
    "red_spider_mite": "red_spider_mite",
    "rust_level_1": "rust_nhe",
    "rust_level_2": "rust_trung_binh",
    "rust_level_3": "rust_nang",
    "rust_level_4": "rust_nang",
}

#: Nhãn nhị phân (khoẻ / không khoẻ) — suy từ multiclass, KHÔNG đọc cột `state`
#: của đa giác (nguồn đó lệch 6 ca, xem ghi chú dataset).
BINARY_OF: Final[dict[str, str]] = {
    c: ("healthy" if c == "healthy" else "unhealthy") for c in CLASSES_6
}

#: Lớp hiếm tới mức không đánh giá riêng được — bảng phải ghi kèm n khi nhắc tới.
LOP_QUA_IT: Final[frozenset[str]] = frozenset({"rust_level_3", "rust_level_4"})


def band_of(nhan: str) -> str:
    try:
        return BAND_OF[nhan]
    except KeyError:
        raise KeyError(
            f"Nhan '{nhan}' khong co trong hop dong. Hop le: {list(CLASSES_6)}"
        ) from None


def chi_so_lop(nhan: str) -> int:
    try:
        return CLASSES_6.index(nhan)
    except ValueError:
        raise KeyError(f"Nhan '{nhan}' khong co trong CLASSES_6") from None


# ---------------------------------------------------------------- split

SEED: Final[int] = 42
N_FOLDS: Final[int] = 5
#: Chia theo CÂY, không theo lá và tuyệt đối không theo tên file: hai mặt của
#: cùng chiếc lá mà nằm hai bên train/test là rò rỉ chắc chắn.
GROUP_COL: Final[str] = "plant_id"

# ---------------------------------------------------------------- ảnh

#: Kích thước DUY NHẤT cho mọi bước: ảnh PNG đã khoá, train, đánh giá, export
#: ONNX và app. KHÔNG lấy `input_size` gốc của kiến trúc trong timm: lượt
#: 20260930-1450 đánh giá mobilevitv2_050 ở 256 (phóng PNG 224 lên) nhưng
#: export và app chạy 224 — 9,3% ảnh fold 4 đổi nhãn so với model đã đánh giá,
#: mà phép đối chiếu ONNX-vs-PyTorch (cả hai phía 224) không thấy gì.
#: Đo: `reports/kich_thuoc_dau_vao.json`. Phép chặn: `kiem_kich_thuoc_export()`.
IMG_SIZE: Final[int] = 224


def kich_thuoc_danh_gia(ho_so: dict) -> int | None:
    """Kích thước model đã được ĐÁNH GIÁ, đọc từ hồ sơ trong `de_export.json`.

    Hồ sơ từ 18/10/2026 ghi `kich_thuoc_danh_gia`. Hồ sơ cũ không có khoá đó;
    khi ấy notebook đánh giá ở `input_size` của chính hồ sơ.
    """
    if ho_so.get("kich_thuoc_danh_gia") is not None:
        return int(ho_so["kich_thuoc_danh_gia"])
    kt = ho_so.get("input_size")
    return int(kt[-1]) if kt else None


def kiem_kich_thuoc_export(ho_so: dict, kich_thuoc_export: int) -> None:
    """Ném ValueError nếu model sắp export ở kích thước KHÁC lúc đánh giá.

    Export lệch kích thước vẫn chạy, ONNX vẫn khớp PyTorch (vì cả hai chạy cùng
    kích thước export), app vẫn ra nhãn với confidence cao — chỉ là nhãn của
    một điều kiện chưa từng được đánh giá. Chỉ số trong bảng so sánh khi đó
    không còn mô tả model trong app.
    """
    dg = kich_thuoc_danh_gia(ho_so)
    if dg is None:
        raise ValueError(
            f"{ho_so.get('bi_danh', '?')}: ho so khong ghi kich thuoc danh gia "
            "(thieu ca kich_thuoc_danh_gia lan input_size).")
    if dg != kich_thuoc_export:
        raise ValueError(
            f"{ho_so.get('bi_danh', '?')}: danh gia o {dg} nhung export o "
            f"{kich_thuoc_export}. Chi so trong bang so sanh se KHONG mo ta model "
            f"trong app. Train lai voi kich thuoc {IMG_SIZE} (contract.IMG_SIZE).")


#: `leafsquare` = cửa sổ VUÔNG quanh khung bao lá (dùng mặt nạ RoCoLe).
#: `letterbox` = chèn viền cho cả ảnh thành vuông — giữ lại để làm ablation,
#: vì nó vứt ~86% pixel (lá chỉ còn chiếm trung vị 14,2% khung).
CHE_DO_CAT: Final[tuple[str, ...]] = ("leafsquare", "letterbox")
LEAF_SCALE: Final[float] = 1.4
#: Neo cạnh ô vuông vào cạnh NGẮN của khung bao lá. Lá cà phê dài 2,74:1, nên
#: neo vào cạnh dài cho ô vuông chỉ 11,6% pixel là lá — tệ hơn cả letterbox.
#: Đổi lại, neo cạnh ngắn cắt mất hai đầu lá. Xem `preprocess.py`.
LEAF_NEO: Final[str] = "ngan"

# ---------------------------------------------------------------- CSV dự đoán

#: Cột định danh + sự thật. `fold` phải KHỚP file split đã đóng băng — đó là
#: cách duy nhất phát hiện một notebook đã lặng lẽ tự chia lại (quy tắc 12).
PRED_ID_COLUMNS: Final[tuple[str, ...]] = (
    "png",
    "fold",
    "plant_id",
    "leaf_id",
    "face",
    "y_true",
    "y_pred",
)

#: Cột xác suất, đúng thứ tự CLASSES_6.
PRED_PROB_COLUMNS: Final[tuple[str, ...]] = tuple(f"p_{c}" for c in CLASSES_6)

#: Cột khai báo lần chạy. `run_id` phải khác nhau giữa các lần train.
PRED_META_COLUMNS: Final[tuple[str, ...]] = ("framework", "arch", "run_id")

PRED_COLUMNS: Final[tuple[str, ...]] = (
    PRED_ID_COLUMNS + PRED_PROB_COLUMNS + PRED_META_COLUMNS
)

#: Dung sai khi kiểm tra tổng xác suất = 1 (float32 cộng dồn 6 số).
TOL_XAC_SUAT: Final[float] = 2e-3
