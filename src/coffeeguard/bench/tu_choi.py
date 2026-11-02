"""Cho model quyền "không kết luận", và gộp hai mặt của cùng một lá.

Ba câu hỏi, đo trên file dự đoán out-of-fold sẵn có — không cần train lại:

1. **Độ phủ – độ chính xác.** Nếu app chỉ trả lời khi xác suất lớp cao nhất
   >= ngưỡng, thì trả lời được bao nhiêu ảnh, và trong số đó đúng bao nhiêu?
2. **Gộp hai mặt lá.** Nếu người dùng chụp cả mặt trên lẫn mặt dưới rồi lấy
   trung bình xác suất, kết quả đổi bao nhiêu so với chỉ chụp một mặt?
3. **Hiệu chỉnh (calibration).** Khi model nói "chắc 80%", nó có đúng ~80%
   không? Đo bằng ECE và sơ đồ độ tin cậy. Ngưỡng ở câu 1 chỉ có nghĩa nếu xác
   suất được hiệu chỉnh — nếu không, "ngưỡng 0,7" chỉ là một con số tuỳ ý.

Các bẫy được chặn ở đây:

* **Độ phủ chung che độ phủ từng lớp.** Ở ngưỡng cao, lớp hiếm (cấp 3, cấp 4)
  gần như không bao giờ được trả lời, trong khi độ phủ chung vẫn ~50% nhờ lá
  khoẻ. Vì vậy `duong_do_phu` luôn trả độ phủ THEO TỪNG DẢI.
* **So hai ngưỡng phải so ở cùng độ phủ.** Accuracy 97% ở độ phủ 50% không "tốt
  hơn" accuracy 94% ở độ phủ 62%. Bảng luôn có cột độ phủ cạnh accuracy.
* **macro-F1 trên tập được trả lời** quy ước: dải không còn ảnh nào và không bị
  đoán lần nào tính F1 = 0 (giống `metrics._macro_f1_ma`). Quy ước này BẢO THỦ —
  nó khác `sklearn.f1_score` (sklearn bỏ dải vắng), nên số không khớp sklearn ở
  ngưỡng cao. Đừng "sửa" cho khớp: bỏ dải vắng là thưởng cho model vì đã từ
  chối cả một lớp.
* **Gộp hai mặt chỉ trên các lá cùng nhãn** là tập con LỆCH (nhiều lá khoẻ hơn
  hẳn tập đầy đủ). Mặc định `so_sanh_gop_hai_mat` chấm trên MỌI ảnh: nhãn gộp
  của lá được gán cho cả hai ảnh, rồi so với nhãn của CHÍNH ảnh đó. Lá có hai mặt
  khác nhãn thì nhãn gộp đúng được tối đa một mặt — đó là sự thật, không phải lỗi.
* **Bootstrap theo cây, theo cặp** — cùng lý do như `metrics.bootstrap_chenh_lech`.
* **ECE phụ thuộc cách chia bin.** Số bin và kiểu bin (đều theo độ rộng / đều
  theo số ảnh) đổi được ECE vài điểm. `ece` luôn trả kèm `so_bin`, `kieu_bin`;
  báo cáo phải ghi cả hai, và nên báo cả hai kiểu bin.
* **ECE chung che lệch ở lớp bệnh** — cùng kiểu bẫy với độ phủ chung: lá khoẻ
  chiếm đa số và thường được hiệu chỉnh tốt, kéo ECE xuống. `lech_theo_du_doan`
  tách theo dải ĐƯỢC DỰ ĐOÁN (cái người dùng thấy trên màn hình): "khi app nói
  gỉ sắt nặng, độ tin cậy trung bình bao nhiêu, đúng bao nhiêu".
* ECE ở đây là **top-label, 6 lớp**: tin cậy = xác suất lớp cao nhất, "đúng" =
  `y_true == y_pred` trên 6 lớp — cùng định nghĩa với cột `accuracy` của
  `duong_do_phu`, để ngưỡng và hiệu chỉnh nói về cùng một con số.
* **Trộn hai định nghĩa "đúng" làm lệch biến mất.** Lấy tin cậy 6 lớp (p cấp 3)
  rồi chấm "đúng" theo dải (cấp 3 hay 4 đều là "nặng") thì tin cậy bị tính
  THIẾU so với cái được chấm. `lech_theo_du_doan(dinh_nghia=...)` chỉ cho hai cặp
  nhất quán: `"6_lop"` (tin cậy 6 lớp, đúng 6 lớp — cái app hiện hiển thị) và
  `"dai"` (tin cậy = tổng xác suất của dải dự đoán, đúng theo dải — dùng khi báo
  cáo theo 5 dải). Trên lượt A, `mobilevitv2_050` báo "nặng": lệch −0,095 nếu
  trộn, +0,154 theo dải (`Docs/Khoang_trong_bai_bao.md` §6.2).
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import metrics as bm

COT_XAC_SUAT: tuple[str, ...] = tuple(f"p_{c}" for c in ct.CLASSES_6)
NGUONG_MAC_DINH: tuple[float, ...] = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9)


def ma_tran_xac_suat(pred: pd.DataFrame) -> np.ndarray:
    """Ma trận xác suất (n, 6) theo đúng thứ tự `CLASSES_6`.

    Ném lỗi nếu `y_pred` không phải argmax của các cột xác suất: khi đó một trong
    hai đã bị giải mã sai thứ tự lớp, và mọi ngưỡng tính trên xác suất sẽ áp lên
    nhầm lớp mà không có gì báo.
    """
    thieu = [c for c in COT_XAC_SUAT if c not in pred.columns]
    if thieu:
        raise ValueError(f"File du doan thieu cot xac suat: {thieu}")
    P = pred[list(COT_XAC_SUAT)].to_numpy(dtype=np.float64)
    argmax = np.asarray(ct.CLASSES_6)[P.argmax(axis=1)]
    lech = int((argmax != pred["y_pred"].to_numpy()).sum())
    if lech:
        raise ValueError(
            f"{lech}/{len(pred)} dong co y_pred khac argmax cua xac suat — "
            "nhan va xac suat dang theo hai thu tu lop khac nhau."
        )
    return P


def _mau_bootstrap_cum(
    groups,
    ham: Callable[[np.ndarray], float],
    n_lan: int,
    seed: int,
) -> np.ndarray:
    """Các giá trị `ham` trên `n_lan` mẫu bootstrap theo cụm (đã bỏ NaN).

    Cùng `seed` thì cùng các mẫu với `_bootstrap_cum`, nên CI và p-value tính từ
    hai hàm này luôn khớp nhau.
    """
    g = np.asarray([str(x) for x in groups])
    nhom = np.unique(g)
    vi_tri = {k: np.flatnonzero(g == k) for k in nhom}
    rng = np.random.default_rng(seed)
    mau = np.empty(n_lan, dtype=np.float64)
    for i in range(n_lan):
        chon = rng.choice(nhom, size=len(nhom), replace=True)
        mau[i] = ham(np.concatenate([vi_tri[k] for k in chon]))
    return mau[~np.isnan(mau)]


def p_bootstrap(mau: np.ndarray) -> float:
    """p-value hai phía cho giả thuyết "hiệu = 0", đọc từ phân bố bootstrap của hiệu.

    p = 2 · (min(#{d* <= 0}, #{d* >= 0}) + 1) / (B + 1), chặn ở 1. Số +1 tránh
    p = 0 khi không mẫu nào vượt 0 (p nhỏ nhất là 2/(B+1)). Đây là phép đảo CI
    phân vị: CI 95% không chứa 0 thì p xấp xỉ < 0,05.
    """
    mau = np.asarray(mau, dtype=np.float64)
    if len(mau) == 0:
        return float("nan")
    k = min(int((mau <= 0).sum()), int((mau >= 0).sum()))
    return float(min(1.0, 2 * (k + 1) / (len(mau) + 1)))


def _bootstrap_cum(
    groups,
    ham: Callable[[np.ndarray], float],
    n_lan: int,
    seed: int,
    muc: float,
) -> tuple[float, float]:
    """CI bootstrap theo cụm cho một hàm bất kỳ của tập chỉ số dòng.

    `ham(idx)` nhận mảng chỉ số dòng (có lặp, vì lấy mẫu có hoàn lại) và trả một
    số. Lần lấy mẫu nào `ham` trả NaN (vd không còn ảnh nào được trả lời) thì bỏ.
    """
    mau = _mau_bootstrap_cum(groups, ham, n_lan, seed)
    if len(mau) == 0:
        return float("nan"), float("nan")
    return (float(np.quantile(mau, (1 - muc) / 2)),
            float(np.quantile(mau, 1 - (1 - muc) / 2)))


def duong_do_phu(
    pred: pd.DataFrame,
    nguong: Sequence[float] = NGUONG_MAC_DINH,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
) -> pd.DataFrame:
    """Bảng độ phủ – độ chính xác theo ngưỡng tin cậy, kèm CI95 theo cây.

    Mỗi dòng một ngưỡng. App trả lời ảnh khi xác suất lớp cao nhất >= ngưỡng;
    còn lại là "không kết luận".

    Cột: `nguong`, `n_tra_loi`, `do_phu` (+CI), `accuracy` trên ảnh được trả lời
    (+CI), `macro_f1_dai` trên ảnh được trả lời (+CI, quy ước bảo thủ — xem đầu
    file), và `do_phu__<dải>` cho từng dải trong `CLASSES_BAND`.
    """
    P = ma_tran_xac_suat(pred)
    tin_cay = P.max(axis=1)
    yt6 = pred["y_true"].to_numpy()
    yp6 = pred["y_pred"].to_numpy()
    ytd = np.asarray([ct.band_of(c) for c in yt6])
    ypd = np.asarray([ct.band_of(c) for c in yp6])
    ytd_m = bm._ma_hoa(ytd, ct.CLASSES_BAND)
    ypd_m = bm._ma_hoa(ypd, ct.CLASSES_BAND)
    dung = yt6 == yp6
    groups = pred[group_col].to_numpy()

    hang = []
    for t in nguong:
        nhan = tin_cay >= t

        def _phu(idx, nhan=nhan):
            return float(nhan[idx].mean())

        def _acc(idx, nhan=nhan):
            m = nhan[idx]
            return float(dung[idx][m].mean()) if m.any() else float("nan")

        def _f1(idx, nhan=nhan):
            m = nhan[idx]
            if not m.any():
                return float("nan")
            return bm._macro_f1_ma(ytd_m[idx][m], ypd_m[idx][m], len(ct.CLASSES_BAND))

        tat_ca = np.arange(len(pred))
        dong = {"nguong": float(t), "n_tra_loi": int(nhan.sum())}
        for ten, ham in (("do_phu", _phu), ("accuracy", _acc), ("macro_f1_dai", _f1)):
            lo, hi = _bootstrap_cum(groups, ham, n_bootstrap, seed, muc)
            dong[ten] = round(ham(tat_ca), 4)
            dong[f"{ten}_ci_lo"] = round(lo, 4)
            dong[f"{ten}_ci_hi"] = round(hi, 4)
        for dai in ct.CLASSES_BAND:
            thuoc = ytd == dai
            dong[f"do_phu__{dai}"] = round(float(nhan[thuoc].mean()), 4) if thuoc.any() else float("nan")
        hang.append(dong)
    return pd.DataFrame(hang)


KIEU_BIN: tuple[str, ...] = ("deu_do_rong", "deu_so_anh")


def _gan_bin(tin_cay: np.ndarray, so_bin: int, kieu_bin: str) -> np.ndarray:
    """Chỉ số bin (0..so_bin-1) cho từng ảnh.

    `deu_do_rong`: bin [k/so_bin, (k+1)/so_bin), bin cuối đóng ở 1,0.
    `deu_so_anh`: xếp theo tin cậy, chia thành `so_bin` phần gần bằng số ảnh.
    Ảnh trùng tin cậy có thể rơi vào hai bin kề nhau — chấp nhận, vì bin cố định
    theo giá trị sẽ làm số ảnh lệch hẳn khi model hay ra đúng 1,0.
    """
    if kieu_bin == "deu_do_rong":
        return np.clip((tin_cay * so_bin).astype(int), 0, so_bin - 1)
    if kieu_bin == "deu_so_anh":
        hang = np.empty(len(tin_cay), dtype=np.int64)
        hang[np.argsort(tin_cay, kind="stable")] = np.arange(len(tin_cay))
        return (hang * so_bin) // len(tin_cay)
    raise ValueError(f"kieu_bin phai thuoc {KIEU_BIN}, nhan {kieu_bin!r}")


def _ece_mce(tin_cay: np.ndarray, dung: np.ndarray, so_bin: int,
             kieu_bin: str) -> tuple[float, float]:
    if len(tin_cay) == 0:
        return float("nan"), float("nan")
    b = _gan_bin(tin_cay, so_bin, kieu_bin)
    n = np.bincount(b, minlength=so_bin)
    co = n > 0
    tc_tb = np.bincount(b, weights=tin_cay, minlength=so_bin)[co] / n[co]
    acc = np.bincount(b, weights=dung.astype(float), minlength=so_bin)[co] / n[co]
    lech = np.abs(tc_tb - acc)
    return float((n[co] / len(tin_cay) * lech).sum()), float(lech.max())


def _tin_cay_va_dung(pred: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    P = ma_tran_xac_suat(pred)
    return P.max(axis=1), (pred["y_true"].to_numpy() == pred["y_pred"].to_numpy())


def so_do_tin_cay(
    pred: pd.DataFrame,
    so_bin: int = 10,
    kieu_bin: str = "deu_do_rong",
) -> pd.DataFrame:
    """Dữ liệu cho sơ đồ độ tin cậy (reliability diagram). Một dòng mỗi bin.

    Cột: `bin`, `tin_cay_min`, `tin_cay_max` (giá trị thật trong bin, không phải
    mép bin), `n`, `tin_cay_tb`, `accuracy`, `lech` = `tin_cay_tb - accuracy`
    (dương = model QUÁ TỰ TIN). Bin rỗng vẫn có dòng, với `n` = 0 và NaN, để đồ
    thị không nối liền qua chỗ trống.
    """
    tin_cay, dung = _tin_cay_va_dung(pred)
    b = _gan_bin(tin_cay, so_bin, kieu_bin)
    hang = []
    for k in range(so_bin):
        m = b == k
        if m.any():
            tb, acc = float(tin_cay[m].mean()), float(dung[m].mean())
            hang.append({"bin": k, "tin_cay_min": round(float(tin_cay[m].min()), 4),
                         "tin_cay_max": round(float(tin_cay[m].max()), 4),
                         "n": int(m.sum()), "tin_cay_tb": round(tb, 4),
                         "accuracy": round(acc, 4), "lech": round(tb - acc, 4)})
        else:
            hang.append({"bin": k, "tin_cay_min": np.nan, "tin_cay_max": np.nan,
                         "n": 0, "tin_cay_tb": np.nan, "accuracy": np.nan,
                         "lech": np.nan})
    return pd.DataFrame(hang)


def ece(
    pred: pd.DataFrame,
    so_bin: int = 10,
    kieu_bin: str = "deu_do_rong",
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
) -> dict:
    """ECE và MCE top-label 6 lớp, kèm CI theo cây.

    ECE = tổng theo bin (n_bin / n) · |tin cậy TB − accuracy|; MCE = lệch lớn nhất
    trong các bin không rỗng. Với `deu_so_anh`, bin được chia lại trên MỖI mẫu
    bootstrap (bin phụ thuộc dữ liệu, cố định bin từ mẫu gốc sẽ làm CI hẹp giả).

    `lech_tb` = tin cậy TB − accuracy trên mọi ảnh: dương là quá tự tin. ECE luôn
    >= |lech_tb|; ECE lớn mà |lech_tb| nhỏ nghĩa là quá tự tin ở vùng này, thiếu
    tự tin ở vùng kia.
    """
    tin_cay, dung = _tin_cay_va_dung(pred)
    groups = pred[group_col].to_numpy()

    def _e(idx):
        return _ece_mce(tin_cay[idx], dung[idx], so_bin, kieu_bin)[0]

    def _lech(idx):
        return float(tin_cay[idx].mean() - dung[idx].mean())

    gia_tri, mce = _ece_mce(tin_cay, dung, so_bin, kieu_bin)
    lo, hi = _bootstrap_cum(groups, _e, n_bootstrap, seed, muc)
    l_lo, l_hi = _bootstrap_cum(groups, _lech, n_bootstrap, seed, muc)
    return {
        "ece": round(gia_tri, 4), "ece_ci": [round(lo, 4), round(hi, 4)],
        "mce": round(mce, 4),
        "lech_tb": round(_lech(np.arange(len(pred))), 4),
        "lech_tb_ci": [round(l_lo, 4), round(l_hi, 4)],
        "so_bin": so_bin, "kieu_bin": kieu_bin, "n_anh": int(len(pred)),
        "muc_tin_cay": muc, "n_bootstrap": n_bootstrap,
        "don_vi_lay_mau": "cum (cay)",
    }


def xac_suat_dai(pred: pd.DataFrame) -> np.ndarray:
    """Ma trận xác suất (n, 5) theo thứ tự `CLASSES_BAND`: cộng xác suất 6 lớp
    vào dải của nó (cấp 3 + cấp 4 -> nặng)."""
    P = ma_tran_xac_suat(pred)
    Q = np.zeros((len(pred), len(ct.CLASSES_BAND)))
    for i, c in enumerate(ct.CLASSES_6):
        Q[:, ct.CLASSES_BAND.index(ct.band_of(c))] += P[:, i]
    return Q


DINH_NGHIA_DUNG: tuple[str, ...] = ("6_lop", "dai")


def _tin_cay_va_dung_theo(pred: pd.DataFrame, dinh_nghia: str) -> tuple[np.ndarray, np.ndarray]:
    """Cặp (tin cậy, đúng) nhất quán theo `dinh_nghia` — xem đầu file."""
    if dinh_nghia not in DINH_NGHIA_DUNG:
        raise ValueError(f"dinh_nghia phai thuoc {DINH_NGHIA_DUNG}, nhan {dinh_nghia!r}")
    if dinh_nghia == "6_lop":
        return _tin_cay_va_dung(pred)
    Q = xac_suat_dai(pred)
    yp = bm._ma_hoa([ct.band_of(c) for c in pred["y_pred"]], ct.CLASSES_BAND)
    yt = bm._ma_hoa([ct.band_of(c) for c in pred["y_true"]], ct.CLASSES_BAND)
    return Q[np.arange(len(pred)), yp], yp == yt


def lech_theo_du_doan(
    pred: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
    dinh_nghia: str = "6_lop",
) -> pd.DataFrame:
    """Hiệu chỉnh tách theo dải ĐƯỢC DỰ ĐOÁN. Một dòng mỗi dải trong `CLASSES_BAND`.

    Trả lời câu người dùng thật sự gặp: "app nói *gỉ sắt nặng*, chắc X% — tin
    được không?". Cột: `dai_du_doan`, `n`, `tin_cay_tb`, `accuracy`, `lech` (+CI
    theo cây; dương = quá tự tin), `dinh_nghia`. Dải chưa bao giờ được dự đoán có
    `n` = 0 và NaN.

    `dinh_nghia`: `"6_lop"` — tin cậy = xác suất lớp cao nhất, đúng = khớp 6 lớp;
    `"dai"` — tin cậy = xác suất của dải dự đoán, đúng = khớp dải. Không có lựa
    chọn trộn (xem đầu file).
    """
    tin_cay, dung = _tin_cay_va_dung_theo(pred, dinh_nghia)
    dai = np.asarray([ct.band_of(c) for c in pred["y_pred"]])
    groups = pred[group_col].to_numpy()
    hang = []
    for d in ct.CLASSES_BAND:
        m = dai == d
        if not m.any():
            hang.append({"dai_du_doan": d, "n": 0, "tin_cay_tb": np.nan,
                         "accuracy": np.nan, "lech": np.nan,
                         "lech_ci_lo": np.nan, "lech_ci_hi": np.nan,
                         "dinh_nghia": dinh_nghia})
            continue

        def _lech(idx, m=m):
            mm = m[idx]
            if not mm.any():
                return float("nan")
            return float(tin_cay[idx][mm].mean() - dung[idx][mm].mean())

        lo, hi = _bootstrap_cum(groups, _lech, n_bootstrap, seed, muc)
        tb, acc = float(tin_cay[m].mean()), float(dung[m].mean())
        hang.append({"dai_du_doan": d, "n": int(m.sum()), "tin_cay_tb": round(tb, 4),
                     "accuracy": round(acc, 4), "lech": round(tb - acc, 4),
                     "lech_ci_lo": round(lo, 4), "lech_ci_hi": round(hi, 4),
                     "dinh_nghia": dinh_nghia})
    return pd.DataFrame(hang)


def gop_hai_mat(pred: pd.DataFrame) -> pd.DataFrame:
    """Gộp hai mặt của mỗi lá bằng trung bình xác suất. Một dòng mỗi ẢNH.

    Trả lại đúng các dòng của `pred` (cùng thứ tự), thêm cột:
    `y_pred_gop` (nhãn sau khi gộp), `tin_cay_gop`, `nhan_hai_mat_khac` (True nếu
    hai mặt của lá có `y_true` khác nhau), `mot_mat` (True nếu lá chỉ có một ảnh).

    Lá chỉ có một ảnh (RoCoLe trong đồ án: 1.559 ảnh, thiếu một mặt) giữ nguyên
    dự đoán của chính nó — số lá như vậy hiện ở `so_sanh_gop_hai_mat`.

    Ném lỗi nếu một lá có hơn hai ảnh (`leaf_id` sai), hoặc hai mặt nằm ở hai fold
    khác nhau — dấu hiệu file split đã rò rỉ, và khi đó dự đoán của một mặt đến từ
    model đã thấy mặt kia lúc train.
    """
    P = ma_tran_xac_suat(pred)
    if "leaf_id" not in pred.columns:
        raise ValueError("File du doan thieu cot leaf_id")
    leaf = pred["leaf_id"].astype(str).to_numpy()

    so_anh = pd.Series(leaf).value_counts()
    sai = so_anh[so_anh > 2]
    if len(sai):
        raise ValueError(
            f"{len(sai)} la co hon 2 anh (vd {sai.index[0]}: {sai.iloc[0]} anh) — "
            "leaf_id sai."
        )
    so_fold = pred.groupby(leaf)["fold"].nunique()
    ro_ri = so_fold[so_fold > 1]
    if len(ro_ri):
        raise ValueError(
            f"{len(ro_ri)} la co hai mat o hai fold khac nhau (vd {ro_ri.index[0]}) — "
            "split dang ro ri."
        )

    trung_binh = pd.DataFrame(P).groupby(leaf).transform("mean").to_numpy()
    so_nhan = pred.groupby(leaf)["y_true"].transform("nunique").to_numpy()
    ra = pred.copy()
    ra["y_pred_gop"] = np.asarray(ct.CLASSES_6)[trung_binh.argmax(axis=1)]
    ra["tin_cay_gop"] = trung_binh.max(axis=1)
    ra["nhan_hai_mat_khac"] = so_nhan > 1
    ra["mot_mat"] = pd.Series(leaf).map(so_anh).to_numpy() == 1
    return ra


def so_sanh_gop_hai_mat(
    pred: pd.DataFrame,
    chi_la_cung_nhan: bool = False,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
) -> dict:
    """Gộp hai mặt so với một mặt, TRÊN CÙNG CÁC ẢNH, CI95 của hiệu theo cây.

    Mỗi ảnh có hai dự đoán: của riêng nó, và của lá sau khi gộp. Cả hai so với
    `y_true` của chính ảnh, nên hai phép đo nằm trên cùng dòng — so theo cặp.

    `chi_la_cung_nhan=True` chỉ giữ lá có hai mặt cùng nhãn. Tập con này lệch về
    lá khoẻ; chỉ dùng để đối chiếu, không dùng làm số chính.
    """
    g = gop_hai_mat(pred)
    if chi_la_cung_nhan:
        g = g[~g["nhan_hai_mat_khac"]].reset_index(drop=True)

    yt = [ct.band_of(c) for c in g["y_true"]]
    y_gop = [ct.band_of(c) for c in g["y_pred_gop"]]
    y_mot = [ct.band_of(c) for c in g["y_pred"]]
    f1 = bm.bootstrap_chenh_lech(yt, y_gop, y_mot, g[group_col], ct.CLASSES_BAND,
                                 n_bootstrap, seed, muc)

    dung_gop = (g["y_true"] == g["y_pred_gop"]).to_numpy()
    dung_mot = (g["y_true"] == g["y_pred"]).to_numpy()
    hieu = dung_gop.astype(float) - dung_mot.astype(float)
    lo, hi = _bootstrap_cum(g[group_col], lambda idx: float(hieu[idx].mean()),
                            n_bootstrap, seed, muc)
    return {
        "n_anh": int(len(g)),
        "n_la": int(g["leaf_id"].nunique()),
        "n_la_mot_mat": int(g.loc[g["mot_mat"], "leaf_id"].nunique()),
        "chi_la_cung_nhan": chi_la_cung_nhan,
        "macro_f1_dai_mot_mat": round(bm.macro_f1(yt, y_mot, ct.CLASSES_BAND), 4),
        "macro_f1_dai_gop": round(bm.macro_f1(yt, y_gop, ct.CLASSES_BAND), 4),
        "hieu_macro_f1_dai": f1["hieu"],
        "hieu_macro_f1_dai_ci": [f1["ci_lo"], f1["ci_hi"]],
        "accuracy_mot_mat": round(float(dung_mot.mean()), 4),
        "accuracy_gop": round(float(dung_gop.mean()), 4),
        "hieu_accuracy": round(float(hieu.mean()), 4),
        "hieu_accuracy_ci": [round(lo, 4), round(hi, 4)],
        "ket_luan_duoc_f1": f1["ket_luan_duoc"],
        "ket_luan_duoc_accuracy": bool(lo > 0 or hi < 0),
        "muc_tin_cay": muc,
        "n_bootstrap": n_bootstrap,
        "don_vi_lay_mau": "cum (cay), THEO CAP",
    }


# ------------------------------------------------------------ hai biện pháp hiển nhiên
# `Docs/Khoang_trong_bai_bao.md` §6.3 — cả hai đều KHÔNG sửa được lệch ở lá bệnh;
# giữ ở đây vì kết quả âm là lý do cho tập conformal (`conformal.py`). Cả hai học
# tham số trên 4 fold KHÁC rồi áp lên fold còn lại: học trên chính fold được chấm
# làm ECE / accuracy đẹp giả mà không có gì báo.
#
# Giới hạn: fold khác là MODEL khác (lý do `conformal.py` hiệu chỉnh trong fold).
# Ở đây chấp nhận vì T học được gần như bằng nhau giữa các fold (xem `T_theo_fold`);
# nếu chúng lệch xa nhau thì kết quả âm này không còn đọc được là "biện pháp thất bại".

LUOI_NHIET_DO: np.ndarray = np.exp(np.linspace(np.log(0.2), np.log(5.0), 200))


def _chia_nhiet_do(logp: np.ndarray, T: float) -> np.ndarray:
    z = logp / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _hoc_nhiet_do(logp: np.ndarray, y: np.ndarray, luoi: np.ndarray) -> float:
    """T trên lưới có NLL nhỏ nhất. Lưới (không tối ưu liên tục) để kết quả tất định."""
    hang = np.arange(len(y))
    nll = [-np.log(_chia_nhiet_do(logp, T)[hang, y] + 1e-12).mean() for T in luoi]
    return float(luoi[int(np.argmin(nll))])


def nhiet_do_cheo_fold(
    pred: pd.DataFrame,
    luoi: np.ndarray = LUOI_NHIET_DO,
) -> tuple[pd.DataFrame, dict[int, float]]:
    """Temperature scaling một nhiệt độ, học trên các fold khác.

    File dự đoán chỉ có xác suất, không có logit; log xác suất bằng logit trừ một
    hằng số mỗi dòng nên softmax(log p / T) đúng là temperature scaling. Trả bản
    sao `pred` với xác suất mới (argmax không đổi vì T > 0) và {fold: T}.
    T < 1 làm xác suất nhọn hơn (model đang THIẾU tự tin), T > 1 làm tù hơn.
    """
    logp = np.log(np.clip(ma_tran_xac_suat(pred), 1e-12, 1.0))
    y = bm._ma_hoa(pred["y_true"], ct.CLASSES_6)
    fold = pred["fold"].to_numpy()
    P2 = np.empty_like(logp)
    T_theo_fold: dict[int, float] = {}
    for f in np.unique(fold):
        m = fold == f
        T = _hoc_nhiet_do(logp[~m], y[~m], luoi)
        T_theo_fold[int(f)] = round(T, 4)
        P2[m] = _chia_nhiet_do(logp[m], T)
    out = pred.copy()
    for i, c in enumerate(ct.CLASSES_6):
        out[f"p_{c}"] = P2[:, i]
    out["y_pred"] = np.asarray(ct.CLASSES_6)[P2.argmax(axis=1)]
    return out, T_theo_fold


def danh_gia_nhiet_do(
    pred: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> dict:
    """ECE trước/sau temperature scaling chéo fold, và lệch khi báo từng dải bệnh
    (định nghĩa `"dai"`) trước/sau. ECE chung giảm KHÔNG có nghĩa lệch ở lá nặng
    giảm — đó là điều hàm này phải cho thấy."""
    sau, T = nhiet_do_cheo_fold(pred)
    e0 = ece(pred, n_bootstrap=n_bootstrap, seed=seed, group_col=group_col)
    e1 = ece(sau, n_bootstrap=n_bootstrap, seed=seed, group_col=group_col)
    l0 = lech_theo_du_doan(pred, n_bootstrap=n_bootstrap, seed=seed,
                           group_col=group_col, dinh_nghia="dai").set_index("dai_du_doan")
    l1 = lech_theo_du_doan(sau, n_bootstrap=n_bootstrap, seed=seed,
                           group_col=group_col, dinh_nghia="dai").set_index("dai_du_doan")
    kq = {"T_theo_fold": T, "ece_truoc": e0["ece"], "ece_truoc_ci": e0["ece_ci"],
          "ece_sau": e1["ece"], "ece_sau_ci": e1["ece_ci"]}
    for d in ct.CLASSES_BAND:
        kq[f"lech_dai__{d}__truoc"] = l0.loc[d, "lech"]
        kq[f"lech_dai__{d}__sau"] = l1.loc[d, "lech"]
        kq[f"lech_dai__{d}__sau_ci"] = [float(l1.loc[d, "lech_ci_lo"]), float(l1.loc[d, "lech_ci_hi"])]
    return kq


LUOI_NGUONG: np.ndarray = np.round(np.arange(0.20, 1.00, 0.01), 2)


def nguong_theo_dai(
    pred: pd.DataFrame,
    muc_tieu: float = 0.85,
    n_toi_thieu: int = 10,
    dinh_nghia: str = "dai",
) -> tuple[np.ndarray, pd.DataFrame]:
    """Mỗi dải DỰ ĐOÁN một ngưỡng riêng, học trên các fold khác.

    Ngưỡng của (fold f, dải b) = ngưỡng nhỏ nhất trên `LUOI_NGUONG` mà trên các
    fold ≠ f, các ảnh được đoán dải b có tin cậy >= ngưỡng (ít nhất `n_toi_thieu`
    ảnh) đạt accuracy >= `muc_tieu`. Không ngưỡng nào đạt -> dải đó không bao giờ
    được trả lời ở fold f (ngưỡng = +∞).

    Trả (mặt nạ "được trả lời" theo dòng, bảng ngưỡng fold × dải). Tin cậy và
    "đúng" theo `dinh_nghia` (xem đầu file) — không trộn.
    """
    tin_cay, dung = _tin_cay_va_dung_theo(pred, dinh_nghia)
    dai = np.asarray([ct.band_of(c) for c in pred["y_pred"]])
    fold = pred["fold"].to_numpy()
    tra_loi = np.zeros(len(pred), dtype=bool)
    hang = []
    for f in np.unique(fold):
        m = fold == f
        for b in ct.CLASSES_BAND:
            hoc = (~m) & (dai == b)
            t_chon = np.inf
            for t in LUOI_NGUONG:
                s = hoc & (tin_cay >= t)
                if s.sum() >= n_toi_thieu and dung[s].mean() >= muc_tieu:
                    t_chon = float(t)
                    break
            ap = m & (dai == b)
            tra_loi[ap] = tin_cay[ap] >= t_chon
            hang.append({"fold": int(f), "dai_du_doan": b, "nguong": t_chon})
    return tra_loi, pd.DataFrame(hang)


def so_nguong_cung_do_phu(
    pred: pd.DataFrame,
    muc_tieu: float = 0.85,
    n_toi_thieu: int = 10,
    dinh_nghia: str = "dai",
) -> pd.DataFrame:
    """Ngưỡng theo dải so với MỘT ngưỡng chung, ở CÙNG độ phủ tổng.

    Ngưỡng chung chọn bằng lượng tử để trả lời đúng số ảnh mà ngưỡng theo dải trả
    lời (so ở độ phủ khác nhau là so sai — xem đầu file). Hai dòng `chung` /
    `theo_dai`, cột: `do_phu`, `accuracy`, `do_phu__<dải thật>`.
    """
    tra_dai, _ = nguong_theo_dai(pred, muc_tieu, n_toi_thieu, dinh_nghia)
    tin_cay, dung = _tin_cay_va_dung_theo(pred, dinh_nghia)
    k = int(tra_dai.sum())
    tra_chung = np.zeros(len(pred), dtype=bool)
    if k:
        tra_chung[np.argsort(-tin_cay, kind="stable")[:k]] = True
    yt = np.asarray([ct.band_of(c) for c in pred["y_true"]])
    hang = []
    for ten, m in (("chung", tra_chung), ("theo_dai", tra_dai)):
        dong = {"cach": ten, "dinh_nghia": dinh_nghia, "muc_tieu": muc_tieu,
                "do_phu": round(float(m.mean()), 4),
                "accuracy": round(float(dung[m].mean()), 4) if m.any() else np.nan}
        for b in ct.CLASSES_BAND:
            thuoc = yt == b
            dong[f"do_phu__{b}"] = round(float(m[thuoc].mean()), 4) if thuoc.any() else np.nan
        hang.append(dong)
    return pd.DataFrame(hang)
