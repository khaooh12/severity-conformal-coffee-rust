"""Chỉ số đánh giá — và những cái bẫy đi kèm.

Ba điều bắt buộc, không phải tuỳ chọn:

1. **Accuracy là chỉ số vô dụng ở đây.** `healthy` chiếm 50,7%, nên một model
   đoán "healthy" mọi lúc đạt 50,7%. Mọi hàm ở đây luôn trả accuracy KÈM
   `accuracy_da_doan_bua` để không ai đọc nhầm.

2. **Phải ghi n của từng lớp.** `rust_level_4` có 29 ảnh; recall của nó nhảy
   3,4 điểm phần trăm mỗi khi đúng/sai thêm một ảnh.

3. **Bootstrap phải theo CỤM (cây), không theo ảnh.** Bốn ảnh của cùng một cây
   không độc lập (cùng lá, cùng buổi chụp, cùng nền). Lấy mẫu lại từng ảnh sẽ
   cho khoảng tin cậy HẸP HƠN sự thật -> "model A hơn model B có ý nghĩa thống
   kê" trong khi không phải.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from coffeeguard.bench import contract as ct


def _ma_hoa(nhan, classes) -> np.ndarray:
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[x] for x in nhan], dtype=np.int64)


def macro_f1(y_true, y_pred, classes=ct.CLASSES_6) -> float:
    return _macro_f1_ma(_ma_hoa(y_true, classes), _ma_hoa(y_pred, classes), len(classes))


def _macro_f1_ma(yt: np.ndarray, yp: np.ndarray, so_lop: int) -> float:
    """macro-F1 trên nhãn ĐÃ mã hoá thành số. Bootstrap gọi thẳng hàm này để
    không mã hoá lại chuỗi ở mỗi lần lấy mẫu — cùng công thức, cùng kết quả."""
    f1s = []
    for k in range(so_lop):
        tp = int(((yt == k) & (yp == k)).sum())
        fp = int(((yt != k) & (yp == k)).sum())
        fn = int(((yt == k) & (yp != k)).sum())
        f1s.append(0.0 if (2 * tp + fp + fn) == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def bang_tung_lop(y_true, y_pred, classes=ct.CLASSES_6) -> pd.DataFrame:
    yt, yp = _ma_hoa(y_true, classes), _ma_hoa(y_pred, classes)
    rows = []
    for k, c in enumerate(classes):
        tp = int(((yt == k) & (yp == k)).sum())
        fp = int(((yt != k) & (yp == k)).sum())
        fn = int(((yt == k) & (yp != k)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {
                "lop": c,
                "n": tp + fn,          # n THẬT — bắt buộc hiện trong báo cáo
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "hiem": c in ct.LOP_QUA_IT,
            }
        )
    return pd.DataFrame(rows)


def ma_tran_nham_lan(y_true, y_pred, classes=ct.CLASSES_6) -> pd.DataFrame:
    yt, yp = _ma_hoa(y_true, classes), _ma_hoa(y_pred, classes)
    m = np.zeros((len(classes), len(classes)), dtype=int)
    for a, b in zip(yt, yp):
        m[a, b] += 1
    return pd.DataFrame(m, index=list(classes), columns=list(classes))


def bootstrap_cum(
    y_true,
    y_pred,
    groups,
    classes=ct.CLASSES_6,
    n_lan: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
) -> dict:
    """Khoảng tin cậy bootstrap **theo cụm** cho macro-F1.

    Lấy mẫu lại CÂY (có hoàn lại), không lấy mẫu lại ảnh. Ảnh trong cùng một cây
    tương quan mạnh; bỏ qua điều đó làm khoảng tin cậy hẹp giả tạo.
    """
    yt = np.asarray(list(y_true))
    yp = np.asarray(list(y_pred))
    g = np.asarray([str(x) for x in groups])
    yt_m, yp_m = _ma_hoa(yt, classes), _ma_hoa(yp, classes)

    nhom = np.unique(g)
    vi_tri = {k: np.flatnonzero(g == k) for k in nhom}
    rng = np.random.default_rng(seed)

    mau = np.empty(n_lan, dtype=np.float64)
    for i in range(n_lan):
        chon = rng.choice(nhom, size=len(nhom), replace=True)
        idx = np.concatenate([vi_tri[k] for k in chon])
        mau[i] = _macro_f1_ma(yt_m[idx], yp_m[idx], len(classes))

    lo = float(np.quantile(mau, (1 - muc) / 2))
    hi = float(np.quantile(mau, 1 - (1 - muc) / 2))
    return {
        "diem": round(macro_f1(yt, yp, classes), 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "muc_tin_cay": muc,
        "n_bootstrap": n_lan,
        "don_vi_lay_mau": "cum (cay)",
        "n_cum": int(len(nhom)),
    }


@dataclass
class KetQuaDanhGia:
    run_id: str
    framework: str
    arch: str
    n_anh: int
    macro_f1_6: float
    macro_f1_dai: float
    accuracy: float
    accuracy_da_doan_bua: float
    theo_fold: list[float]
    f1_trung_binh_fold: float
    f1_do_lech_fold: float
    ci_6: dict = field(default_factory=dict)
    ci_dai: dict = field(default_factory=dict)
    bang_lop: pd.DataFrame | None = None
    bang_dai: pd.DataFrame | None = None
    nham_lan: pd.DataFrame | None = None

    def dong_tom_tat(self) -> dict:
        return {
            "run_id": self.run_id,
            "framework": self.framework,
            "arch": self.arch,
            "n": self.n_anh,
            "macroF1_5dai": round(self.macro_f1_dai, 4),
            "macroF1_5dai_CI95": f"[{self.ci_dai.get('ci_lo')}, {self.ci_dai.get('ci_hi')}]",
            "macroF1_6lop": round(self.macro_f1_6, 4),
            "F1_tb_5fold": round(self.f1_trung_binh_fold, 4),
            "F1_do_lech_fold": round(self.f1_do_lech_fold, 4),
            "accuracy": round(self.accuracy, 4),
            "acc_doan_bua": round(self.accuracy_da_doan_bua, 4),
        }


def danh_gia(
    pred: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> KetQuaDanhGia:
    """Đánh giá một file dự đoán đã gộp đủ 5 fold."""
    yt6 = pred.y_true.tolist()
    yp6 = pred.y_pred.tolist()
    ytd = [ct.band_of(c) for c in yt6]
    ypd = [ct.band_of(c) for c in yp6]

    theo_fold = [
        macro_f1(sub.y_true, sub.y_pred, ct.CLASSES_6)
        for _, sub in pred.groupby("fold", sort=True)
    ]
    dem = pd.Series(yt6).value_counts()

    # Cot khai bao lan chay la viec cua `predictions.kiem_chung`, khong phai cua
    # ham nay. Cham diem mot DataFrame chua gan nhan (vd ngay sau khi train, hoac
    # baseline trong notebook) van phai chay duoc, chi la nhan "?".
    def _meta(ten: str) -> str:
        return str(pred[ten].iat[0]) if ten in pred.columns and len(pred) else "?"

    return KetQuaDanhGia(
        run_id=_meta("run_id"),
        framework=_meta("framework"),
        arch=_meta("arch"),
        n_anh=int(len(pred)),
        macro_f1_6=macro_f1(yt6, yp6, ct.CLASSES_6),
        macro_f1_dai=macro_f1(ytd, ypd, ct.CLASSES_BAND),
        accuracy=float(np.mean(np.asarray(yt6) == np.asarray(yp6))),
        accuracy_da_doan_bua=float(dem.iloc[0] / dem.sum()),
        theo_fold=[round(x, 4) for x in theo_fold],
        f1_trung_binh_fold=float(np.mean(theo_fold)),
        f1_do_lech_fold=float(np.std(theo_fold, ddof=1)) if len(theo_fold) > 1 else 0.0,
        ci_6=bootstrap_cum(yt6, yp6, pred[group_col], ct.CLASSES_6, n_bootstrap, seed),
        ci_dai=bootstrap_cum(ytd, ypd, pred[group_col], ct.CLASSES_BAND, n_bootstrap, seed),
        bang_lop=bang_tung_lop(yt6, yp6, ct.CLASSES_6),
        bang_dai=bang_tung_lop(ytd, ypd, ct.CLASSES_BAND),
        nham_lan=ma_tran_nham_lan(yt6, yp6, ct.CLASSES_6),
    )


def bootstrap_chenh_lech(
    y_true,
    y_pred_a,
    y_pred_b,
    groups,
    classes=ct.CLASSES_BAND,
    n_lan: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
) -> dict:
    """Khoảng tin cậy bootstrap cho **HIỆU** macro-F1 giữa hai model — theo CẶP.

    Vì sao phải có hàm này, không dùng "hai CI95 có giao nhau không"
    ---------------------------------------------------------------
    Bốn model được đánh giá trên **CÙNG 1.559 ảnh**, cùng một file split đã đóng
    băng. Nên đây là so sánh **theo cặp**, không phải hai mẫu độc lập.

    So hai khoảng tin cậy **biên** là phép thử sai và **quá bảo thủ**: hai CI có
    thể giao nhau trong khi hiệu của chúng vẫn khác 0 rõ rệt. Lý do là phần lớn
    độ rộng của mỗi CI đến từ *fold/cây nào rơi vào đâu* — một nguồn biến thiên
    mà **cả hai model cùng chịu**, nên nó triệt tiêu khi lấy hiệu.

    Bằng chứng ngay trong dữ liệu đồ án: `mobilevitv2_050` hơn `mobilenetv2` ở
    **cả 5/5 fold**, chênh đều +0,0148…+0,1149, vậy mà CI95 biên vẫn giao nhau.

    Cách đúng: mỗi lần bootstrap lấy lại **cùng một tập cây** rồi tính macro-F1
    của CẢ HAI model trên đúng tập đó, lấy hiệu. CI của hiệu không chứa 0 thì
    kết luận được.

    Hàm này KHÔNG thay `bootstrap_cum` — CI biên vẫn cần để báo cáo độ bất định
    của từng model. Hai thứ trả lời hai câu khác nhau.
    """
    yt = np.asarray(list(y_true))
    ya = np.asarray(list(y_pred_a))
    yb = np.asarray(list(y_pred_b))
    g = np.asarray([str(x) for x in groups])
    if not (len(yt) == len(ya) == len(yb) == len(g)):
        raise ValueError(
            f"Do dai lech: y_true={len(yt)} a={len(ya)} b={len(yb)} groups={len(g)}. "
            "Hai model phai duoc can theo cung thu tu anh truoc khi goi ham nay."
        )

    yt_m, ya_m, yb_m = (_ma_hoa(v, classes) for v in (yt, ya, yb))
    nhom = np.unique(g)
    vi_tri = {k: np.flatnonzero(g == k) for k in nhom}
    rng = np.random.default_rng(seed)

    mau = np.empty(n_lan, dtype=np.float64)
    for i in range(n_lan):
        chon = rng.choice(nhom, size=len(nhom), replace=True)
        idx = np.concatenate([vi_tri[k] for k in chon])
        # CUNG mot idx cho ca hai -> bien thien do chon cay bi triet tieu
        mau[i] = (_macro_f1_ma(yt_m[idx], ya_m[idx], len(classes))
                  - _macro_f1_ma(yt_m[idx], yb_m[idx], len(classes)))

    lo = float(np.quantile(mau, (1 - muc) / 2))
    hi = float(np.quantile(mau, 1 - (1 - muc) / 2))
    diem = macro_f1(yt, ya, classes) - macro_f1(yt, yb, classes)
    return {
        "hieu": round(float(diem), 4),
        "ci_lo": round(lo, 4),
        "ci_hi": round(hi, 4),
        "ket_luan_duoc": bool(lo > 0 or hi < 0),
        "ty_le_thang": round(float(np.mean(mau > 0)), 4),
        "muc_tin_cay": muc,
        "n_bootstrap": n_lan,
        "don_vi_lay_mau": "cum (cay), THEO CAP",
        "n_cum": int(len(nhom)),
    }


def so_sanh_theo_cap(
    ten_a: str,
    ten_b: str,
    pred_a,
    pred_b,
    classes=ct.CLASSES_BAND,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> dict:
    """So hai file dự đoán theo cặp. Tự căn theo cột `png` trước khi so.

    Căn tường minh chứ không tin thứ tự dòng: hai notebook khác nhau có thể ghi
    theo thứ tự khác, và nếu lệch thì hiệu vẫn tính ra một con số **trông hợp
    lý** mà không có gì báo.
    """
    a = pred_a.set_index("png").sort_index()
    b = pred_b.set_index("png").sort_index()
    chung = a.index.intersection(b.index)
    if len(chung) != len(a) or len(chung) != len(b):
        raise ValueError(
            f"{ten_a} va {ten_b} khong cung tap anh: {len(a)} / {len(b)} / "
            f"chung {len(chung)}. Hai model phai chay tren cung file split."
        )
    a, b = a.loc[chung], b.loc[chung]
    if not (a.y_true.values == b.y_true.values).all():
        raise ValueError(
            f"{ten_a} va {ten_b} co y_true KHAC nhau tren cung anh — mot trong "
            "hai file da giai ma nhan sai thu tu lop."
        )

    dung_dai = tuple(classes) == tuple(ct.CLASSES_BAND)
    yt = [ct.band_of(c) for c in a.y_true] if dung_dai else list(a.y_true)
    ya = [ct.band_of(c) for c in a.y_pred] if dung_dai else list(a.y_pred)
    yb = [ct.band_of(c) for c in b.y_pred] if dung_dai else list(b.y_pred)

    kq = bootstrap_chenh_lech(yt, ya, yb, a[group_col], classes, n_bootstrap, seed)
    kq["a"], kq["b"] = ten_a, ten_b
    return kq


def cau_ket_luan(kq: dict) -> str:
    """Một dòng tiếng người từ kết quả `so_sanh_theo_cap`."""
    a, b, h = kq["a"], kq["b"], kq["hieu"]
    lo, hi = kq["ci_lo"], kq["ci_hi"]
    if kq["ket_luan_duoc"]:
        tot, kem = (a, b) if h > 0 else (b, a)
        return (f"{tot} > {kem}: hieu {abs(h):+.4f}, CI95 cua HIEU "
                f"[{lo}, {hi}] KHONG chua 0 -> ket luan duoc.")
    return (f"{a} vs {b}: hieu {h:+.4f}, CI95 cua HIEU [{lo}, {hi}] CHUA 0 "
            f"-> 'tuong duong' (thang {kq['ty_le_thang']*100:.0f}% lan bootstrap).")


def chenh_lech_co_y_nghia(a: KetQuaDanhGia, b: KetQuaDanhGia) -> str:
    """So hai CI95 **biên** — phép thử BẢO THỦ, chỉ dùng để tham khảo.

    ⚠ ƯU TIÊN `so_sanh_theo_cap()` cho kết luận chính thức. Hàm này so hai
    khoảng tin cậy biên như thể hai mẫu độc lập, trong khi bốn model được đánh
    giá trên **CÙNG một tập ảnh** — tức là so sánh theo cặp. Hai CI biên giao
    nhau **không** có nghĩa là hiệu bằng 0; phần lớn độ rộng của chúng đến từ
    biến thiên do chọn cây, mà cả hai model cùng chịu.

    Giữ lại vì "CI biên giao nhau" vẫn là một phát biểu đúng và dễ hiểu, và vì
    khi nó nói *khác nhau* thì kết luận đó chắc chắn (bảo thủ theo hướng an toàn).

    Quy tắc 13: công bố "EfficientNet thắng" vì hơn 0,8% macro-F1 một lần chạy
    là lỗi học thuật mà không công cụ nào báo. Hàm này báo.
    """
    lo_a, hi_a = a.ci_dai["ci_lo"], a.ci_dai["ci_hi"]
    lo_b, hi_b = b.ci_dai["ci_lo"], b.ci_dai["ci_hi"]
    giao = not (hi_a < lo_b or hi_b < lo_a)
    hieu = a.macro_f1_dai - b.macro_f1_dai
    if giao:
        return (
            f"KHONG ket luan duoc: {a.arch} {a.macro_f1_dai:.4f} vs {b.arch} "
            f"{b.macro_f1_dai:.4f} (chenh {hieu:+.4f}) nhung CI95 GIAO NHAU "
            f"[{lo_a}, {hi_a}] / [{lo_b}, {hi_b}]. Bao cao phai viet 'tuong duong'."
        )
    tot, kem = (a, b) if hieu > 0 else (b, a)
    return (
        f"{tot.arch} > {kem.arch}: CI95 KHONG giao nhau "
        f"({tot.macro_f1_dai:.4f} vs {kem.macro_f1_dai:.4f})."
    )


def holm(p) -> np.ndarray:
    """p-value đã hiệu chỉnh Holm–Bonferroni (kiểm soát FWER), cùng thứ tự đầu vào.

    p_(i) sắp tăng dần được nhân (m − i + 1), lấy cực đại tích luỹ để giữ đơn điệu,
    chặn ở 1. Bác bỏ ở mức α khi p_holm <= α. NaN bị từ chối thay vì bỏ qua: bỏ một
    phép so sánh làm m nhỏ đi và hiệu chỉnh nhẹ hơn thực tế.
    """
    p = np.asarray(p, dtype=np.float64)
    if np.isnan(p).any():
        raise ValueError("holm: co p-value NaN")
    m = len(p)
    thu_tu = np.argsort(p, kind="stable")
    dieu_chinh = np.maximum.accumulate((m - np.arange(m)) * p[thu_tu])
    ra = np.empty(m)
    ra[thu_tu] = np.minimum(dieu_chinh, 1.0)
    return ra
