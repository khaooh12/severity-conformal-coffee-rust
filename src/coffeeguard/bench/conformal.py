"""Tập dự đoán conformal theo từng dải — thay "không kết luận" bằng "một tập dải".

`Docs/Khoang_trong_bai_bao.md` §6.4 (G9). Thay vì im lặng khi không chắc, app trả
một TẬP dải (vd "gỉ sắt trung bình hoặc nặng"), với bảo đảm MỖI dải thật được phủ
>= 1 − α (conformal Mondrian: hiệu chỉnh riêng theo nhãn thật). Điểm không phù
hợp = 1 − xác suất của dải (`tu_choi.xac_suat_dai`).

Các bẫy được chặn ở đây:

* **Hiệu chỉnh trên fold khác là SAI.** Mỗi fold là một model khác; ngưỡng học
  trên xác suất của model fold 2 không nói gì về model fold 3. Bảo đảm độ phủ chỉ
  đúng khi tập hiệu chỉnh và tập chấm do CÙNG model dự đoán. Vì vậy hiệu chỉnh
  luôn diễn ra BÊN TRONG một fold: chia cây của fold làm hai nửa, hiệu chỉnh trên
  nửa này, chấm nửa kia, rồi đổi vai. (Bản khám phá đầu tiên làm chéo fold —
  `scripts/kham_pha/lech_dai_nang.py`, phần b, đã xoá — xem lịch sử git — và cho số sai.)
* **Chia nửa theo ẢNH thì rò rỉ.** Hai mặt của cùng lá, hai lá của cùng cây rơi
  vào hai nửa -> ngưỡng được học trên "bản sao" của ảnh đang chấm. Chia theo cây.
* **Một lần chia nửa là một con số ngẫu nhiên.** Lặp `so_lan` lần với seed khác
  nhau rồi lấy trung bình; CI bootstrap theo cây lấy trên đại lượng THEO ẢNH đã
  trung bình qua các lần lặp.
* **Tập hiệu chỉnh quá nhỏ thì tập phình.** Với n ảnh hiệu chỉnh của một dải, nếu
  ⌈(n+1)(1−α)⌉ > n thì ngưỡng là +∞ (dải đó luôn vào tập) — đúng lý thuyết, nhưng
  làm tập vô dụng. RoCoLe chỉ có 18–19 ảnh dải nặng mỗi fold, ≈ 9 mỗi nửa: α = 0,10
  đã sát giới hạn này. `tom_tat` trả `n_hieu_chinh_min` và `ty_le_nguong_vo_cuc`
  để thấy ngay.
* **Độ phủ chung che độ phủ từng dải** — cùng bẫy như `tu_choi.duong_do_phu`.
  Bảng luôn tách theo dải thật.
* **Trung bình qua các lần chia che dao động của từng lần.** Bảo đảm conformal là
  trung bình trên cách chọn tập hiệu chỉnh; với ≈ 9 ảnh dải nặng mỗi nửa fold, một
  lần chia cụ thể có thể hụt xa mục tiêu. `tom_tat` trả kèm độ lệch chuẩn, min,
  max theo lần chia và tỉ lệ nửa fold dưới mục tiêu.

Đối chứng (`diem`, `che_do`, `hieu_chinh` của `tap_du_doan`) — để phản biện thấy
kết luận không phụ thuộc lựa chọn của bài:

* `diem`: `"thr"` (1 − p, mặc định), `"aps"` [Romano 2020, bản không ngẫu nhiên],
  `"raps"` [Angelopoulos 2021; λ = `RAPS_LAMBDA`, k = `RAPS_K` — ASSUMED, không
  tinh chỉnh], `"ord"` (APS thứ bậc của Lu 2022: tập là đoạn liền nhau trên
  `TRUC_THU_BAC`, mọc tham lam từ dải có xác suất lớn nhất).
* `che_do`: `"theo_dai"` (Mondrian) hoặc `"bien"` (một ngưỡng chung).
* `hieu_chinh`: `"trong_fold"` (nửa cây, đúng lý thuyết) hoặc `"cheo_fold"`
  (hiệu chỉnh trên 4 fold kia — cùng sơ đồ với `tu_choi.nhiet_do_cheo_fold`; chỉ
  dùng làm thí nghiệm kiểm chứng xem sơ đồ hiệu chỉnh có đổi kết luận không).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import metrics as bm
from coffeeguard.bench import tu_choi as tc

#: Vị trí trên trục gỉ sắt của các dải; nhện đỏ không nằm trên trục.
TRUC_GI_SAT: dict[str, int] = {"healthy": 0, "rust_nhe": 1, "rust_trung_binh": 2, "rust_nang": 3}

#: Thứ tự dải cho conformal thứ bậc. Nhện đỏ không nằm trên trục gỉ sắt; đặt nó ở
#: đầu, cạnh khoẻ, là GIẢ ĐỊNH của bài (không có thứ tự tự nhiên nào cho nó).
TRUC_THU_BAC: tuple[str, ...] = ("red_spider_mite", "healthy", "rust_nhe",
                                 "rust_trung_binh", "rust_nang")
CACH_DIEM: tuple[str, ...] = ("thr", "aps", "raps", "ord")
CHE_DO: tuple[str, ...] = ("theo_dai", "bien")
HIEU_CHINH: tuple[str, ...] = ("trong_fold", "cheo_fold")
RAPS_LAMBDA, RAPS_K = 0.1, 1


def _nguong(diem: np.ndarray, alpha: float) -> float:
    """Lượng tử conformal: phần tử thứ ⌈(n+1)(1−α)⌉ của điểm đã sắp; +∞ nếu thiếu ảnh."""
    n = len(diem)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    if n == 0 or k > n:
        return float("inf")
    return float(np.sort(diem)[k - 1])


def _kiem_cay_mot_fold(pred: pd.DataFrame, group_col: str) -> None:
    so_fold = pred.groupby(group_col)["fold"].nunique()
    sai = so_fold[so_fold > 1]
    if len(sai):
        raise ValueError(
            f"{len(sai)} cay co anh o nhieu fold (vd {sai.index[0]}) — file du doan "
            "khong phai chia theo cay; hieu chinh trong fold se ro ri."
        )


def diem_khong_phu_hop(Q: np.ndarray, diem: str = "thr") -> np.ndarray:
    """Điểm không phù hợp s(x, y) cho MỌI dải y, dạng (n, 5). Tập = {y : s <= q}."""
    n, K = Q.shape
    hang = np.arange(n)[:, None]
    if diem == "thr":
        return 1 - Q
    if diem in ("aps", "raps"):
        thu_tu = np.argsort(-Q, axis=1, kind="stable")
        s = np.empty_like(Q)
        s[hang, thu_tu] = np.cumsum(Q[hang, thu_tu], axis=1)
        if diem == "raps":
            hang_y = np.empty((n, K))
            hang_y[hang, thu_tu] = np.arange(1, K + 1)
            s = s + RAPS_LAMBDA * np.maximum(0, hang_y - RAPS_K)
        return s
    if diem == "ord":
        vi_tri = [ct.CLASSES_BAND.index(b) for b in TRUC_THU_BAC]
        Qt = Q[:, vi_tri]
        st = np.empty_like(Qt)
        for i in range(n):
            lo = hi = int(Qt[i].argmax())
            tong = Qt[i, lo]
            st[i, lo] = tong
            while lo > 0 or hi < K - 1:
                trai = Qt[i, lo - 1] if lo > 0 else -1.0
                phai = Qt[i, hi + 1] if hi < K - 1 else -1.0
                if trai >= phai:
                    lo -= 1
                    tong += Qt[i, lo]
                    st[i, lo] = tong
                else:
                    hi += 1
                    tong += Qt[i, hi]
                    st[i, hi] = tong
        s = np.empty_like(Q)
        s[:, vi_tri] = st
        return s
    raise ValueError(f"diem phai thuoc {CACH_DIEM}, nhan {diem!r}")


def chia_nua_cay(
    pred: pd.DataFrame,
    so_lan: int = 20,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """`so_lan` lần chia; mỗi lần là danh sách cặp (mặt nạ hiệu chỉnh, mặt nạ chấm),
    hai cặp mỗi fold (chia cây của fold làm hai nửa rồi đổi vai). Dùng chung cho
    conformal và `hieu_chinh.py` để mọi phương pháp hiệu chỉnh trên CÙNG các nửa."""
    fold = pred["fold"].to_numpy()
    cay = pred[group_col].astype(str).to_numpy()
    ra = []
    for r in range(so_lan):
        rng = np.random.default_rng(seed + r)
        cap = []
        for f in np.unique(fold):
            trong = fold == f
            ds_cay = np.unique(cay[trong])
            rng.shuffle(ds_cay)
            nua_a = trong & np.isin(cay, ds_cay[: len(ds_cay) // 2])
            nua_b = trong & ~nua_a
            cap += [(nua_a, nua_b), (nua_b, nua_a)]
        ra.append(cap)
    return ra


def chia_cheo_fold(pred: pd.DataFrame) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    """Một "lần chia": hiệu chỉnh trên 4 fold kia, chấm fold còn lại (model khác!)."""
    fold = pred["fold"].to_numpy()
    return [[(fold != f, fold == f) for f in np.unique(fold)]]


def cac_lan_chia(
    pred: pd.DataFrame,
    hieu_chinh: str,
    so_lan: int = 20,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    if hieu_chinh == "trong_fold":
        _kiem_cay_mot_fold(pred, group_col)
        return chia_nua_cay(pred, so_lan, seed, group_col)
    if hieu_chinh == "cheo_fold":
        return chia_cheo_fold(pred)
    raise ValueError(f"hieu_chinh phai thuoc {HIEU_CHINH}, nhan {hieu_chinh!r}")


def tap_du_doan(
    pred: pd.DataFrame,
    alpha: float = 0.20,
    so_lan: int = 20,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
    diem: str = "thr",
    che_do: str = "theo_dai",
    hieu_chinh: str = "trong_fold",
) -> tuple[np.ndarray, dict]:
    """Tập dự đoán cho từng ảnh, lặp `so_lan` lần chia nửa.

    Trả `(S, thong_tin)`: `S` có dạng (so_lan, n, 5), bool, theo thứ tự
    `CLASSES_BAND` (so_lan = 1 khi `hieu_chinh="cheo_fold"`); `thong_tin` gồm
    `n_hieu_chinh_min` (số ảnh hiệu chỉnh ít nhất của một ngưỡng),
    `ty_le_nguong_vo_cuc` (tỉ lệ ngưỡng = +∞ trên mọi nửa × dải × lần lặp) và
    `phu_nua` (lần × nửa fold × dải: độ phủ trên nửa được chấm, NaN nếu nửa đó
    không có ảnh của dải) và `vo_cuc_dai` (số ngưỡng = +∞ của từng dải, cộng qua
    mọi nửa × lần lặp; toàn 0 ở chế độ biên vì ngưỡng không gắn với dải nào).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha phai trong (0, 1), nhan {alpha}")
    if che_do not in CHE_DO:
        raise ValueError(f"che_do phai thuoc {CHE_DO}, nhan {che_do!r}")
    cac_lan = cac_lan_chia(pred, hieu_chinh, so_lan, seed, group_col)
    Q = tc.xac_suat_dai(pred)
    s = diem_khong_phu_hop(Q, diem)
    yt = bm._ma_hoa([ct.band_of(c) for c in pred["y_true"]], ct.CLASSES_BAND)
    K = len(ct.CLASSES_BAND)

    S = np.zeros((len(cac_lan), len(pred), K), dtype=bool)
    phu_nua = np.full((len(cac_lan), len(cac_lan[0]), K), np.nan)
    n_min, so_vo_cuc, so_nguong = np.inf, 0, 0
    vo_cuc_dai = np.zeros(K, dtype=int)
    for r, cap in enumerate(cac_lan):
        for j, (hc, cham) in enumerate(cap):
            if che_do == "bien":
                d = s[np.flatnonzero(hc), yt[hc]]
                n_min = min(n_min, len(d))
                q = _nguong(d, alpha)
                so_nguong += 1
                so_vo_cuc += int(np.isinf(q))
                S[r, cham] = s[cham] <= q
            else:
                for c in range(K):
                    d = s[hc & (yt == c), c]
                    n_min = min(n_min, len(d))
                    q = _nguong(d, alpha)
                    so_nguong += 1
                    so_vo_cuc += int(np.isinf(q))
                    vo_cuc_dai[c] += int(np.isinf(q))
                    S[r, cham, c] = s[cham, c] <= q
            for c in range(K):
                m = cham & (yt == c)
                if m.any():
                    phu_nua[r, j, c] = S[r, m, c].mean()
    return S, {"n_hieu_chinh_min": int(n_min),
               "ty_le_nguong_vo_cuc": round(so_vo_cuc / so_nguong, 4),
               "phu_nua": phu_nua,
               "vo_cuc_dai": vo_cuc_dai}


def lien_nhau(S: np.ndarray) -> np.ndarray:
    """True nếu các dải gỉ sắt (khoẻ, nhẹ, TB, nặng) trong tập liền nhau trên trục.

    Nhện đỏ bỏ qua (không nằm trên trục). Tập "khoẻ hoặc nặng" là không liền nhau —
    nông dân không biết làm gì với nó. Phép đo tự đặt, chưa đối chiếu tài liệu.
    """
    vi_tri = np.array([TRUC_GI_SAT.get(b, -1) for b in ct.CLASSES_BAND])
    tren_truc = vi_tri >= 0
    T = S[..., tren_truc]
    pos = vi_tri[tren_truc]
    so = T.sum(-1)
    lo = np.where(T, pos, 99).min(-1)
    hi = np.where(T, pos, -1).max(-1)
    return (so <= 1) | (hi - lo + 1 == so)


def _theo_anh(S: np.ndarray, yt: np.ndarray) -> dict[str, np.ndarray]:
    """Đại lượng theo ảnh, đã trung bình qua các lần lặp."""
    co = S.sum(-1)
    mot = co == 1
    dung_mot = mot & S[:, np.arange(S.shape[1]), yt]
    return {
        "co_tap": co.mean(0),
        "phu": S[:, np.arange(S.shape[1]), yt].mean(0),
        "mot_dai": mot.mean(0),
        "dung_mot_dai": dung_mot.mean(0),
        "lien_nhau": lien_nhau(S).mean(0),
        "rong": (co == 0).mean(0),
    }


def tom_tat(
    pred: pd.DataFrame,
    alpha: float = 0.20,
    so_lan: int = 20,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
    diem: str = "thr",
    che_do: str = "theo_dai",
    hieu_chinh: str = "trong_fold",
) -> dict:
    """Một dòng tóm tắt cho một model.

    `phu__<dải>`: tỉ lệ ảnh có dải thật đó được tập chứa (mục tiêu >= 1 − α).
    `co_tap_tb` (+CI theo cây): số dải trung bình trong tập (1 = chắc chắn, 5 = vô
    dụng). `ty_le_mot_dai`, `acc_khi_mot_dai`: khi tập chỉ có một dải thì đúng bao
    nhiêu. `ty_le_lien_nhau`: xem `lien_nhau`. `co_tap__<dải>`: cỡ tập theo dải thật.

    Phân tán (bảo đảm chỉ đúng TRUNG BÌNH): `phu_lan_sd/min/max__<dải>` — độ phủ
    của dải qua từng lần chia; `phu_nua_min__<dải>`, `nua_duoi_muc__<dải>` — độ phủ
    thấp nhất và tỉ lệ nửa fold có độ phủ < 1 − α (≈ 9 ảnh dải nặng mỗi nửa).
    """
    S, tt = tap_du_doan(pred, alpha, so_lan, seed, group_col, diem, che_do, hieu_chinh)
    phu_nua = tt.pop("phu_nua")
    tt.pop("vo_cuc_dai")
    yt = bm._ma_hoa([ct.band_of(c) for c in pred["y_true"]], ct.CLASSES_BAND)
    a = _theo_anh(S, yt)
    lo, hi = tc._bootstrap_cum(pred[group_col], lambda idx: float(a["co_tap"][idx].mean()),
                               n_bootstrap, seed, muc)
    mot = a["mot_dai"].sum()
    dong = {
        "diem": diem, "che_do": che_do, "hieu_chinh": hieu_chinh,
        "alpha": alpha, "so_lan": int(S.shape[0]), "n_anh": int(len(pred)),
        "co_tap_tb": round(float(a["co_tap"].mean()), 4),
        "co_tap_ci_lo": round(lo, 4), "co_tap_ci_hi": round(hi, 4),
        "ty_le_mot_dai": round(float(a["mot_dai"].mean()), 4),
        "acc_khi_mot_dai": round(float(a["dung_mot_dai"].sum() / mot), 4) if mot else float("nan"),
        "ty_le_lien_nhau": round(float(a["lien_nhau"].mean()), 4),
        "ty_le_rong": round(float(a["rong"].mean()), 4),
        **tt,
    }
    for c, b in enumerate(ct.CLASSES_BAND):
        m = yt == c
        dong[f"phu__{b}"] = round(float(a["phu"][m].mean()), 4) if m.any() else float("nan")
    for c, b in enumerate(ct.CLASSES_BAND):
        m = yt == c
        dong[f"co_tap__{b}"] = round(float(a["co_tap"][m].mean()), 4) if m.any() else float("nan")
    for c, b in enumerate(ct.CLASSES_BAND):
        m = yt == c
        if not m.any():
            continue
        lan = S[:, m, c].mean(1)
        nua = phu_nua[..., c][~np.isnan(phu_nua[..., c])]
        dong[f"phu_lan_sd__{b}"] = round(float(lan.std(ddof=1)), 4) if len(lan) > 1 else 0.0
        dong[f"phu_lan_min__{b}"] = round(float(lan.min()), 4)
        dong[f"phu_lan_max__{b}"] = round(float(lan.max()), 4)
        dong[f"phu_nua_min__{b}"] = round(float(nua.min()), 4)
        dong[f"nua_duoi_muc__{b}"] = round(float((nua < 1 - alpha - 1e-12).mean()), 4)
    return dong


def so_sanh_co_tap(
    ten_a: str,
    ten_b: str,
    pred_a: pd.DataFrame,
    pred_b: pd.DataFrame,
    alpha: float = 0.20,
    so_lan: int = 20,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
) -> dict:
    """Hiệu cỡ tập TB (A − B) theo cặp trên cùng ảnh, CI95 bootstrap theo cây.

    Âm = A cho tập hẹp hơn (tốt hơn, khi cả hai cùng đạt độ phủ). Căn theo `png`;
    hai file phải cùng fold cho từng ảnh, nếu không chúng không cùng một split và
    nửa hiệu chỉnh của hai model sẽ khác nhau.
    """
    a = pred_a.sort_values("png").reset_index(drop=True)
    b = pred_b.sort_values("png").reset_index(drop=True)
    if len(a) != len(b) or not (a["png"].to_numpy() == b["png"].to_numpy()).all():
        raise ValueError(f"{ten_a} va {ten_b} khong cung tap anh: {len(a)} / {len(b)}")
    if not (a["fold"].to_numpy() == b["fold"].to_numpy()).all():
        raise ValueError(f"{ten_a} va {ten_b} khong cung file split (fold lech)")
    if not (a["y_true"].to_numpy() == b["y_true"].to_numpy()).all():
        raise ValueError(f"{ten_a} va {ten_b} co y_true khac nhau tren cung anh")
    yt = bm._ma_hoa([ct.band_of(c) for c in a["y_true"]], ct.CLASSES_BAND)
    ca = _theo_anh(tap_du_doan(a, alpha, so_lan, seed, group_col)[0], yt)["co_tap"]
    cb = _theo_anh(tap_du_doan(b, alpha, so_lan, seed, group_col)[0], yt)["co_tap"]
    h = ca - cb
    lo, hi = tc._bootstrap_cum(a[group_col], lambda idx: float(h[idx].mean()),
                               n_bootstrap, seed, muc)
    mau = tc._mau_bootstrap_cum(a[group_col], lambda idx: float(h[idx].mean()),
                                n_bootstrap, seed)
    return {"a": ten_a, "b": ten_b, "alpha": alpha,
            "co_tap_a": round(float(ca.mean()), 4), "co_tap_b": round(float(cb.mean()), 4),
            "hieu": round(float(h.mean()), 4), "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
            "ket_luan_duoc": bool(lo > 0 or hi < 0),
            "p_bootstrap": tc.p_bootstrap(mau),
            "don_vi_lay_mau": "cum (cay), THEO CAP"}
