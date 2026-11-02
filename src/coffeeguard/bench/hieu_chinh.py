"""Hiệu chỉnh xác suất và ngưỡng theo dải với CÙNG sơ đồ hiệu chỉnh như conformal.

Phản biện của bản nháp G2 + G9 (31/10/2026): temperature scaling và ngưỡng theo
dải (`tu_choi.py`) học trên 4 fold KHÁC, còn conformal (`conformal.py`) hiệu chỉnh
trong fold, trên nửa cây của CHÍNH model đó. So hai nhóm này là so hai sơ đồ hiệu
chỉnh khác nhau. Module này chạy mọi phương pháp trên đúng các cặp (nửa hiệu chỉnh,
nửa chấm) của `conformal.chia_nua_cay` — cùng hạt giống, cùng cây — và vẫn cho chạy
sơ đồ chéo fold (`hieu_chinh="cheo_fold"`) để thấy sơ đồ có đổi kết luận không.

Ba cách hiệu chỉnh xác suất, đều học trên log xác suất 6 lớp (file dự đoán không có
logit; log p = logit − hằng số mỗi dòng):

* `nhiet_do` — chia cho một T (Guo 2017). Không đổi argmax.
* `vector` — z' = w ⊙ log p + b (12 tham số). Trên log xác suất, đây là Dirichlet
  calibration dạng chéo (Kull 2019); KHÔNG trùng hẳn vector scaling trên logit vì
  hằng số mỗi dòng nhân với w không còn là hằng số. Có thể đổi argmax.
* `dirichlet` — z' = W log p + b, W đầy đủ 6 × 6 (42 tham số), phạt kiểu ODIR
  (phần ngoài đường chéo và b). Có thể đổi argmax.

Các bẫy được chặn ở đây:

* **Tập hiệu chỉnh nhỏ thì tham số bay.** Nửa fold có ≈ 155 ảnh, cấp 4 có khi
  0–3 ảnh; không phạt thì b của cấp 4 trôi về −∞. Hệ số phạt `LAMBDA` là ASSUMED,
  không tinh chỉnh (Kull 2019 chọn bằng kiểm định chéo) — báo cáo phải ghi.
* **Đổi argmax thì đổi tập "ảnh được báo nặng".** Độ lệch sau hiệu chỉnh tính trên
  dự đoán MỚI, mỗi lần chia một tập; lấy trung bình độ lệch qua các lần chia.
* **Ngưỡng theo dải trên nửa fold** chỉ có ≈ 8–10 ảnh được báo nặng để học; với
  `n_toi_thieu = 10` thì gần như luôn +∞. Bảng trả `ty_le_nguong_vo_cuc` để thấy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from coffeeguard.bench import conformal as cf
from coffeeguard.bench import contract as ct
from coffeeguard.bench import metrics as bm
from coffeeguard.bench import tu_choi as tc

PHUONG_PHAP: tuple[str, ...] = ("nhiet_do", "vector", "dirichlet")
LAMBDA: float = 1e-2

#: Ma trận (6, 5) cộng xác suất 6 lớp vào dải.
_SANG_DAI = np.zeros((len(ct.CLASSES_6), len(ct.CLASSES_BAND)))
for _i, _c in enumerate(ct.CLASSES_6):
    _SANG_DAI[_i, ct.CLASSES_BAND.index(ct.band_of(_c))] = 1.0
_NANG = ct.CLASSES_BAND.index("rust_nang")
_DAI_CUA_LOP = _SANG_DAI.argmax(1)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _hoc(phuong_phap: str, logp: np.ndarray, y: np.ndarray) -> dict:
    """Tham số hiệu chỉnh trên (logp, y). Khởi tạo tại phép đồng nhất."""
    if phuong_phap == "nhiet_do":
        return {"T": tc._hoc_nhiet_do(logp, y, tc.LUOI_NHIET_DO)}
    n, K = logp.shape
    Y = np.eye(K)[y]
    if phuong_phap == "vector":
        def f(t):
            w, b = t[:K], t[K:]
            P = _softmax(logp * w + b)
            dz = (P - Y) / n
            nll = -np.log(P[np.arange(n), y] + 1e-12).mean()
            phat = LAMBDA * (((w - 1) ** 2).sum() + (b ** 2).sum()) / K
            g = np.concatenate([(dz * logp).sum(0) + 2 * LAMBDA * (w - 1) / K,
                                dz.sum(0) + 2 * LAMBDA * b / K])
            return nll + phat, g
        t0 = np.concatenate([np.ones(K), np.zeros(K)])
        t = minimize(f, t0, jac=True, method="L-BFGS-B").x
        return {"w": t[:K], "b": t[K:]}
    if phuong_phap == "dirichlet":
        ngoai = ~np.eye(K, dtype=bool)

        def f(t):
            W, b = t[: K * K].reshape(K, K), t[K * K:]
            P = _softmax(logp @ W.T + b)
            dz = (P - Y) / n
            nll = -np.log(P[np.arange(n), y] + 1e-12).mean()
            phat = LAMBDA * ((W[ngoai] ** 2).sum() / (K * (K - 1)) + (b ** 2).sum() / K)
            gW = dz.T @ logp + 2 * LAMBDA * np.where(ngoai, W, 0) / (K * (K - 1))
            return nll + phat, np.concatenate([gW.ravel(), dz.sum(0) + 2 * LAMBDA * b / K])
        t0 = np.concatenate([np.eye(K).ravel(), np.zeros(K)])
        t = minimize(f, t0, jac=True, method="L-BFGS-B").x
        return {"W": t[: K * K].reshape(K, K), "b": t[K * K:]}
    raise ValueError(f"phuong_phap phai thuoc {PHUONG_PHAP}, nhan {phuong_phap!r}")


def _ap(phuong_phap: str, th: dict, logp: np.ndarray) -> np.ndarray:
    if phuong_phap == "nhiet_do":
        return tc._chia_nhiet_do(logp, th["T"])
    if phuong_phap == "vector":
        return _softmax(logp * th["w"] + th["b"])
    return _softmax(logp @ th["W"].T + th["b"])


def xac_suat_sau(
    pred: pd.DataFrame,
    phuong_phap: str,
    hieu_chinh: str = "trong_fold",
    so_lan: int = 20,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> tuple[list[np.ndarray], list[dict]]:
    """Xác suất 6 lớp sau hiệu chỉnh, một ma trận (n, 6) mỗi lần chia, và danh
    sách tham số đã học (một mỗi nửa / fold)."""
    logp = np.log(np.clip(tc.ma_tran_xac_suat(pred), 1e-12, 1.0))
    y = bm._ma_hoa(pred["y_true"], ct.CLASSES_6)
    ds_P, ds_th = [], []
    for cap in cf.cac_lan_chia(pred, hieu_chinh, so_lan, seed, group_col):
        P2 = np.full_like(logp, np.nan)
        for hc, cham in cap:
            th = _hoc(phuong_phap, logp[hc], y[hc])
            P2[cham] = _ap(phuong_phap, th, logp[cham])
            ds_th.append(th)
        ds_P.append(P2)
    return ds_P, ds_th


def _do(P: np.ndarray, yt6: np.ndarray, ytd: np.ndarray):
    """(tin cậy theo dải của dải dự đoán, đúng theo dải, có báo nặng, ECE 6 lớp, macro-F1 dải)."""
    yp6 = P.argmax(1)
    ypd = _DAI_CUA_LOP[yp6]
    Q = P @ _SANG_DAI
    e, _ = tc._ece_mce(P.max(1), yp6 == yt6, 10, "deu_do_rong")
    f1 = bm._macro_f1_ma(ytd, ypd, len(ct.CLASSES_BAND))
    return Q[np.arange(len(P)), ypd], ypd == ytd, ypd == _NANG, e, f1


def danh_gia(
    pred: pd.DataFrame,
    phuong_phap: str,
    hieu_chinh: str = "trong_fold",
    so_lan: int = 20,
    n_bootstrap: int = 2000,
    seed: int = ct.SEED,
    muc: float = 0.95,
    group_col: str = ct.GROUP_COL,
) -> dict:
    """ECE 6 lớp và độ lệch khi báo "gỉ sắt nặng" (định nghĩa dải), trước/sau.

    `lech_nang_sau` = trung bình qua các lần chia của (tin cậy TB − accuracy) trên
    các ảnh được báo nặng ở lần đó; CI95 bootstrap theo cây trên cùng các lần chia.
    `ece_sau_min/max`: phạm vi qua các lần chia.
    """
    ds_P, ds_th = xac_suat_sau(pred, phuong_phap, hieu_chinh, so_lan, seed, group_col)
    yt6 = bm._ma_hoa(pred["y_true"], ct.CLASSES_6)
    ytd = _DAI_CUA_LOP[yt6]
    truoc = _do(tc.ma_tran_xac_suat(pred), yt6, ytd)
    sau = [_do(P, yt6, ytd) for P in ds_P]
    groups = pred[group_col].to_numpy()
    that_nang = ytd == _NANG

    def _ci(C, D, M):
        def _lech(idx):
            m = M[:, idx]
            so = m.sum(1)
            ok = so > 0
            if not ok.any():
                return float("nan")
            l = ((C[:, idx] * m).sum(1) - (D[:, idx] * m).sum(1))[ok] / so[ok]
            return float(l.mean())
        lo, hi = tc._bootstrap_cum(groups, _lech, n_bootstrap, seed, muc)
        return round(_lech(np.arange(len(pred))), 4), round(lo, 4), round(hi, 4)

    l0 = _ci(truoc[0][None], truoc[1][None], truoc[2][None])
    l1 = _ci(np.stack([s[0] for s in sau]), np.stack([s[1] for s in sau]),
             np.stack([s[2] for s in sau]))
    ece_sau = np.array([s[3] for s in sau])
    kq = {
        "phuong_phap": phuong_phap, "hieu_chinh": hieu_chinh, "so_lan": len(ds_P),
        "ece_truoc": round(truoc[3], 4),
        "ece_sau": round(float(ece_sau.mean()), 4),
        "ece_sau_min": round(float(ece_sau.min()), 4),
        "ece_sau_max": round(float(ece_sau.max()), 4),
        "macro_f1_dai_truoc": round(truoc[4], 4),
        "macro_f1_dai_sau": round(float(np.mean([s[4] for s in sau])), 4),
        "n_bao_nang_truoc": int(truoc[2].sum()),
        "n_bao_nang_sau": round(float(np.mean([s[2].sum() for s in sau])), 1),
        # tỉ lệ lá nặng thật được báo nặng: lệch về 0 nhờ báo nặng ít đi thì cột này tụt
        "recall_nang_truoc": round(float(truoc[2][that_nang].mean()), 4),
        "recall_nang_sau": round(float(np.mean([s[2][that_nang].mean() for s in sau])), 4),
        "lech_nang_truoc": l0[0], "lech_nang_truoc_ci_lo": l0[1], "lech_nang_truoc_ci_hi": l0[2],
        "lech_nang_sau": l1[0], "lech_nang_sau_ci_lo": l1[1], "lech_nang_sau_ci_hi": l1[2],
    }
    if phuong_phap == "nhiet_do":
        T = np.array([t["T"] for t in ds_th])
        kq.update({"T_min": round(float(T.min()), 4), "T_max": round(float(T.max()), 4),
                   "ty_le_T_duoi_1": round(float((T < 1).mean()), 4)})
    return kq


def _chon_nguong(tin_cay: np.ndarray, dung: np.ndarray, muc_tieu: float, n_toi_thieu: int) -> float:
    """Như `tu_choi.nguong_theo_dai`: ngưỡng nhỏ nhất trên lưới đạt accuracy mục tiêu."""
    for t in tc.LUOI_NGUONG:
        s = tin_cay >= t
        if s.sum() >= n_toi_thieu and dung[s].mean() >= muc_tieu:
            return float(t)
    return float("inf")


def nguong_theo_dai(
    pred: pd.DataFrame,
    hieu_chinh: str = "trong_fold",
    muc_tieu: float = 0.85,
    n_toi_thieu: int = 10,
    so_lan: int = 20,
    seed: int = ct.SEED,
    group_col: str = ct.GROUP_COL,
) -> pd.DataFrame:
    """Ngưỡng theo dải DỰ ĐOÁN so với ngưỡng chung ở CÙNG độ phủ, trên cùng sơ đồ chia.

    Như `tu_choi.so_nguong_cung_do_phu` (định nghĩa dải), nhưng ngưỡng học trên nửa
    hiệu chỉnh và áp lên nửa chấm của CÙNG fold. Ngưỡng chung so sánh được chọn mỗi
    lần chia để trả lời đúng số ảnh mà ngưỡng theo dải trả lời. Mọi số là trung bình
    qua các lần chia. `hieu_chinh="cheo_fold"` cho lại đúng số của `tu_choi`.
    """
    tin_cay, dung = tc._tin_cay_va_dung_theo(pred, "dai")
    dai = np.asarray([ct.band_of(c) for c in pred["y_pred"]])
    yt = np.asarray([ct.band_of(c) for c in pred["y_true"]])
    cac_lan = cf.cac_lan_chia(pred, hieu_chinh, so_lan, seed, group_col)
    so_vo_cuc, so_nguong = 0, 0
    tich = {"chung": [], "theo_dai": []}
    for cap in cac_lan:
        tra_dai = np.zeros(len(pred), dtype=bool)
        for hc, cham in cap:
            for b in ct.CLASSES_BAND:
                hoc = hc & (dai == b)
                t = _chon_nguong(tin_cay[hoc], dung[hoc], muc_tieu, n_toi_thieu)
                so_nguong += 1
                so_vo_cuc += int(np.isinf(t))
                ap = cham & (dai == b)
                tra_dai[ap] = tin_cay[ap] >= t
        k = int(tra_dai.sum())
        tra_chung = np.zeros(len(pred), dtype=bool)
        if k:
            tra_chung[np.argsort(-tin_cay, kind="stable")[:k]] = True
        for ten, m in (("chung", tra_chung), ("theo_dai", tra_dai)):
            dong = {"do_phu": m.mean(), "accuracy": dung[m].mean() if m.any() else np.nan}
            for b in ct.CLASSES_BAND:
                dong[f"do_phu__{b}"] = m[yt == b].mean()
            tich[ten].append(dong)
    hang = []
    for ten in ("chung", "theo_dai"):
        tb = pd.DataFrame(tich[ten]).mean()
        hang.append({"cach": ten, "hieu_chinh": hieu_chinh, "muc_tieu": muc_tieu,
                     "n_toi_thieu": n_toi_thieu, "so_lan": len(cac_lan),
                     "ty_le_nguong_vo_cuc": round(so_vo_cuc / so_nguong, 4),
                     **{c: round(float(v), 4) for c, v in tb.items()}})
    return pd.DataFrame(hang)
