"""Canh gác: tập dự đoán conformal theo từng dải (`bench/conformal.py`).

Các lỗi im lặng được canh:

* Hiệu chỉnh trên fold khác (model khác) -> độ phủ hụt ở fold có model kém, con
  số độ phủ chung vẫn trông ổn.
* Lượng tử lệch một bậc (⌈(n+1)(1−α)⌉ thành ⌈n(1−α)⌉) -> độ phủ hụt nhẹ.
* File split rò rỉ (cây trải nhiều fold) mà vẫn hiệu chỉnh "trong fold".
* So cỡ tập hai model mà hai file lệch thứ tự dòng hoặc lệch split.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coffeeguard.bench import contract as ct
from coffeeguard.bench import conformal as cf

GOC = Path(__file__).resolve().parent.parent
B = ct.CLASSES_BAND


def _tao_pred(n_cay: int = 200, seed: int = 0, do_tot_fold=None) -> pd.DataFrame:
    """Mỗi cây 2 ảnh, cả cây cùng một fold. Lớp thật được cộng thêm `do_tot`
    vào logit — `do_tot_fold[f]` cho phép mỗi fold là một "model" tốt/kém khác nhau."""
    r = np.random.default_rng(seed)
    hang = []
    for c in range(n_cay):
        fold = c % 5
        tot = 2.0 if do_tot_fold is None else do_tot_fold[fold]
        for k in range(2):
            that = r.integers(0, 6)
            z = r.normal(size=6)
            z[that] += tot
            p = np.exp(z) / np.exp(z).sum()
            hang.append({"png": f"C{c}_{k}.png", "fold": fold, "plant_id": f"C{c}",
                         "leaf_id": f"C{c}_{k}", "y_true": ct.CLASSES_6[that],
                         "y_pred": ct.CLASSES_6[int(p.argmax())],
                         **{f"p_{x}": v for x, v in zip(ct.CLASSES_6, p)}})
    return pd.DataFrame(hang)


def test_luong_tu_dung_bac_va_vo_cuc_khi_thieu_anh() -> None:
    d = np.arange(1, 10) / 10          # 9 điểm
    assert cf._nguong(d, 0.10) == 0.9  # ⌈10·0,9⌉ = 9 -> phần tử thứ 9
    assert cf._nguong(d, 0.20) == 0.8  # ⌈10·0,8⌉ = 8
    assert np.isinf(cf._nguong(d[:8], 0.10))  # ⌈9·0,9⌉ = 9 > 8
    assert np.isinf(cf._nguong(np.array([]), 0.2))


def test_do_phu_moi_dai_dat_muc_tieu() -> None:
    pred = _tao_pred()
    kq = cf.tom_tat(pred, alpha=0.2, so_lan=10, n_bootstrap=50)
    yt = pred["y_true"].map(ct.band_of)
    for b in B:
        if (yt == b).sum() < 30:
            continue
        assert kq[f"phu__{b}"] >= 0.8 - 0.05, (b, kq[f"phu__{b}"])


def test_hieu_chinh_trong_fold_giu_do_phu_o_fold_model_kem() -> None:
    """Fold 0 là model kém hẳn. Hiệu chỉnh chéo fold sẽ hụt độ phủ ở fold 0;
    hiệu chỉnh trong fold thì không."""
    pred = _tao_pred(n_cay=300, do_tot_fold=[0.3, 3.0, 3.0, 3.0, 3.0])
    S, _ = cf.tap_du_doan(pred, alpha=0.2, so_lan=10)
    yt = np.array([B.index(ct.band_of(c)) for c in pred["y_true"]])
    phu = S[:, np.arange(len(pred)), yt].mean(0)
    for f in range(5):
        m = pred["fold"].to_numpy() == f
        assert phu[m].mean() >= 0.8 - 0.05, (f, phu[m].mean())


def test_tu_choi_file_cay_trai_nhieu_fold() -> None:
    pred = _tao_pred(n_cay=50)
    pred.loc[0, "fold"] = (pred.loc[1, "fold"] + 1) % 5   # hai ảnh của C0 ở hai fold
    with pytest.raises(ValueError, match="nhieu fold"):
        cf.tap_du_doan(pred, so_lan=1)


def test_lien_nhau_tren_truc_gi_sat() -> None:
    def tap(*ten):
        s = np.zeros(len(B), dtype=bool)
        for t in ten:
            s[B.index(t)] = True
        return s

    S = np.stack([tap("healthy", "rust_nang"), tap("rust_nhe", "rust_trung_binh"),
                  tap("red_spider_mite", "healthy"), tap(),
                  tap("healthy", "rust_nhe", "rust_nang")])
    assert cf.lien_nhau(S).tolist() == [False, True, True, True, False]


def test_so_sanh_khong_phu_thuoc_thu_tu_dong() -> None:
    a, b = _tao_pred(seed=1), _tao_pred(seed=2)
    b = b.assign(y_true=a["y_true"])
    kq1 = cf.so_sanh_co_tap("a", "b", a, b, so_lan=3, n_bootstrap=50)
    kq2 = cf.so_sanh_co_tap("a", "b", a, b.sample(frac=1, random_state=0), so_lan=3, n_bootstrap=50)
    assert kq1["hieu"] == kq2["hieu"]


def test_so_sanh_tu_choi_hai_split_khac_nhau() -> None:
    a = _tao_pred(seed=1)
    b = a.copy()
    b.loc[b.plant_id == "C0", "fold"] = 4
    with pytest.raises(ValueError, match="split"):
        cf.so_sanh_co_tap("a", "b", a, b, so_lan=1, n_bootstrap=10)


def test_ghim_so_luot_A_mobilevitv2_050() -> None:
    """Số trong `Docs/Khoang_trong_bai_bao.md` §6.4 (α = 0,20)."""
    p = GOC / "reports/predictions/pred__pytorch-timm__mobilevitv2_050__20261022-1539-luotA.csv"
    if not p.exists():
        pytest.skip(f"thiếu {p.name}")
    kq = cf.tom_tat(pd.read_csv(p), alpha=0.2, n_bootstrap=100)
    assert kq["co_tap_tb"] == pytest.approx(1.362, abs=0.005)
    assert kq["ty_le_lien_nhau"] >= 0.99
    assert kq["ty_le_rong"] == pytest.approx(0.069, abs=0.005)
    for b in B:
        assert kq[f"phu__{b}"] >= 0.80


def test_vo_cuc_dai_dem_dung_dai_thieu_anh_va_khop_ty_le_chung() -> None:
    """Dải hiếm không đủ ảnh hiệu chỉnh phải hiện ra là +∞ ở CHÍNH dải đó; tổng
    theo dải phải khớp tỉ lệ chung — nếu không, câu "80/84 ngưỡng +∞ là của dải
    nặng" trong bài không có cơ sở."""
    pred = _tao_pred(n_cay=200)
    # dải nặng chỉ còn 6 ảnh -> mỗi nửa fold < 9 ảnh -> ở α = 0,10 luôn +∞
    nang = pred["y_true"].map(ct.band_of) == "rust_nang"
    bo = pred.index[nang][6:]
    pred = pred.drop(bo).reset_index(drop=True)
    S, tt = cf.tap_du_doan(pred, alpha=0.10, so_lan=3)
    vc = tt["vo_cuc_dai"]
    so_nguong = 3 * 10 * len(B)
    assert vc.sum() / so_nguong == pytest.approx(tt["ty_le_nguong_vo_cuc"], abs=1e-4)
    assert vc[B.index("rust_nang")] == 3 * 10
    _, tt_b = cf.tap_du_doan(pred, alpha=0.10, so_lan=3, che_do="bien")
    assert tt_b["vo_cuc_dai"].sum() == 0


def test_so_sanh_co_tap_p_bootstrap_khop_ci() -> None:
    """CI không chứa 0 thì p < 0,05 và ngược lại (cùng các mẫu bootstrap)."""
    pred = _tao_pred()
    kem = _tao_pred(do_tot_fold=[0.5] * 5)
    kem[["png", "fold", "plant_id", "y_true"]] = pred[["png", "fold", "plant_id", "y_true"]]
    kq = cf.so_sanh_co_tap("tot", "kem", pred, kem, so_lan=3, n_bootstrap=200)
    assert kq["ket_luan_duoc"] and kq["p_bootstrap"] < 0.05
    kq0 = cf.so_sanh_co_tap("a", "a", pred, pred.copy(), so_lan=3, n_bootstrap=200)
    assert kq0["hieu"] == 0 and kq0["p_bootstrap"] == 1.0
