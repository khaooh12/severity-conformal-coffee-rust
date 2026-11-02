"""Canh gác: hiệu chỉnh trên CÙNG sơ đồ chia với conformal (`bench/hieu_chinh.py`),
và các đối chứng conformal (APS / RAPS / thứ bậc, biên / theo dải, trong / chéo fold).

Các lỗi im lặng được canh:

* Hai nhóm phương pháp hiệu chỉnh trên hai sơ đồ khác nhau mà bảng vẫn đặt cạnh
  nhau -> so không công bằng (phản biện bản nháp G2 + G9, 31/10/2026).
* Nửa hiệu chỉnh và nửa chấm chung cây, hoặc khác fold -> rò rỉ / sai model.
* Bản "chéo fold" viết lại lệch khỏi `tu_choi` -> hai bảng cùng tên khác số.
* Conformal thứ bậc trả tập không liền nhau -> mất đúng tính chất nó được chọn.
* Chỉ báo độ phủ trung bình -> không thấy lần chia hụt mục tiêu.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coffeeguard.bench import conformal as cf
from coffeeguard.bench import contract as ct
from coffeeguard.bench import hieu_chinh as hc
from coffeeguard.bench import tu_choi as tc

B = ct.CLASSES_BAND


def _tao_pred(n_cay: int = 150, seed: int = 0, nhon: float = 1.0) -> pd.DataFrame:
    """Mỗi cây 2 ảnh cùng fold. `nhon` > 1 làm xác suất quá tự tin (lũy thừa logit)."""
    r = np.random.default_rng(seed)
    hang = []
    for c in range(n_cay):
        for k in range(2):
            that = r.integers(0, 6)
            z = r.normal(size=6)
            z[that] += 1.5
            p = np.exp(nhon * z) / np.exp(nhon * z).sum()
            hang.append({"png": f"C{c}_{k}.png", "fold": c % 5, "plant_id": f"C{c}",
                         "leaf_id": f"C{c}_{k}", "y_true": ct.CLASSES_6[that],
                         "y_pred": ct.CLASSES_6[int(p.argmax())],
                         **{f"p_{x}": v for x, v in zip(ct.CLASSES_6, p)}})
    return pd.DataFrame(hang)


# ------------------------------------------------------------ sơ đồ chia

def test_nua_hieu_chinh_va_nua_cham_khong_chung_cay_va_cung_fold() -> None:
    pred = _tao_pred()
    cay = pred["plant_id"].to_numpy()
    fold = pred["fold"].to_numpy()
    for cap in cf.chia_nua_cay(pred, so_lan=3):
        phu = np.zeros(len(pred), dtype=int)
        for a, b in cap:
            assert not (a & b).any()
            assert not set(cay[a]) & set(cay[b])
            assert len(set(fold[a]) | set(fold[b])) == 1
            phu += b
        assert (phu == 1).all()   # mỗi ảnh được chấm đúng một lần mỗi lần chia


def test_ban_cheo_fold_trung_so_voi_tu_choi() -> None:
    pred = _tao_pred(nhon=2.0)
    P, th = hc.xac_suat_sau(pred, "nhiet_do", "cheo_fold")
    cu, T = tc.nhiet_do_cheo_fold(pred)
    assert np.allclose(P[0], tc.ma_tran_xac_suat(cu))
    assert sorted(t["T"] for t in th) == pytest.approx(sorted(T.values()), abs=1e-4)

    moi = hc.nguong_theo_dai(pred, "cheo_fold").set_index("cach")
    cu = tc.so_nguong_cung_do_phu(pred).set_index("cach")
    for c in ("do_phu", "accuracy", "do_phu__rust_nang"):
        assert moi[c].to_dict() == pytest.approx(cu[c].to_dict(), abs=1e-4)


def test_trong_fold_khong_hoc_tu_nua_duoc_cham() -> None:
    """Làm hỏng xác suất của nửa được chấm không được đổi tham số đã học."""
    pred = _tao_pred()
    _, th1 = hc.xac_suat_sau(pred, "nhiet_do", "trong_fold", so_lan=1)
    hc_mask, cham = cf.chia_nua_cay(pred, so_lan=1)[0][0]
    hong = pred.copy()
    cot = [f"p_{c}" for c in ct.CLASSES_6]
    hong.loc[cham, cot] = hong.loc[cham, cot].to_numpy() ** 5
    hong.loc[cham, cot] = hong.loc[cham, cot].div(hong.loc[cham, cot].sum(axis=1), axis=0)
    hong["y_pred"] = np.asarray(ct.CLASSES_6)[hong[cot].to_numpy().argmax(1)]
    _, th2 = hc.xac_suat_sau(hong, "nhiet_do", "trong_fold", so_lan=1)
    assert th1[0]["T"] == th2[0]["T"]       # cặp đầu: học trên hc_mask, không chạm `cham`


@pytest.mark.parametrize("pp", ["nhiet_do", "vector", "dirichlet"])
def test_hieu_chinh_giam_ece_khi_model_qua_tu_tin(pp) -> None:
    pred = _tao_pred(n_cay=300, nhon=3.0)
    kq = hc.danh_gia(pred, pp, "cheo_fold", n_bootstrap=20)
    assert kq["ece_sau"] < kq["ece_truoc"] - 0.03, kq


def test_nhiet_do_khong_doi_du_doan_con_vector_thi_co_the() -> None:
    pred = _tao_pred(nhon=2.0)
    kq = hc.danh_gia(pred, "nhiet_do", "trong_fold", so_lan=2, n_bootstrap=20)
    assert kq["n_bao_nang_sau"] == kq["n_bao_nang_truoc"]
    assert kq["macro_f1_dai_sau"] == kq["macro_f1_dai_truoc"]


def test_tu_choi_phuong_phap_la() -> None:
    with pytest.raises(ValueError, match="phuong_phap"):
        hc.danh_gia(_tao_pred(n_cay=20), "platt", n_bootstrap=5)


# ------------------------------------------------------------ conformal đối chứng

def test_aps_cong_don_theo_thu_tu_xac_suat() -> None:
    Q = np.array([[0.1, 0.5, 0.2, 0.15, 0.05]])
    s = cf.diem_khong_phu_hop(Q, "aps")[0]
    assert s == pytest.approx([0.95, 0.5, 0.7, 0.85, 1.0])
    r = cf.diem_khong_phu_hop(Q, "raps")[0]
    assert r == pytest.approx(s + cf.RAPS_LAMBDA * np.array([3, 0, 1, 2, 4]))


def test_thu_bac_moc_lien_nhau_tren_truc() -> None:
    # theo CLASSES_BAND: khoẻ, nhện đỏ, nhẹ, TB, nặng; trục: nhện đỏ, khoẻ, nhẹ, TB, nặng
    Q = np.array([[0.05, 0.05, 0.3, 0.35, 0.25]])
    s = cf.diem_khong_phu_hop(Q, "ord")[0]
    # TB (0,35) -> nhẹ (0,30 > 0,25) -> nặng -> khoẻ -> nhện đỏ
    assert s == pytest.approx([0.95, 1.0, 0.65, 0.35, 0.9])


def test_thu_bac_bien_luon_lien_nhau_va_bien_dat_do_phu_chung() -> None:
    pred = _tao_pred(n_cay=300)
    S, _ = cf.tap_du_doan(pred, 0.2, so_lan=5, diem="ord", che_do="bien")
    assert cf.lien_nhau(S).all()
    yt = np.array([B.index(ct.band_of(c)) for c in pred["y_true"]])
    assert S[:, np.arange(len(pred)), yt].mean() >= 0.8 - 0.03


@pytest.mark.parametrize("diem", cf.CACH_DIEM)
def test_moi_diem_theo_dai_dat_muc_tieu_tung_dai(diem) -> None:
    pred = _tao_pred(n_cay=300)
    kq = cf.tom_tat(pred, 0.2, so_lan=5, n_bootstrap=20, diem=diem)
    yt = pred["y_true"].map(ct.band_of)
    for b in B:
        if (yt == b).sum() >= 60:
            assert kq[f"phu__{b}"] >= 0.8 - 0.05, (diem, b, kq[f"phu__{b}"])


def test_bao_phan_tan_theo_lan_chia() -> None:
    pred = _tao_pred()
    kq = cf.tom_tat(pred, 0.2, so_lan=6, n_bootstrap=20)
    for b in B:
        assert kq[f"phu_lan_min__{b}"] <= kq[f"phu__{b}"] <= kq[f"phu_lan_max__{b}"]
        assert 0 <= kq[f"nua_duoi_muc__{b}"] <= 1
        assert kq[f"phu_nua_min__{b}"] <= kq[f"phu_lan_min__{b}"] + 1e-9


def test_cheo_fold_chi_mot_lan_va_tu_choi_tham_so_la() -> None:
    pred = _tao_pred(n_cay=50)
    S, _ = cf.tap_du_doan(pred, 0.2, so_lan=7, hieu_chinh="cheo_fold")
    assert S.shape[0] == 1
    with pytest.raises(ValueError, match="diem"):
        cf.tap_du_doan(pred, 0.2, so_lan=1, diem="lac")
    with pytest.raises(ValueError, match="che_do"):
        cf.tap_du_doan(pred, 0.2, so_lan=1, che_do="lac")
    with pytest.raises(ValueError, match="hieu_chinh"):
        cf.tap_du_doan(pred, 0.2, so_lan=1, hieu_chinh="lac")
