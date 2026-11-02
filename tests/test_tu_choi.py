"""Canh gác: độ phủ – độ chính xác và gộp hai mặt lá (`bench/tu_choi.py`).

Các lỗi im lặng được canh:

* Nhãn `y_pred` và cột xác suất theo hai thứ tự lớp khác nhau -> ngưỡng áp lên
  nhầm lớp, bảng vẫn ra số.
* Bootstrap theo ảnh thay vì theo cây -> CI hẹp giả tạo.
* Chỉ báo độ phủ chung -> lớp hiếm bị từ chối hết mà không ai thấy.
* Gộp hai mặt khi hai mặt nằm ở hai fold -> một mặt được dự đoán bởi model đã
  thấy mặt kia lúc train.
* Gộp hai mặt rồi lặng lẽ bỏ các lá hai mặt khác nhãn -> số đẹp hơn sự thật.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coffeeguard.bench import contract as ct
from coffeeguard.bench import tu_choi as tc

GOC = Path(__file__).resolve().parent.parent


def _tao_pred(n_cay: int = 40, seed: int = 0) -> pd.DataFrame:
    """Mỗi cây 2 lá × 2 mặt; xác suất ngẫu nhiên, y_pred = argmax."""
    r = np.random.default_rng(seed)
    hang = []
    for c in range(n_cay):
        for la in range(2):
            that = ct.CLASSES_6[r.integers(0, 6)]
            for mat in ("upper", "under"):
                p = r.dirichlet(np.ones(6))
                hang.append({
                    "png": f"C{c}L{la}{mat}.png", "fold": c % 5,
                    "plant_id": f"C{c}", "leaf_id": f"C{c}_{la}", "face": mat,
                    "y_true": that, "y_pred": ct.CLASSES_6[int(p.argmax())],
                    **{f"p_{k}": v for k, v in zip(ct.CLASSES_6, p)},
                })
    return pd.DataFrame(hang)


# ------------------------------------------------------------ độ phủ

def test_tu_choi_khi_y_pred_khac_argmax_xac_suat() -> None:
    pred = _tao_pred()
    pred.loc[3, "y_pred"] = next(c for c in ct.CLASSES_6 if c != pred.loc[3, "y_pred"])
    with pytest.raises(ValueError, match="argmax"):
        tc.duong_do_phu(pred, n_bootstrap=20)


def test_nguong_0_la_tra_loi_moi_anh() -> None:
    pred = _tao_pred()
    bang = tc.duong_do_phu(pred, nguong=(0.0,), n_bootstrap=50)
    assert bang.loc[0, "do_phu"] == 1.0
    assert bang.loc[0, "accuracy"] == round(float((pred.y_true == pred.y_pred).mean()), 4)


def test_do_phu_khong_tang_theo_nguong_va_ci_bao_diem() -> None:
    bang = tc.duong_do_phu(_tao_pred(), nguong=(0.0, 0.2, 0.3, 0.4), n_bootstrap=200)
    assert (bang["do_phu"].diff().dropna() <= 0).all()
    for ten in ("do_phu", "accuracy"):
        assert (bang[f"{ten}_ci_lo"] <= bang[ten]).all()
        assert (bang[ten] <= bang[f"{ten}_ci_hi"]).all()


def test_bao_do_phu_tung_dai_khi_lop_hiem_bi_tu_choi_het() -> None:
    """Lớp hiếm luôn có xác suất thấp -> độ phủ của dải đó phải về 0, dù độ phủ
    chung vẫn cao. Bảng chỉ có độ phủ chung sẽ giấu điều này."""
    pred = _tao_pred(n_cay=60, seed=1)
    hiem = pred.y_true.isin(["rust_level_3", "rust_level_4"])
    P = np.full((len(pred), 6), 0.02)
    dung = np.array([ct.chi_so_lop(c) for c in pred.y_true])
    # sau khi chuẩn hoá: lớp hiếm ~0,47, lớp khác ~0,90
    P[np.arange(len(pred)), dung] = np.where(hiem, 0.09, 0.9)
    P = P / P.sum(axis=1, keepdims=True)
    for k, c in enumerate(ct.CLASSES_6):
        pred[f"p_{c}"] = P[:, k]
    pred["y_pred"] = pred["y_true"]

    bang = tc.duong_do_phu(pred, nguong=(0.7,), n_bootstrap=20)
    assert bang.loc[0, "do_phu__rust_nang"] == 0.0
    assert bang.loc[0, "do_phu"] > 0.7


def test_ci_theo_cay_rong_hon_theo_anh() -> None:
    """Bốn ảnh giống hệt nhau trong một cây: lấy mẫu theo cây phải cho CI rộng
    hơn lấy mẫu theo ảnh. Nếu bằng nhau, hàm đang bỏ qua cột nhóm."""
    pred = _tao_pred(n_cay=50, seed=2)
    for c in ct.CLASSES_6:
        pred[f"p_{c}"] = pred.groupby("plant_id")[f"p_{c}"].transform("first")
    pred["y_pred"] = pred.groupby("plant_id")["y_pred"].transform("first")

    theo_cay = tc.duong_do_phu(pred, nguong=(0.3,), n_bootstrap=400)
    theo_anh = tc.duong_do_phu(pred, nguong=(0.3,), n_bootstrap=400, group_col="png")
    rong = lambda b: b.loc[0, "do_phu_ci_hi"] - b.loc[0, "do_phu_ci_lo"]  # noqa: E731
    assert rong(theo_cay) > 1.4 * rong(theo_anh)


# ------------------------------------------------------------ gộp hai mặt

def test_gop_tu_choi_khi_hai_mat_khac_fold() -> None:
    pred = _tao_pred()
    pred.loc[0, "fold"] = (pred.loc[0, "fold"] + 1) % 5
    with pytest.raises(ValueError, match="ro ri"):
        tc.gop_hai_mat(pred)


def test_la_thieu_mot_mat_giu_du_doan_rieng_va_duoc_dem() -> None:
    """RoCoLe trong đồ án có 1.559 ảnh: một lá chỉ còn một mặt."""
    pred = _tao_pred().iloc[1:].reset_index(drop=True)
    g = tc.gop_hai_mat(pred)
    assert g.loc[0, "mot_mat"] and g.loc[0, "y_pred_gop"] == g.loc[0, "y_pred"]
    assert tc.so_sanh_gop_hai_mat(pred, n_bootstrap=20)["n_la_mot_mat"] == 1


def test_gop_tu_choi_khi_la_co_hon_hai_anh() -> None:
    pred = _tao_pred()
    pred.loc[2, "leaf_id"] = pred.loc[0, "leaf_id"]
    pred.loc[2, "fold"] = pred.loc[0, "fold"]
    with pytest.raises(ValueError, match="hon 2 anh"):
        tc.gop_hai_mat(pred)


def test_hai_mat_giong_nhau_thi_gop_khong_doi_gi() -> None:
    pred = _tao_pred(seed=3)
    for c in ct.CLASSES_6:
        pred[f"p_{c}"] = pred.groupby("leaf_id")[f"p_{c}"].transform("first")
    pred["y_pred"] = pred.groupby("leaf_id")["y_pred"].transform("first")
    kq = tc.so_sanh_gop_hai_mat(pred, n_bootstrap=100)
    assert kq["hieu_accuracy"] == 0.0 and kq["hieu_macro_f1_dai"] == 0.0


def test_mac_dinh_giu_ca_la_hai_mat_khac_nhan() -> None:
    pred = _tao_pred(seed=4)
    pred.loc[0, "y_true"] = next(c for c in ct.CLASSES_6 if c != pred.loc[1, "y_true"])
    du = tc.so_sanh_gop_hai_mat(pred, n_bootstrap=50)
    con = tc.so_sanh_gop_hai_mat(pred, chi_la_cung_nhan=True, n_bootstrap=50)
    assert du["n_anh"] == len(pred)
    assert con["n_anh"] == len(pred) - 2


def test_gop_can_theo_leaf_id_khong_theo_thu_tu_dong() -> None:
    pred = _tao_pred(seed=5)
    a = tc.gop_hai_mat(pred).set_index("png").y_pred_gop
    b = tc.gop_hai_mat(pred.sample(frac=1.0, random_state=9)).set_index("png").y_pred_gop
    assert (a.sort_index() == b.sort_index()).all()


# ------------------------------------------------------------ hiệu chỉnh

def _dat_tin_cay(pred: pd.DataFrame, tin_cay: np.ndarray, dung: np.ndarray) -> pd.DataFrame:
    """Ghi đè xác suất: lớp dự đoán có xác suất `tin_cay`, phần còn lại chia đều.
    `dung[i]` quyết định y_pred có bằng y_true không. Cần tin_cay > 1/6."""
    pred = pred.copy()
    yt = np.array([ct.chi_so_lop(c) for c in pred.y_true])
    yp = np.where(dung, yt, (yt + 1) % 6)
    P = np.repeat(((1 - tin_cay) / 5)[:, None], 6, axis=1)
    P[np.arange(len(pred)), yp] = tin_cay
    for k, c in enumerate(ct.CLASSES_6):
        pred[f"p_{c}"] = P[:, k]
    pred["y_pred"] = np.asarray(ct.CLASSES_6)[yp]
    return pred


def test_ece_bang_khoang_cach_khi_moi_anh_cung_tin_cay() -> None:
    """Mọi ảnh tin cậy 0,9 nhưng chỉ đúng 60% -> ECE = MCE = 0,3, quá tự tin."""
    pred = _tao_pred(n_cay=50)
    dung = np.arange(len(pred)) % 5 < 3
    pred = _dat_tin_cay(pred, np.full(len(pred), 0.9), dung)
    kq = tc.ece(pred, n_bootstrap=50)
    assert kq["ece"] == pytest.approx(0.3, abs=1e-4)
    assert kq["mce"] == pytest.approx(0.3, abs=1e-4)
    assert kq["lech_tb"] == pytest.approx(0.3, abs=1e-4)


def test_ece_bang_0_khi_hieu_chinh_hoan_hao() -> None:
    """Mỗi bin: tin cậy c, đúng đúng tỉ lệ c -> ECE = 0 với cả hai kiểu bin."""
    pred = _tao_pred(n_cay=50)  # 200 ảnh = 4 nhóm × 50
    tin_cay = np.repeat([0.4, 0.6, 0.8, 1.0], 50)
    dung = np.concatenate([np.arange(50) < 50 * c for c in (0.4, 0.6, 0.8, 1.0)])
    pred = _dat_tin_cay(pred, tin_cay, dung)
    assert tc.ece(pred, n_bootstrap=20)["ece"] == pytest.approx(0.0, abs=1e-9)
    assert tc.ece(pred, so_bin=4, kieu_bin="deu_so_anh", n_bootstrap=20)["ece"] == \
        pytest.approx(0.0, abs=1e-9)


def test_lech_trai_dau_bu_nhau_trong_lech_tb_nhung_khong_trong_ece() -> None:
    """Nửa tin cậy 0,9, nửa 0,3, cả hai đều đúng 60%: lệch trung bình = 0 nhưng
    ECE = 0,3. Chỉ báo 'tin cậy TB ≈ accuracy' sẽ giấu điều này."""
    pred = _tao_pred(n_cay=50)
    tin_cay = np.repeat([0.9, 0.3], 100)
    dung = np.concatenate([np.arange(100) < 60, np.arange(100) < 60])
    pred = _dat_tin_cay(pred, tin_cay, dung)
    kq = tc.ece(pred, n_bootstrap=20)
    assert abs(kq["lech_tb"]) < 0.01
    assert kq["ece"] == pytest.approx(0.3, abs=1e-4)


def test_so_do_dem_du_anh_ke_ca_tin_cay_1() -> None:
    pred = _tao_pred()
    pred = _dat_tin_cay(pred, np.linspace(0.2, 1.0, len(pred)), np.ones(len(pred), bool))
    so_do = tc.so_do_tin_cay(pred)
    assert len(so_do) == 10 and so_do["n"].sum() == len(pred)
    assert so_do.loc[9, "tin_cay_max"] == 1.0
    assert so_do.loc[0, "n"] == 0 and np.isnan(so_do.loc[0, "accuracy"])


def test_bin_deu_so_anh_chia_gan_bang_nhau() -> None:
    pred = _tao_pred(n_cay=40)  # 160 ảnh
    so_do = tc.so_do_tin_cay(pred, so_bin=8, kieu_bin="deu_so_anh")
    assert (so_do["n"] == 20).all()
    with pytest.raises(ValueError, match="kieu_bin"):
        tc.so_do_tin_cay(pred, kieu_bin="quantile")


def test_ece_ci_theo_cay_bao_diem_va_rong_hon_theo_anh() -> None:
    pred = _tao_pred(n_cay=60, seed=6)
    for c in ct.CLASSES_6:
        pred[f"p_{c}"] = pred.groupby("plant_id")[f"p_{c}"].transform("first")
    pred["y_pred"] = pred.groupby("plant_id")["y_pred"].transform("first")
    pred["y_true"] = pred.groupby("plant_id")["y_true"].transform("first")
    theo_cay = tc.ece(pred, n_bootstrap=400)
    theo_anh = tc.ece(pred, n_bootstrap=400, group_col="png")
    lo, hi = theo_cay["lech_tb_ci"]
    assert lo <= theo_cay["lech_tb"] <= hi
    rong = lambda k: k["lech_tb_ci"][1] - k["lech_tb_ci"][0]  # noqa: E731
    assert rong(theo_cay) > 1.4 * rong(theo_anh)


def test_lech_theo_du_doan_lo_ra_lop_benh_qua_tu_tin() -> None:
    """Lá khoẻ hiệu chỉnh tốt, dự đoán gỉ sắt nặng tin cậy 0,9 nhưng sai hết:
    bảng tách theo dải phải chỉ ra đúng dải đó."""
    pred = _tao_pred(n_cay=60, seed=7)
    pred["y_true"] = np.where(np.arange(len(pred)) % 4 == 0, "rust_level_3", "healthy")
    nang = pred.y_true.eq("rust_level_3").to_numpy()
    tin_cay = np.where(nang, 0.9, 0.95)
    # luôn sai: _dat_tin_cay đoán lớp kế tiếp, rust_level_3 -> rust_level_4,
    # vẫn thuộc dải rust_nang nhưng sai ở mức 6 lớp
    pred = _dat_tin_cay(pred, tin_cay, ~nang)
    assert (pred.loc[nang, "y_pred"] == "rust_level_4").all()

    bang = tc.lech_theo_du_doan(pred, n_bootstrap=50).set_index("dai_du_doan")
    assert bang.loc["rust_nang", "lech"] == pytest.approx(0.9, abs=1e-4)
    assert bang.loc["rust_nang", "lech_ci_lo"] > 0
    assert bang.loc["healthy", "lech"] == pytest.approx(-0.05, abs=1e-4)
    assert bang.loc["red_spider_mite", "n"] == 0


# ------------------------------------------------------------ dữ liệu thật

def test_nguong_07_khop_bao_cao_tuan_4() -> None:
    """Báo cáo tuần 4 ghi: ngưỡng >= 0,7 trả lời 62,5% ảnh, đúng 94,2%."""
    p = GOC / "reports/predictions/pred__pytorch-timm__mobilevitv2_050__20260930-1450.csv"
    if not p.exists():
        pytest.skip(f"thiếu {p.name}")
    bang = tc.duong_do_phu(pd.read_csv(p), nguong=(0.7,), n_bootstrap=50)
    assert bang.loc[0, "do_phu"] == pytest.approx(0.625, abs=5e-4)
    assert bang.loc[0, "accuracy"] == pytest.approx(0.942, abs=5e-4)


def test_ece_khop_khoang_trong_bai_bao_muc_3() -> None:
    """`Khoang_trong_bai_bao.md` §3 ghi: ECE 0,073 (10 bin đều), tính chung thiếu tự
    tin, nhưng dự đoán gỉ sắt nặng quá tự tin +0,154."""
    p = GOC / "reports/predictions/pred__pytorch-timm__mobilevitv2_050__20260930-1450.csv"
    if not p.exists():
        pytest.skip(f"thiếu {p.name}")
    pred = pd.read_csv(p)
    kq = tc.ece(pred, n_bootstrap=50)
    assert kq["ece"] == pytest.approx(0.0734, abs=5e-4)
    assert kq["lech_tb"] == pytest.approx(-0.0601, abs=5e-4)
    bang = tc.lech_theo_du_doan(pred, n_bootstrap=50).set_index("dai_du_doan")
    assert bang.loc["rust_nang", "lech"] == pytest.approx(0.1537, abs=5e-4)
    assert bang.loc["healthy", "lech"] == pytest.approx(-0.1157, abs=5e-4)


# ------------------------------------------------- lệch theo định nghĩa "đúng"

def test_lech_theo_dai_dung_xac_suat_dai_va_dung_theo_dai() -> None:
    """Đoán cấp 3, thật cấp 4: sai theo 6 lớp, đúng theo dải. Tin cậy theo dải
    phải là p3 + p4, không phải p3."""
    p = {f"p_{c}": 0.0 for c in ct.CLASSES_6}
    p.update(p_rust_level_3=0.5, p_rust_level_4=0.4, p_healthy=0.1)
    pred = pd.DataFrame([{"png": f"{i}.png", "fold": 0, "plant_id": f"C{i}",
                          "y_true": "rust_level_4", "y_pred": "rust_level_3", **p}
                         for i in range(10)])
    d6 = tc.lech_theo_du_doan(pred, n_bootstrap=20).set_index("dai_du_doan").loc["rust_nang"]
    dd = tc.lech_theo_du_doan(pred, n_bootstrap=20, dinh_nghia="dai").set_index("dai_du_doan").loc["rust_nang"]
    assert (d6.tin_cay_tb, d6.accuracy) == (0.5, 0.0)
    assert (dd.tin_cay_tb, dd.accuracy) == (0.9, 1.0)
    assert dd.lech == pytest.approx(-0.1)


def test_lech_tu_choi_dinh_nghia_tron() -> None:
    with pytest.raises(ValueError, match="dinh_nghia"):
        tc.lech_theo_du_doan(_tao_pred(), n_bootstrap=10, dinh_nghia="tin6_dung_dai")


def test_ghim_lech_dai_nang_luot_A() -> None:
    """`Docs/Khoang_trong_bai_bao.md` §6.2: mobilevitv2_050 báo nặng, theo dải +0,154."""
    p = GOC / "reports/predictions/pred__pytorch-timm__mobilevitv2_050__20261022-1539-luotA.csv"
    if not p.exists():
        pytest.skip(f"thiếu {p.name}")
    r = tc.lech_theo_du_doan(pd.read_csv(p), n_bootstrap=200, dinh_nghia="dai")
    r = r.set_index("dai_du_doan").loc["rust_nang"]
    assert r.n == 79
    assert r.lech == pytest.approx(0.154, abs=0.001)
    assert r.lech_ci_lo > 0


# ------------------------------------------------------------ §6.3: hai biện pháp hiển nhiên

def _tao_pred_qua_tu_tin(T_that: float = 2.0, n_cay: int = 300, seed: int = 0) -> pd.DataFrame:
    """Nhãn rút từ softmax(z) (đã hiệu chỉnh), nhưng file ghi softmax(z·T_that) —
    model quá tự tin đúng một nhiệt độ T_that."""
    r = np.random.default_rng(seed)
    hang = []
    for c in range(n_cay):
        for k in range(2):
            z = r.normal(scale=1.5, size=6)
            that = r.choice(6, p=np.exp(z) / np.exp(z).sum())
            p = np.exp(z * T_that) / np.exp(z * T_that).sum()
            hang.append({"png": f"C{c}_{k}.png", "fold": c % 5, "plant_id": f"C{c}",
                         "leaf_id": f"C{c}_{k}", "y_true": ct.CLASSES_6[that],
                         "y_pred": ct.CLASSES_6[int(p.argmax())],
                         **{f"p_{x}": v for x, v in zip(ct.CLASSES_6, p)}})
    return pd.DataFrame(hang)


def test_nhiet_do_tim_lai_T_va_giu_argmax() -> None:
    pred = _tao_pred_qua_tu_tin(2.0)
    sau, T = tc.nhiet_do_cheo_fold(pred)
    for t in T.values():
        assert t == pytest.approx(2.0, rel=0.2)
    assert (sau["y_pred"] == pred["y_pred"]).all()
    tc.ma_tran_xac_suat(sau)   # không ném lỗi: xác suất và nhãn cùng thứ tự lớp
    assert tc.ece(sau, n_bootstrap=10)["ece"] < tc.ece(pred, n_bootstrap=10)["ece"]


def test_nhiet_do_cua_mot_fold_khong_hoc_tu_chinh_fold_do() -> None:
    """Đổi hẳn fold 0 thành model thiếu tự tin: T của fold 0 không được nhúc nhích
    (nó học trên fold 1–4), T của các fold khác thì phải đổi."""
    a = _tao_pred_qua_tu_tin(2.0)
    b = _tao_pred_qua_tu_tin(2.0)
    m = b["fold"] == 0
    P = b.loc[m, list(tc.COT_XAC_SUAT)].to_numpy() ** 0.2
    b.loc[m, list(tc.COT_XAC_SUAT)] = P / P.sum(1, keepdims=True)
    _, Ta = tc.nhiet_do_cheo_fold(a)
    _, Tb = tc.nhiet_do_cheo_fold(b)
    assert Ta[0] == Tb[0]
    assert all(Ta[f] != Tb[f] for f in range(1, 5))


def test_nguong_theo_dai_hoc_tren_fold_khac() -> None:
    a = _tao_pred_qua_tu_tin(1.0)
    b = a.copy()
    m = b["fold"] == 0
    b.loc[m, "y_true"] = b.loc[m, "y_pred"]      # fold 0 bỗng đúng hết
    _, na = tc.nguong_theo_dai(a, muc_tieu=0.6)
    _, nb = tc.nguong_theo_dai(b, muc_tieu=0.6)
    f0 = na["fold"] == 0
    assert na[f0].equals(nb[f0])
    assert not na[~f0].equals(nb[~f0])


def test_so_nguong_o_cung_do_phu_va_tu_choi_dinh_nghia_tron() -> None:
    bang = tc.so_nguong_cung_do_phu(_tao_pred_qua_tu_tin(1.0), muc_tieu=0.6).set_index("cach")
    assert bang.loc["chung", "do_phu"] == bang.loc["theo_dai", "do_phu"]
    with pytest.raises(ValueError, match="dinh_nghia"):
        tc.nguong_theo_dai(_tao_pred(), dinh_nghia="tron")


def _doc_luot_A(arch: str) -> pd.DataFrame:
    p = GOC / f"reports/predictions/pred__pytorch-timm__{arch}__20261022-1539-luotA.csv"
    if not p.exists():
        pytest.skip(f"thiếu {p.name}")
    return pd.read_csv(p)


def test_ghim_nhiet_do_luot_A() -> None:
    """§6.3: ECE giảm nhưng lệch khi báo nặng KHÔNG giảm (còn tăng)."""
    r = tc.danh_gia_nhiet_do(_doc_luot_A("mobilevitv2_050"), n_bootstrap=100)
    assert all(0.7 < t < 0.85 for t in r["T_theo_fold"].values())
    assert r["ece_truoc"] == pytest.approx(0.068, abs=0.001)
    assert r["ece_sau"] == pytest.approx(0.043, abs=0.001)
    assert r["lech_dai__rust_nang__sau"] >= r["lech_dai__rust_nang__truoc"]
    assert r["lech_dai__rust_nang__sau_ci"][0] > 0


def test_ghim_nguong_theo_dai_luot_A() -> None:
    """§6.3: ngưỡng theo dải tăng accuracy nhưng gần như im lặng với lá nặng."""
    b = tc.so_nguong_cung_do_phu(_doc_luot_A("effnetv2_b0")).set_index("cach")
    assert b.loc["theo_dai", "do_phu"] == pytest.approx(0.533, abs=0.001)
    assert b.loc["chung", "accuracy"] == pytest.approx(0.862, abs=0.001)
    assert b.loc["theo_dai", "accuracy"] == pytest.approx(0.911, abs=0.001)
    assert b.loc["theo_dai", "do_phu__rust_nang"] < 0.05 < b.loc["chung", "do_phu__rust_nang"]


def test_p_bootstrap_hai_phia_va_khong_bang_0() -> None:
    assert tc.p_bootstrap(np.ones(999)) == pytest.approx(2 / 1000)
    assert tc.p_bootstrap(-np.ones(999)) == pytest.approx(2 / 1000)
    assert tc.p_bootstrap(np.r_[np.ones(50), -np.ones(50)]) == 1.0
    assert np.isnan(tc.p_bootstrap(np.array([])))


def test_mau_bootstrap_cung_seed_cho_cung_ci() -> None:
    """CI và p-value phải đọc từ CÙNG các mẫu, nếu không có thể mâu thuẫn nhau."""
    g = np.repeat(np.arange(30), 2)
    x = np.random.default_rng(1).normal(size=60)
    mau = tc._mau_bootstrap_cum(g, lambda i: float(x[i].mean()), 300, 7)
    lo, hi = tc._bootstrap_cum(g, lambda i: float(x[i].mean()), 300, 7, 0.95)
    assert (lo, hi) == (float(np.quantile(mau, 0.025)), float(np.quantile(mau, 0.975)))
