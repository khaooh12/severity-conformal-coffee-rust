"""Canh gác tầng benchmark.

Mỗi test ở đây tương ứng một lỗi **im lặng**: nếu nó đỏ thì nghĩa là một lỗi
không gây exception, không làm chỉ số xấu đi, và không có gì báo — vừa lọt vào.

Đừng sửa test cho xanh. Sửa nguyên nhân.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from coffeeguard.bench import contract as ct
from coffeeguard.bench import metrics as bm
from coffeeguard.bench import paths as P
from coffeeguard.bench import predictions as bpred
from coffeeguard.bench import preprocess as bprep

# --------------------------------------------------------------- hợp đồng lớp


def test_thu_tu_lop_bat_bien():
    """Thứ tự lớp là BẤT BIẾN — model đã train theo đúng thứ tự này.

    Đổi thứ tự (kể cả sắp xếp lại cho 'gọn') làm mọi checkpoint cũ giải mã ra
    nhãn SAI với confidence VẪN CAO, không exception. Nếu thật sự phải đổi thì
    phải train lại toàn bộ và đổi cả `registry.json`.
    """
    assert ct.CLASSES_6 == (
        "healthy",
        "red_spider_mite",
        "rust_level_1",
        "rust_level_2",
        "rust_level_3",
        "rust_level_4",
    )


def test_moi_lop_co_dai_tuong_ung():
    assert set(ct.BAND_OF) == set(ct.CLASSES_6)
    assert set(ct.BAND_OF.values()) == set(ct.CLASSES_BAND)


def test_registry_web_khop_contract():
    """`registry.json` của PWA mang bản sao ánh xạ dải — hai bản phải khớp.

    Lệch nhau thì PWA gộp dải sai mà vẫn hiện kết quả bình thường: người dùng
    thấy 'gỉ sắt nặng' trong khi model nói 'nhẹ', và không có gì báo.
    """
    reg_path = P.ROOT / "web" / "public" / "models" / "registry.json"
    if not reg_path.exists():
        pytest.skip("chua co registry.json cua PWA")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    assert tuple(reg["classes_band"]) == ct.CLASSES_BAND
    assert reg["band_of"] == ct.BAND_OF


def test_registry_web_khai_dung_so_lop():
    """Mỗi model trong registry phải khai đúng `classes` của chính nó (quy tắc 10)."""
    reg_path = P.ROOT / "web" / "public" / "models" / "registry.json"
    if not reg_path.exists():
        pytest.skip("chua co registry.json cua PWA")
    reg = json.loads(reg_path.read_text(encoding="utf-8"))
    for m in reg.get("models", []):
        assert isinstance(m.get("classes"), list), f"model {m.get('id')} thieu 'classes'"
        assert len(m["classes"]) >= 2
        assert len(m["mean"]) == 3 and len(m["std"]) == 3, "mean/std phai di kem model"


# ------------------------------------------------------------ cửa sổ cắt ảnh


def test_neo_canh_ngan_thi_o_vuong_nho_hon_canh_dai():
    """Neo cạnh NGẮN cho ô vuông nhỏ hơn hẳn — đó là cả mục đích của nó.

    Neo nhầm vào cạnh dài thì với lá 2,74:1 ô vuông toàn nền (11,6% là lá, tệ
    hơn cả letterbox) mà code vẫn chạy trơn tru và ảnh trông vẫn 'hợp lý'.
    """
    la = dict(leaf_x=200.0, leaf_y=400.0, leaf_w=1600.0, leaf_h=400.0)
    _, _, ngan = bprep.cua_so_vuong(**la, scale=1.4, neo="ngan")
    _, _, dai = bprep.cua_so_vuong(**la, scale=1.4, neo="dai")
    assert ngan == 560 and dai == 2240
    assert ngan < dai


def test_cua_so_luon_tam_tai_tam_la():
    la = dict(leaf_x=200.0, leaf_y=400.0, leaf_w=1600.0, leaf_h=400.0)
    for neo in ("ngan", "dai"):
        x0, y0, canh = bprep.cua_so_vuong(**la, scale=1.4, neo=neo)
        assert x0 + canh / 2 == pytest.approx(1000.0, abs=1.0)
        assert y0 + canh / 2 == pytest.approx(600.0, abs=1.0)


def test_cat_vuong_giu_nguyen_kich_thuoc_khi_tran_ra_ngoai():
    """Cửa sổ tràn mép ảnh thì chèn viền, KHÔNG co lại.

    Co lại sẽ đổi tỉ lệ lá/khung ở đúng những ảnh lá nằm sát mép — tức là lén
    đưa thêm một biến vào dữ liệu, và biến đó tương quan với vị trí chụp.
    """
    from PIL import Image

    im = Image.new("RGB", (100, 80), (10, 20, 30))
    ra = bprep.cat_vuong(im, x0=-50, y0=-50, canh=200)
    assert ra.size == (200, 200)
    assert ra.getpixel((0, 0)) == (0, 0, 0)


def test_ti_le_la_khong_bao_gio_vuot_100_phan_tram():
    """Đo bằng rasterize đa giác, nên không thể vượt 1.0.

    Công thức cũ `dien_tich_la / canh**2` vượt 100% khi cửa sổ cắt mất đầu lá —
    và con số vô lý đó vẫn in ra bình thường rồi đi thẳng vào báo cáo.
    """
    poly = json.dumps([0, 0, 1000, 0, 1000, 200, 0, 200])  # lá 1000x200
    x0, y0, canh = bprep.cua_so_vuong(0, 0, 1000, 200, scale=1.0, neo="ngan")
    frac = bprep.ti_le_la_trong_cua_so(poly, x0, y0, canh)
    assert frac is not None
    assert 0.9 <= frac <= 1.0


def test_ti_le_la_tra_none_khi_thieu_da_giac():
    assert bprep.ti_le_la_trong_cua_so("", 0, 0, 100) is None
    assert bprep.ti_le_la_trong_cua_so("[]", 0, 0, 100) is None


# ------------------------------------------------------------------ chỉ số


def _du_doan_gia(n=200, seed=0, hoan_hao=True):
    rng = np.random.default_rng(seed)
    y = rng.choice(list(ct.CLASSES_6), size=n, p=[0.5, 0.1, 0.22, 0.11, 0.04, 0.03])
    yp = y.copy() if hoan_hao else rng.choice(list(ct.CLASSES_6), size=n)
    Pm = np.full((n, len(ct.CLASSES_6)), 0.02)
    for i, c in enumerate(yp):
        Pm[i, ct.CLASSES_6.index(c)] = 1 - 0.02 * (len(ct.CLASSES_6) - 1)
    df = pd.DataFrame(
        {
            "png": [f"C{i}P1H1.png" for i in range(n)],
            "fold": rng.integers(0, 5, n),
            "plant_id": [f"C{i // 4}P1" for i in range(n)],
            "leaf_id": [f"C{i // 4}P1_1" for i in range(n)],
            "face": "upper",
            "y_true": y,
            "y_pred": yp,
        }
    )
    for j, c in enumerate(ct.CLASSES_6):
        df[f"p_{c}"] = Pm[:, j]
    df["framework"], df["arch"], df["run_id"] = "test", "gia", "r1"
    return df


def test_macro_f1_hoan_hao_bang_1():
    df = _du_doan_gia(hoan_hao=True)
    assert bm.macro_f1(df.y_true, df.y_pred) == pytest.approx(1.0)


def test_bang_tung_lop_luon_co_cot_n():
    """Không có `n` thì recall của `rust_level_4` (29 ảnh) đọc như recall của 791 ảnh."""
    df = _du_doan_gia()
    bang = bm.bang_tung_lop(df.y_true, df.y_pred)
    assert "n" in bang.columns
    assert bang.n.sum() == len(df)
    assert list(bang.lop) == list(ct.CLASSES_6)


def test_bootstrap_lay_mau_theo_cum_khong_theo_anh():
    """Bootstrap phải lấy mẫu lại CÂY. Lấy theo ảnh cho CI hẹp giả tạo.

    Ảnh cùng một cây không độc lập (cùng lá, cùng buổi chụp, cùng nền). Coi
    chúng độc lập làm khoảng tin cậy hẹp hơn sự thật -> kết luận 'A hơn B có ý
    nghĩa thống kê' trong khi không phải.
    """
    df = _du_doan_gia(n=400, seed=1, hoan_hao=False)
    cum = bm.bootstrap_cum(df.y_true, df.y_pred, df.plant_id, n_lan=200, seed=0)
    anh = bm.bootstrap_cum(df.y_true, df.y_pred, df.png, n_lan=200, seed=0)
    assert cum["n_cum"] < anh["n_cum"]
    assert (cum["ci_hi"] - cum["ci_lo"]) > (anh["ci_hi"] - anh["ci_lo"])


def test_danh_gia_luon_kem_accuracy_doan_bua():
    """Accuracy luôn phải đi kèm sàn 'đoán bừa' — nếu không nó bị đọc nhầm."""
    df = _du_doan_gia()
    kq = bm.danh_gia(df, n_bootstrap=100)
    assert kq.accuracy_da_doan_bua > 0.4
    assert "acc_doan_bua" in kq.dong_tom_tat()


def test_ci_giao_nhau_thi_khong_ket_luan():
    """Quy tắc 13: CI giao nhau thì báo cáo phải viết 'tương đương'."""
    a = bm.danh_gia(_du_doan_gia(seed=2, hoan_hao=False), n_bootstrap=200)
    b = bm.danh_gia(_du_doan_gia(seed=3, hoan_hao=False), n_bootstrap=200)
    cau = bm.chenh_lech_co_y_nghia(a, b)
    assert "KHONG ket luan duoc" in cau or ">" in cau


# ------------------------------------------------- kiểm chứng CSV dự đoán


def test_bat_quen_softmax():
    df = _du_doan_gia()
    df["p_healthy"] = df["p_healthy"] * 3
    with pytest.raises(bpred.LoiHopDong, match="tong xac suat"):
        bpred.kiem_chung(df, kiem_split=False)


def test_bat_giai_ma_sai_thu_tu_lop():
    """argmax(xác suất) lệch y_pred = dấu hiệu giải mã bằng danh sách lớp khác.

    Đây chính là quy tắc 10: nhãn sai, confidence vẫn cao, không exception.
    """
    df = _du_doan_gia()
    df.loc[: len(df) // 2, "y_pred"] = "rust_level_4"
    with pytest.raises(bpred.LoiHopDong, match="argmax"):
        bpred.kiem_chung(df, kiem_split=False)


def test_bat_anh_trung_lap():
    df = _du_doan_gia()
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    with pytest.raises(bpred.LoiHopDong, match="nhieu hon mot lan"):
        bpred.kiem_chung(df, kiem_split=False)


def test_bat_nhan_ngoai_hop_dong():
    df = _du_doan_gia()
    df.loc[0, "y_true"] = "rust_level_9"
    with pytest.raises(bpred.LoiHopDong, match="ngoai hop dong"):
        bpred.kiem_chung(df, kiem_split=False)


def test_bat_nhieu_lan_chay_trong_mot_file():
    df = _du_doan_gia()
    df.loc[0, "run_id"] = "r2"
    with pytest.raises(bpred.LoiHopDong, match="run_id"):
        bpred.kiem_chung(df, kiem_split=False)


def test_canh_bao_khi_model_chi_doan_mot_lop():
    """Model sụp về một lớp vẫn đạt ~50,7% accuracy — phải được nói ra."""
    df = _du_doan_gia()
    df["y_pred"] = "healthy"
    for c in ct.CLASSES_6:
        df[f"p_{c}"] = 1.0 if c == "healthy" else 0.0
    cb = bpred.kiem_chung(df, kiem_split=False)
    assert any("DUY NHAT" in c for c in cb)


def test_tu_choi_ghi_de_ket_qua_cu(tmp_path: Path):
    df = _du_doan_gia()
    bpred.ghi_du_doan(df, "test", "gia", "r1", thu_muc=tmp_path)
    with pytest.raises(bpred.LoiHopDong, match="DA TON TAI"):
        bpred.ghi_du_doan(df, "test", "gia", "r1", thu_muc=tmp_path)


# ----------------------------------------------------- split đã đóng băng


def test_split_dong_bang_khop_index():
    """File split phải khớp đúng bản index sinh ra nó (quy tắc 12).

    Index đổi mà split không đổi = fold đang trỏ vào một tập ảnh khác, và mọi
    con số vẫn ra bình thường.
    """
    if not (P.FOLDS_JSON.exists() and P.INDEX_CSV.exists()):
        pytest.skip("chua chay `bench.py index` / `splits`")
    from coffeeguard.bench.splits import doc_split

    meta = doc_split()
    assert meta["index_sha256"] == P.bam_file(P.INDEX_CSV), (
        "Index da doi sau khi split duoc dong bang. Hoac khoi phuc index, hoac "
        "sinh lai split VA train lai TAT CA model."
    )


def test_split_khong_ro_ri_cay_va_la():
    if not (P.FOLDS_JSON.exists() and P.INDEX_CSV.exists()):
        pytest.skip("chua chay `bench.py index` / `splits`")
    from coffeeguard.bench.index import doc_index
    from coffeeguard.bench.splits import doc_split, gan_fold

    df = gan_fold(doc_index(), doc_split())
    for f in sorted(df.fold.unique()):
        tr, te = df[df.fold != f], df[df.fold == f]
        assert not (set(tr.plant_id) & set(te.plant_id)), f"ro ri CAY o fold {f}"
        assert not (set(tr.leaf_id) & set(te.leaf_id)), f"ro ri LA o fold {f}"


def test_moi_lop_co_mat_o_moi_fold():
    """Lớp vắng mặt ở một fold thì macro-F1 fold đó tính trên ít lớp hơn.

    Khi đó độ lệch chuẩn giữa các fold phồng lên vì lý do sai, và ta tưởng
    'kiến trúc này không ổn định'.
    """
    if not P.FOLDS_JSON.exists():
        pytest.skip("chua chay `bench.py splits`")
    from coffeeguard.bench.splits import doc_split

    meta = doc_split()
    for lop, dem in meta["so_anh_moi_lop_theo_fold"].items():
        assert all(n > 0 for n in dem), f"lop '{lop}' vang mat o mot fold: {dem}"
