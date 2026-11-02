"""Canh gác: so sánh hai model phải THEO CẶP, không so hai CI95 biên.

Lỗi được canh (đã mắc thật 30/09/2026): `score` so hai khoảng tin cậy **biên**
như thể hai mẫu độc lập, trong khi mọi model chạy trên **CÙNG 1.559 ảnh**, cùng
một file split đã đóng băng.

Hậu quả rất cụ thể trên dữ liệu thật của đồ án:

| Phép thử | Cặp liền kề (5) | Mọi cặp (15) |
|---|---|---|
| CI95 **biên** không giao nhau (sai, quá bảo thủ) | 1/5 | 9/15 |
| CI95 của **HIỆU**, bootstrap theo cặp (đúng) | 4/5 | 14/15 |

Lỗi ban đầu gồm hai lỗi chồng nhau — so CI biên VÀ chỉ so cặp liền kề — cho 1/5:
toàn bộ 5 kiến trúc học sâu thành **"tương đương"**, tức cả tầng benchmark không
trả lời được câu hỏi nó sinh ra để trả lời. Không có exception, bảng vẫn đẹp.
(Tài liệu từng ghi "1/15": mẫu số sai. Hai test cuối file tính lại bảng này từ
file dự đoán thật và so với `Docs/Quy_tac_thiet_ke.md`.)

Lý do thống kê: phần lớn độ rộng của mỗi CI biên đến từ *fold/cây nào rơi vào
đâu* — nguồn biến thiên mà **cả hai model cùng chịu** — nên nó triệt tiêu khi
lấy hiệu trên cùng một lần lấy mẫu lại.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from coffeeguard.bench import contract as ct
from coffeeguard.bench import metrics as bm


def _tao_pred(rng, n_cay: int, doi_dung: float, seed_lop: int = 0) -> pd.DataFrame:
    """Sinh một file dự đoán giả: mỗi cây 4 ảnh, `doi_dung` là tỉ lệ đoán đúng."""
    r = np.random.default_rng(seed_lop)
    hang = []
    for c in range(n_cay):
        for k in range(4):
            that = ct.CLASSES_6[r.integers(0, len(ct.CLASSES_6))]
            doan = that if rng.random() < doi_dung else ct.CLASSES_6[
                rng.integers(0, len(ct.CLASSES_6))]
            hang.append({"png": f"C{c}P{k}.png", "fold": c % 5,
                         "plant_id": f"C{c}", "y_true": that, "y_pred": doan})
    return pd.DataFrame(hang)


def test_can_theo_png_chu_khong_tin_thu_tu_dong() -> None:
    """Hai file xáo trộn thứ tự dòng vẫn phải so đúng.

    Nếu tin vào thứ tự dòng thì khi hai notebook ghi khác thứ tự, hiệu vẫn tính
    ra một con số TRÔNG HỢP LÝ mà không có gì báo.
    """
    rng = np.random.default_rng(1)
    a = _tao_pred(rng, 40, 0.80)
    b = a.copy()
    b["y_pred"] = a["y_pred"]
    b = b.sample(frac=1.0, random_state=7).reset_index(drop=True)  # xáo trộn

    r = bm.so_sanh_theo_cap("a", "b", a, b, n_bootstrap=200)
    assert abs(r["hieu"]) < 1e-12, (
        f"Cùng một dự đoán, chỉ khác thứ tự dòng, mà hiệu = {r['hieu']}. "
        f"so_sanh_theo_cap đang tin thứ tự dòng thay vì căn theo `png`."
    )


def test_tu_choi_khi_hai_model_khac_tap_anh() -> None:
    """Thiếu/thừa ảnh phải NÉM LỖI, không được lặng lẽ so phần giao."""
    rng = np.random.default_rng(2)
    a = _tao_pred(rng, 30, 0.8)
    b = _tao_pred(rng, 30, 0.7).iloc[:-8]          # thiếu 8 ảnh

    with pytest.raises(ValueError, match="khong cung tap anh"):
        bm.so_sanh_theo_cap("a", "b", a, b, n_bootstrap=50)


def test_tu_choi_khi_y_true_lech_nhau() -> None:
    """Cùng ảnh mà `y_true` khác nhau = một file giải mã nhãn sai thứ tự lớp."""
    rng = np.random.default_rng(3)
    a = _tao_pred(rng, 30, 0.8)
    b = a.copy()
    b.loc[0, "y_true"] = ct.CLASSES_6[(ct.CLASSES_6.index(b.loc[0, "y_true"]) + 1)
                                      % len(ct.CLASSES_6)]

    with pytest.raises(ValueError, match="y_true KHAC nhau"):
        bm.so_sanh_theo_cap("a", "b", a, b, n_bootstrap=50)


def test_theo_cap_MANH_HON_so_hai_ci_bien() -> None:
    """Điểm mấu chốt: phép thử theo cặp phải hẹp hơn hiệu của hai CI biên.

    Dựng hai model mà model B đúng hơn A một chút **trên đúng những ảnh đó**.
    Khoảng tin cậy của HIỆU phải hẹp hơn hẳn khoảng suy ra từ hai CI biên.
    Nếu test này đỏ thì `bootstrap_chenh_lech` đã không dùng CÙNG một lần lấy
    mẫu lại cho cả hai model — và khi đó nó không còn là phép thử theo cặp nữa.
    """
    rng = np.random.default_rng(11)
    a = _tao_pred(rng, 60, 0.55)
    b = a.copy()
    # B sửa đúng 12% số ảnh mà A đoán sai -> hơn A một cách NHẤT QUÁN
    sai = a.y_true.values != a.y_pred.values
    vi_tri = np.flatnonzero(sai)
    sua = rng.choice(vi_tri, size=int(len(vi_tri) * 0.12), replace=False)
    b.loc[sua, "y_pred"] = b.loc[sua, "y_true"]

    cap = bm.so_sanh_theo_cap("a", "b", a, b, n_bootstrap=400)
    rong_cap = cap["ci_hi"] - cap["ci_lo"]

    ytd = [ct.band_of(c) for c in a.sort_values("png").y_true]
    ci_a = bm.bootstrap_cum(ytd, [ct.band_of(c) for c in a.sort_values("png").y_pred],
                            a.sort_values("png").plant_id, ct.CLASSES_BAND, 400)
    ci_b = bm.bootstrap_cum(ytd, [ct.band_of(c) for c in b.sort_values("png").y_pred],
                            b.sort_values("png").plant_id, ct.CLASSES_BAND, 400)
    rong_bien = (ci_a["ci_hi"] - ci_a["ci_lo"]) + (ci_b["ci_hi"] - ci_b["ci_lo"])

    assert rong_cap < rong_bien, (
        f"CI của HIỆU rộng {rong_cap:.4f}, không hẹp hơn tổng hai CI biên "
        f"{rong_bien:.4f}. bootstrap_chenh_lech có thể đã dùng hai lần lấy mẫu "
        f"KHÁC nhau cho hai model — mất hết lợi thế của so sánh theo cặp."
    )


def test_cung_mot_model_thi_hieu_bang_0_va_khong_ket_luan_duoc() -> None:
    """So một model với chính nó: hiệu phải đúng 0 và KHÔNG được kết luận."""
    rng = np.random.default_rng(5)
    a = _tao_pred(rng, 40, 0.75)

    r = bm.so_sanh_theo_cap("a", "a2", a, a.copy(), n_bootstrap=300)
    assert r["hieu"] == 0.0, r["hieu"]
    assert r["ci_lo"] == 0.0 and r["ci_hi"] == 0.0, (r["ci_lo"], r["ci_hi"])
    assert not r["ket_luan_duoc"], (
        "So mot model voi chinh no ma bao 'ket luan duoc' — phep thu hong."
    )


# ------------------------------------------------- con số trên dữ liệu THẬT
#
# Hai test dưới tính lại từ file dự đoán của run 20260930-1450 và so với chữ
# trong tài liệu. Lỗi đã xảy ra: tài liệu ghi "CI biên: 1/15" (đúng ra là 1/5
# cặp liền kề) và "r = 0,951 giữa hai model đầu" (đúng ra là MobileViTv2-050 vs
# MobileNetV2). Cả hai đã đi vào báo cáo nộp giảng viên mà không gì báo.

from pathlib import Path  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

GOC = Path(__file__).resolve().parent.parent
RUN = "20260930-1450"


def _du_lieu_that():
    bang_p = GOC / "reports/bang_so_sanh.csv"
    if not bang_p.exists():
        pytest.skip("thiếu reports/bang_so_sanh.csv")
    bang = pd.read_csv(bang_p).sort_values("macroF1_5dai", ascending=False)
    pred = {}
    for arch, run, fw in zip(bang.arch, bang.run_id, bang.framework):
        p = GOC / "reports/predictions" / f"pred__{fw}__{arch}__{run}.csv"
        if not p.exists():
            pytest.skip(f"thiếu {p.name}")
        pred[arch] = pd.read_csv(p)
    ci = {a: json.loads(c) for a, c in zip(bang.arch, bang.macroF1_5dai_CI95)}
    return list(bang.arch), pred, ci


def _dem(cap: list[tuple[str, str]], ket_luan) -> str:
    return f"{sum(ket_luan(a, b) for a, b in cap)}/{len(cap)}"


def test_bang_quy_tac_18_khop_du_lieu_that() -> None:
    """Bảng số cặp kết luận được trong quy tắc 18 phải tính lại được từ dự đoán."""
    import itertools

    thu_tu, pred, ci = _du_lieu_that()
    san = "logreg_gray32"
    lien_ke = list(zip(thu_tu, thu_tu[1:]))
    moi_cap = list(itertools.combinations(thu_tu, 2))
    sau = [(a, b) for a, b in moi_cap if san not in (a, b)]

    bien = lambda a, b: ci[a][0] > ci[b][1] or ci[b][0] > ci[a][1]  # noqa: E731
    kq_cap = {(a, b): bm.so_sanh_theo_cap(a, b, pred[a], pred[b])["ket_luan_duoc"]
              for a, b in moi_cap}
    theo_cap = lambda a, b: kq_cap[(a, b)]  # noqa: E731

    tinh = {
        "bien": [_dem(lien_ke, bien), _dem(moi_cap, bien), _dem(sau, bien)],
        "cap": [_dem(lien_ke, theo_cap), _dem(moi_cap, theo_cap), _dem(sau, theo_cap)],
    }

    text = (GOC / "Docs/Quy_tac_thiet_ke.md").read_text(encoding="utf-8")
    dong = {
        "bien": re.search(r"^\| CI95 \*\*biên\*\*.*$", text, re.M),
        "cap": re.search(r"^\| CI95 của \*\*HIỆU\*\*.*$", text, re.M),
    }
    for k, m in dong.items():
        assert m, f"quy tắc 18 thiếu dòng bảng cho phép thử '{k}'"
        ghi = re.findall(r"\d+/\d+", m.group(0))
        assert ghi == tinh[k], (
            f"quy tắc 18, phép thử '{k}': tài liệu ghi {ghi}, tính lại được {tinh[k]} "
            "(cặp liền kề · mọi cặp · cặp giữa 5 kiến trúc). Sửa tài liệu, đừng sửa test."
        )


def test_tuong_quan_fold_trong_ket_qua_benchmark_dung_cap_model() -> None:
    """Câu "r = …" trong Ket_qua_benchmark.md phải đúng giá trị VÀ đúng cặp model."""
    _thu_tu, pred, _ci = _du_lieu_that()
    ten = {"mobilevitv2_050": "MobileViTv2-050", "mobilenetv2": "MobileNetV2",
           "resnet18": "ResNet18", "effnet_lite0": "EfficientNet-Lite0",
           "effnet_b0": "EfficientNet-B0"}

    def f1_fold(a):
        return [bm.macro_f1(g.y_true, g.y_pred) for _, g in pred[a].groupby("fold")]

    text = (GOC / "Docs/Ket_qua_benchmark.md").read_text(encoding="utf-8")
    doan = re.search(r"### Theo fold(.*?)\n---", text, re.S)
    assert doan, "Ket_qua_benchmark.md thiếu mục 'Theo fold'"
    doan = re.split(r"\*Đính chính", doan.group(1))[0]  # đính chính được phép nhắc số cũ
    tim = re.findall(r"r = (\d,\d{3})\s+giữa\s+(\S+)\s+và\s+(\S+?)[\s,.)]", doan)
    tim += [(r, a, b) for a, b, r in re.findall(r"giữa\s+(\S+)\s+và\s+(\S+)\s+chỉ là\s+(\d,\d{3})", doan)]
    assert tim, "mục 'Theo fold' không còn câu nào dạng 'r = x,xxx giữa A và B'"
    nguoc = {v: k for k, v in ten.items()}
    for r_ghi, a, b in tim:
        a, b = nguoc[a], nguoc[b]
        r = float(np.corrcoef(f1_fold(a), f1_fold(b))[0, 1])
        assert f"{r:.3f}".replace(".", ",") == r_ghi, (
            f"tài liệu ghi r = {r_ghi} giữa {a} và {b}, tính lại được {r:.3f}")


def test_holm_khop_tinh_tay_giu_thu_tu_va_tu_choi_nan() -> None:
    from coffeeguard.bench import metrics as bm
    p = np.array([0.04, 0.01, 0.03, 0.20])
    # sắp tăng: 0,01·4 = 0,04; 0,03·3 = 0,09; 0,04·2 = 0,08 -> max tích luỹ 0,09; 0,20·1
    assert bm.holm(p) == pytest.approx([0.09, 0.04, 0.09, 0.20])
    assert bm.holm([0.5, 0.6]).max() <= 1.0
    with pytest.raises(ValueError):
        bm.holm([0.01, np.nan])
