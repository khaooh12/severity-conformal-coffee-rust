"""Kiểm tính hợp lệ của CAM — BẢN ĐỊNH NGHĨA DUY NHẤT, ship lên Colab.

Vì sao module này nằm trong `src/coffeeguard/bench/` chứ không phải `scripts/`
----------------------------------------------------------------------------
`coffeeguard_src.zip` (gói mang lên Colab) chỉ chứa `src/`. Đặt phép kiểm ở
`scripts/` thì nó KHÔNG BAO GIỜ chạy ở nơi thật sự train — tức là chỗ duy nhất
mà kết quả kiểm có ý nghĩa, vì cấu trúc đầu model phụ thuộc phiên bản `timm`
mà Colab cài. Môi trường chính của đồ án là Colab, nên phép kiểm phải đi cùng.

Điều kiện CAM hợp lệ
--------------------
`onnxruntime-web` chỉ suy luận, không có gradient → Grad-CAM không chạy được
trong trình duyệt. Thay bằng CAM gốc (Zhou 2016), nhúng sẵn vào graph ONNX. CAM
chỉ đúng khi tồn tại bản đồ `G` đi vào GAP sao cho:

    logits_c == mean_{x,y} [ W_c · G[:,x,y] + b_c ]

Thoả thì `CAM_c(x,y) = W_c · G[:,x,y]` cộng lại ra ĐÚNG logits — cái PWA tô màu
chính là cái model dùng để quyết định. Không thoả thì heatmap vẫn vẽ ra, vẫn
trông hợp lý, và KHÔNG CÓ GÌ BÁO.

⚠ HAI CÁCH KIỂM SAI, đều đã mắc phải rồi sửa (30/09/2026)
----------------------------------------------------------
1. **Kiểm bằng tên module / số kênh.** `timm` để `classifier` là `nn.Linear` ở
   cả MobileNetV3 nên nhìn tên thì "đạt", dù có `act2` (hardswish) NẰM GIỮA
   pooling và classifier. So `feature.shape[1] == head.in_features` bắt được
   MobileNetV3 (576 ≠ 1024) nhưng **cho DƯƠNG TÍNH GIẢ** với họ ConvNeXt và
   EdgeNeXt: số kênh khớp, lớp cuối đúng là `nn.Linear`, nhưng có chuẩn hoá toàn
   cục sau pooling. Đã đo: `convnext_tiny` sai số 1,8e-01 mà kiểm cấu trúc vẫn
   nói "có CAM".
2. **Đo sai số theo `max|logits|`.** Ở khởi tạo ngẫu nhiên, độ lớn logits tuỳ
   tiện → `mobileone_s1` lệch 2,75 vẫn "đạt" còn `mobilenetv3_small_100` lệch
   1,9e-05 lại "loại".

→ Cách đúng: phép thử **cộng tính** với metric **không thang đo**.
"""

from __future__ import annotations

from typing import Callable

NGUONG = 1e-5
"""Sai số cộng tính tối đa còn coi là affine.

Đo thực tế trên 33 kiến trúc: nhóm thoả nằm ở 1e-7–1e-8, nhóm vỡ từ 1e-4 trở
lên. Khoảng trống giữa hai nhóm rộng ba bậc độ lớn nên ngưỡng này không nhạy
cảm — đừng nới nó ra để "cho một kiến trúc đi qua".
"""

KICH_THUOC_KIEM = 224


class _tam_thoi_eval:
    """Đưa model về ``eval()`` trong lúc kiểm, rồi trả lại chế độ cũ.

    ⚠ BẮT BUỘC, không phải cho gọn. Ở chế độ train, BatchNorm dùng **thống kê
    theo batch** — tức phi tuyến theo đầu vào — và Dropout thì ngẫu nhiên. Phép
    thử cộng tính sẽ vỡ ở những kiến trúc có chuẩn hoá trong đường head, cho
    **ÂM TÍNH GIẢ**: kiến trúc hợp lệ bị báo là không thoả.

    Đã mắc thật (30/09/2026): `ModelCAM` ở notebook 02 tạo model rồi kiểm ngay,
    `timm.create_model` trả model ở chế độ **train**, nên `mobilevitv2_050` bị
    báo sai số 1,4e-01 trong khi `kiem_ten` (có `.eval()`) đo được 4,0e-08. Hai
    phép kiểm trong CÙNG một ô nói trái nhau.
    """

    def __init__(self, model):
        self.model = model
        self.dang_train = None

    def __enter__(self):
        self.dang_train = self.model.training
        self.model.eval()
        return self.model

    def __exit__(self, *_):
        self.model.train(self.dang_train)
        return False


def kich_thuoc_goc(model, mac_dinh: int = KICH_THUOC_KIEM) -> int:
    """Kích thước đầu vào GỐC của model theo `timm`, không đoán 224.

    `mobilevitv2_050` chạy 256, `mobilevit_s` 256... Kiểm ở kích thước khác
    kích thước thật thì bản đồ đặc trưng có hình khác, và với kiến trúc có
    attention theo cửa sổ thì kết luận có thể lệch.
    """
    try:
        import timm

        cfg = timm.data.resolve_model_data_config(model)
        return int(cfg["input_size"][-1])
    except Exception:
        return mac_dinh


def sai_so_cong_tinh(
    model,
    head_fn: Callable,
    kich_thuoc: int | None = None,
    seed: int | None = 0,
) -> float:
    """Sai số tương đối của phép thử cộng tính trên ánh xạ ``head_fn``.

    Với mọi ánh xạ affine thì ``f(a+b) + f(0) == f(a) + f(b)``. Sai số đo bằng
    ``‖lhs-rhs‖ / (‖lhs‖+‖rhs‖)`` nên **không phụ thuộc độ lớn** của logits.

    ``head_fn`` nhận bản đồ đặc trưng 4D và trả logits. ``kich_thuoc=None`` thì
    lấy kích thước GỐC của model. Model luôn được đưa về ``eval()`` khi kiểm.
    """
    import torch

    k = kich_thuoc or kich_thuoc_goc(model)
    if seed is not None:
        torch.manual_seed(seed)
    with _tam_thoi_eval(model), torch.no_grad():
        f0 = model.forward_features(torch.randn(1, 3, k, k))
        f1 = model.forward_features(torch.randn(1, 3, k, k))
        lhs = head_fn(f0 + f1) + head_fn(torch.zeros_like(f0))
        rhs = head_fn(f0) + head_fn(f1)
        chia = lhs.norm().item() + rhs.norm().item()
        return (lhs - rhs).norm().item() / max(1e-12, chia)


def sai_so_dung_thuc_cam(
    model,
    lay_G: Callable,
    kich_thuoc: int | None = None,
    seed: int | None = 0,
) -> float:
    """Kiểm TRỰC TIẾP đẳng thức CAM trên bản đồ ``G`` thực sự đi vào GAP.

    Trả sai số tương đối giữa ``mean_xy(CAM) + b`` và ``logits``. Đây là phép
    kiểm mạnh nhất: nó đo đúng cái PWA sẽ vẽ.
    """
    import torch

    kich_thuoc = kich_thuoc or kich_thuoc_goc(model)
    if seed is not None:
        torch.manual_seed(seed)
    with _tam_thoi_eval(model), torch.no_grad():
        f = model.forward_features(torch.randn(1, 3, kich_thuoc, kich_thuoc))
        g = lay_G(model, f)
        lop = model.get_classifier()
        w, b = lop.weight, lop.bias
        logits = (g.mean(dim=(2, 3)) @ w.T) + b
        cam = torch.einsum("bchw,kc->bkhw", g, w)
        tu_cam = cam.mean(dim=(2, 3)) + b
        lech = (tu_cam - logits).abs().max().item()
        return lech / max(1e-12, logits.abs().max().item())


def kiem_cam(model, kich_thuoc: int | None = None) -> dict:
    """Kiểm một model timm đã tạo. Trả dict, KHÔNG ném lỗi.

    An toàn khi gọi trên model đang ở chế độ **train** (`timm.create_model` trả
    về như vậy): hàm tự đưa về `eval()` khi kiểm rồi trả lại chế độ cũ. Nếu
    không thì BatchNorm dùng thống kê theo batch và phép thử cho ÂM TÍNH GIẢ.

    ``kich_thuoc=None`` thì lấy kích thước đầu vào GỐC của model.

    Khoá trả về:
      ``co_cam``      bool — thoả nguyên bản, dùng được luôn
      ``sai_so``      float | None
      ``kenh``        int | None — số kênh của bản đồ đi vào GAP
      ``kich_thuoc``  int — kích thước đã dùng để kiểm
      ``cuu_duoc``    bool — không thoả nguyên bản nhưng đổi thứ tự đầu thì thoả
      ``ly_do``       str — vì sao không thoả (rỗng nếu thoả)
    """
    import torch
    import torch.nn as nn

    kich_thuoc = kich_thuoc or kich_thuoc_goc(model)
    kq = {"co_cam": False, "sai_so": None, "kenh": None,
          "kich_thuoc": kich_thuoc, "cuu_duoc": False, "ly_do": ""}

    lop = model.get_classifier()
    if not isinstance(lop, nn.Linear):
        kq["ly_do"] = f"lop cuoi la {type(lop).__name__}, khong phai nn.Linear"
        return kq

    with _tam_thoi_eval(model), torch.no_grad():
        f = model.forward_features(torch.zeros(1, 3, kich_thuoc, kich_thuoc))
    if f.ndim != 4:
        kq["ly_do"] = f"forward_features tra ve {tuple(f.shape)}, khong phai [B,C,H,W] (ViT?)"
        return kq
    kq["kenh"] = int(f.shape[1])

    e = sai_so_cong_tinh(model, lambda x: model.forward_head(x), kich_thuoc)
    kq["sai_so"] = e
    if e <= NGUONG:
        kq["co_cam"] = True
        return kq

    kq["ly_do"] = (f"phi tuyen giua GAP va classifier (sai so cong tinh {e:.2e} "
                   f"> nguong {NGUONG:.0e})")

    # Thử cứu: đặt conv_head + act TRƯỚC GAP. Chỉ họ EfficientNet/MobileNet có.
    if hasattr(model, "conv_head") and hasattr(model, "act2"):
        try:
            e2 = sai_so_dung_thuc_cam(
                model, lambda m, x: m.act2(m.conv_head(x)), kich_thuoc
            )
            if e2 <= NGUONG:
                kq["cuu_duoc"] = True
                kq["ly_do"] += (f"; CUU DUOC bang cach dat conv_head+act TRUOC GAP "
                                f"(sai so {e2:.2e}) — NHUNG thanh mot kien truc KHAC "
                                f"voi ban cong bo, khong duoc trich accuracy ban goc")
        except Exception:  # pragma: no cover — kiến trúc có head lạ
            pass
    return kq


def kiem_ten(ten_timm: str, so_lop: int = 6, kich_thuoc: int | None = None) -> dict:
    """Tạo model theo tên timm rồi kiểm. Thêm khoá ``ten`` và ``params_M``.

    ``kich_thuoc=None`` thì kiểm ở kích thước đầu vào GỐC của kiến trúc.
    """
    import timm
    import torch

    kq = {"ten": ten_timm, "params_M": None, "co_cam": False, "sai_so": None,
          "kenh": None, "kich_thuoc": None, "cuu_duoc": False, "ly_do": ""}
    torch.manual_seed(0)
    try:
        m = timm.create_model(ten_timm, pretrained=False, num_classes=so_lop).eval()
    except Exception as e:
        kq["ly_do"] = f"khong tao duoc model: {type(e).__name__}: {e}"
        return kq
    kq["params_M"] = sum(q.numel() for q in m.parameters()) / 1e6
    kq.update(kiem_cam(m, kich_thuoc))
    return kq


def chan_neu_mat_cam(ten_kien_truc, so_lop: int = 6) -> dict:
    """Kiểm cả danh sách và NÉM LỖI nếu có kiến trúc mất CAM.

    Dùng ở đầu notebook 02 trên Colab: thà dừng ngay còn hơn train 5 fold rồi mới
    phát hiện bản đồ nhiệt vô nghĩa. Trả dict {ten: ket_qua} khi mọi thứ đều thoả.
    """
    ket = {t: kiem_ten(t, so_lop) for t in ten_kien_truc}
    vo = {t: r for t, r in ket.items() if not r["co_cam"]}
    if vo:
        dong = "\n".join(
            f"  - {t}: {r['ly_do'] or 'khong ro'}" for t, r in vo.items()
        )
        raise RuntimeError(
            "Cac kien truc sau KHONG thoa dang thuc CAM, ban do nhiet se SAI ma "
            "khong bao loi:\n" + dong +
            "\n\nDoi kien truc, hoac neu co y dinh dung ban 'doi thu tu dau' thi "
            "phai khai bao ro va KHONG duoc trich accuracy cua ban cong bo goc.\n"
            "Xem Docs/Ke_hoach_benchmark.md §4."
        )
    return ket
