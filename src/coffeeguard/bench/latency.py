"""Đo tốc độ suy luận — phần dễ đo sai nhất trong cả đồ án.

GVHD muốn thấy **FPS trên thiết bị di động**, nên bảng tốc độ sẽ được đọc kỹ.
Ba cái bẫy, cả ba đều cho ra số trông hợp lý:

1. **Không warmup.** Lần suy luận đầu gồm khởi tạo lazy, cấp phát bộ đệm, biên
   dịch kernel — chậm gấp nhiều lần. Không warmup thì bảng "tốc độ" phản ánh
   THỨ TỰ NẠP MODEL chứ không phải kiến trúc.
2. **Lấy trung bình thay vì trung vị.** Một lần GC hay một lần OS cướp CPU là đủ
   kéo trung bình lệch hẳn. Dùng p50; báo thêm p90 để thấy đuôi.
3. **Không cố định số luồng.** `torch.set_num_threads()` mặc định lấy hết lõi,
   và con số đó khác nhau giữa laptop, Colab, điện thoại. So bốn model mà mỗi
   model chạy số luồng khác nhau là so lịch chạy của hệ điều hành.

Con số ở đây là **latency trên MÁY ĐANG CHẠY**, không phải trên điện thoại. Đo
trên điện thoại thật phải làm trong PWA bằng `performance.now()` quanh
`session.run()` — xem `web/src/lib/infer.js`, trường `latencyMs`.
"""

from __future__ import annotations

import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Callable

WARMUP_MAC_DINH = 20
LAN_CHAY_MAC_DINH = 100


@dataclass
class KetQuaLatency:
    ten: str
    n_warmup: int
    n_chay: int
    so_luong: int | None
    p50_ms: float
    p90_ms: float
    p99_ms: float
    trung_binh_ms: float
    do_lech_ms: float
    nhanh_nhat_ms: float
    fps_p50: float
    thiet_bi: str
    ghi_chu: str = ""

    def dict(self) -> dict:
        return asdict(self)


def mo_ta_may() -> str:
    return (
        f"{platform.system()} {platform.machine()} / "
        f"py{platform.python_version()} / {platform.processor() or 'cpu?'}"
    )


def co_dinh_luong(n: int = 1) -> int | None:
    """Cố định số luồng torch. Trả về số luồng thực đặt được, None nếu chưa có torch."""
    try:
        import torch
    except ImportError:
        return None
    torch.set_num_threads(n)
    try:
        torch.set_num_interop_threads(n)
    except RuntimeError:
        pass  # chỉ đặt được một lần mỗi tiến trình — lần sau bỏ qua là đúng
    return torch.get_num_threads()


def do(
    ham: Callable[[], object],
    ten: str,
    n_warmup: int = WARMUP_MAC_DINH,
    n_chay: int = LAN_CHAY_MAC_DINH,
    so_luong: int | None = 1,
    ghi_chu: str = "",
) -> KetQuaLatency:
    if n_warmup < 5:
        raise ValueError(
            f"n_warmup={n_warmup} qua it. Lan suy luan dau cham gap nhieu lan vi "
            "khoi tao lazy; do khong warmup se cho ra bang 'toc do' phan anh thu "
            "tu nap model. Toi thieu 5, nen dung 20."
        )
    luong_thuc = co_dinh_luong(so_luong) if so_luong else None

    for _ in range(n_warmup):
        ham()

    mau: list[float] = []
    for _ in range(n_chay):
        t0 = time.perf_counter()
        ham()
        mau.append((time.perf_counter() - t0) * 1000.0)

    mau_sx = sorted(mau)
    p = lambda q: mau_sx[min(len(mau_sx) - 1, int(round(q * (len(mau_sx) - 1))))]
    p50 = statistics.median(mau)
    return KetQuaLatency(
        ten=ten,
        n_warmup=n_warmup,
        n_chay=n_chay,
        so_luong=luong_thuc,
        p50_ms=round(p50, 3),
        p90_ms=round(p(0.90), 3),
        p99_ms=round(p(0.99), 3),
        trung_binh_ms=round(statistics.fmean(mau), 3),
        do_lech_ms=round(statistics.pstdev(mau), 3),
        nhanh_nhat_ms=round(mau_sx[0], 3),
        fps_p50=round(1000.0 / p50, 2) if p50 > 0 else float("inf"),
        thiet_bi=mo_ta_may(),
        ghi_chu=ghi_chu,
    )


def do_model_torch(model, dau_vao, ten: str, **kw) -> KetQuaLatency:
    """Đo một `nn.Module` ở chế độ eval + no_grad, batch 1."""
    import torch

    model.eval()
    with torch.inference_mode():
        return do(lambda: model(dau_vao), ten=ten, **kw)


def do_onnx(session, feeds: dict, ten: str, **kw) -> KetQuaLatency:
    """Đo một `onnxruntime.InferenceSession`."""
    return do(lambda: session.run(None, feeds), ten=ten, **kw)


def dem_tham_so(model) -> dict:
    tong = sum(p.numel() for p in model.parameters())
    return {
        "params": int(tong),
        "params_M": round(tong / 1e6, 3),
        "size_MB_fp32": round(tong * 4 / 1048576, 2),
    }
