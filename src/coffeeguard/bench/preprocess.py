"""Giải mã + cắt + thu nhỏ ảnh MỘT LẦN cho tất cả framework.

Vì sao không để mỗi framework tự resize: khi đó ta đang so **thư viện resize**
chứ không so kiến trúc. `tf.image.resize` mặc định `antialias=False` còn PIL thì
có; `tf.io.decode_jpeg` giải DCT khác libjpeg của PIL. Cả hai chênh vài phần
trăm accuracy và **không gây lỗi** — chỉ làm một framework trông kém hơn.

Nên ở đây: giải mã + cắt + thu nhỏ bằng PIL, ghi **PNG không mất mát 224x224**.
Mọi framework sau đó đọc đúng cùng một mảng pixel.

Hai chế độ cắt
--------------
`letterbox`  chèn viền đen cho cả ảnh thành vuông rồi thu nhỏ.
             Lá chỉ chiếm **trung vị 14,0%** pixel — vứt 86% khung cho nền đất.
`leafsquare` cửa sổ VUÔNG quanh tâm khung bao lá, cạnh = `scale * neo`.

NEO VÀO CẠNH NÀO — chỗ này làm sai thì vẫn chạy trơn tru
--------------------------------------------------------
Lá cà phê dài: khung bao trung vị **2,74:1**. Nên ô vuông "bao trọn lá" (neo vào
cạnh DÀI) hoá ra lại chủ yếu là nền, đo thật trên cả 1.559 ảnh:

    neo="dai",  scale 1.4  ->   9,5 / 11,6 / 14,1 %   TỆ HƠN cả letterbox
    neo="ngan", scale 1.4  ->  56   / 61   / 64   %
    neo="ngan", scale 1.0  ->  82   / 89   / 92   %

Mặc định `neo="ngan"`, `scale=1.4`.

**Cái giá phải trả, phải nêu trong báo cáo:** neo vào cạnh ngắn nghĩa là **cắt
mất hai đầu lá** — ô vuông chỉ dài bằng ~0,5 chiều dài lá, nên vết bệnh sát
chóp lá có thể không lọt khung. Đổi lại mỗi vết bệnh còn trong khung được nhiều
pixel hơn ~5 lần. Đây là **đánh đổi**, không phải cải tiến thuần.

Chính vì thế `letterbox` phải được chạy làm **ablation**. Không có nó thì "cửa
sổ vuông giúp ích" mới chỉ là một phép đếm pixel, chưa phải kết quả.

Lưu ý lệch phân phối train/serve
--------------------------------
RoCoLe: lá chiếm ~25% khung gốc. Nông dân được PWA dặn "đưa lá vào khung": 70%+.
Cửa sổ vuông kéo lúc train về gần lúc chạy thật. Phần còn lại giải bằng
augmentation lúc train: `RandomResizedCrop` đa tỉ lệ + dịch chuyển, nhờ vậy qua
nhiều epoch ô vuông rơi vào những đoạn khác nhau của lá và model vẫn thấy hết
lá dù mỗi lần chỉ thấy một đoạn.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from coffeeguard.bench import contract as ct
from coffeeguard.bench import paths as P

Image.MAX_IMAGE_PIXELS = None
RESAMPLE = Image.LANCZOS
MAU_VIEN = (0, 0, 0)


def ten_thu_muc(che_do: str, size: int, scale: float, neo: str = "ngan") -> str:
    if che_do == "leafsquare":
        return f"leafsquare-{neo}{scale:g}x_{size}"
    return f"{che_do}_{size}"


def cua_so_vuong(
    leaf_x: float,
    leaf_y: float,
    leaf_w: float,
    leaf_h: float,
    scale: float,
    neo: str = "ngan",
) -> tuple[int, int, int]:
    """-> (x0, y0, canh) của cửa sổ vuông, toạ độ ảnh gốc (có thể âm / tràn ra ngoài).

    `neo="ngan"` lấy cạnh NGẮN của khung bao làm chuẩn — ô vuông khi đó chủ yếu
    là lá. `neo="dai"` bao trọn lá nhưng với lá dài 2,74:1 thì ô vuông toàn nền,
    tệ hơn cả letterbox. Xem docstring đầu file.
    """
    if neo not in ("ngan", "dai"):
        raise ValueError(f"neo phai la 'ngan' hoac 'dai', nhan '{neo}'")
    cx = leaf_x + leaf_w / 2.0
    cy = leaf_y + leaf_h / 2.0
    goc = min(leaf_w, leaf_h) if neo == "ngan" else max(leaf_w, leaf_h)
    canh = max(int(round(goc * scale)), 1)
    return int(round(cx - canh / 2.0)), int(round(cy - canh / 2.0)), canh


def cat_vuong(im: Image.Image, x0: int, y0: int, canh: int) -> Image.Image:
    """Cắt cửa sổ vuông, phần tràn ra ngoài ảnh thì chèn viền đen.

    Cố ý KHÔNG co cửa sổ lại cho vừa ảnh: co lại sẽ đổi tỉ lệ lá/khung ở đúng
    những ảnh lá nằm sát mép, tức là đưa thêm một biến vào dữ liệu.
    """
    nen = Image.new("RGB", (canh, canh), MAU_VIEN)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(im.width, x0 + canh), min(im.height, y0 + canh)
    if sx1 > sx0 and sy1 > sy0:
        nen.paste(im.crop((sx0, sy0, sx1, sy1)), (sx0 - x0, sy0 - y0))
    return nen


def letterbox(im: Image.Image) -> Image.Image:
    canh = max(im.size)
    nen = Image.new("RGB", (canh, canh), MAU_VIEN)
    nen.paste(im, ((canh - im.width) // 2, (canh - im.height) // 2))
    return nen


@dataclass
class KetQuaXuLy:
    dich: Path
    manifest: dict
    df: pd.DataFrame


def ti_le_la_trong_cua_so(
    leaf_poly: str, x0: int, y0: int, canh: int, do_phan_giai: int = 224
) -> float | None:
    """% pixel là lá THỰC SỰ nằm trong cửa sổ — rasterize đa giác, không ước lượng.

    Không dùng `leaf_area_px / canh**2` được: cửa sổ neo cạnh ngắn **cắt mất hai
    đầu lá**, nên diện tích cả chiếc lá không còn là tử số đúng và tỉ lệ tính ra
    sẽ lớn hơn sự thật (có ảnh vượt 100%). Đó đúng là kiểu sai không ai phát
    hiện: con số vẫn in ra đẹp và vẫn đi thẳng vào báo cáo.
    """
    try:
        diem = json.loads(leaf_poly) if isinstance(leaf_poly, str) else []
    except (TypeError, ValueError):
        return None
    if len(diem) < 6 or canh <= 0:
        return None

    k = do_phan_giai / canh
    xy = [((diem[i] - x0) * k, (diem[i + 1] - y0) * k) for i in range(0, len(diem) - 1, 2)]
    # Mode "L" + fill=255 + đếm khác 0.
    # KHÔNG dùng mode "1" với `histogram()[-1]`: Pillow ghi thẳng giá trị ink vào
    # bộ đệm 8-bit, nên `fill=1` rơi vào bin 1 chứ không phải bin 255, và hàm này
    # trả về 0,0% cho MỌI ảnh. Mà 0% trông y hệt "lá nằm ngoài cửa sổ" — ảnh PNG
    # vẫn đúng, chỉ con số trong báo cáo là sai.
    mask = Image.new("L", (do_phan_giai, do_phan_giai), 0)
    ImageDraw.Draw(mask).polygon(xy, fill=255)
    return float(np.count_nonzero(np.asarray(mask))) / float(do_phan_giai * do_phan_giai)


def _xu_ly_mot(row, src_dir, dst_dir, che_do, size, scale, neo):
    dst = dst_dir / row.png
    if che_do == "leafsquare":
        x0, y0, canh = cua_so_vuong(row.leaf_x, row.leaf_y, row.leaf_w, row.leaf_h, scale, neo)
    else:
        # Phải khớp ĐÚNG offset mà `letterbox()` dán ảnh vào, kể cả làm tròn:
        # lệch một pixel ở đây làm con số "% pixel là lá" sai một cách vô hình.
        canh = max(int(row.img_w), int(row.img_h))
        x0 = -((canh - int(row.img_w)) // 2)
        y0 = -((canh - int(row.img_h)) // 2)
    frac = ti_le_la_trong_cua_so(getattr(row, "leaf_poly", ""), x0, y0, canh)

    if dst.exists():
        return row.png, frac, "bo_qua"
    with Image.open(src_dir / row.file) as im:
        im = im.convert("RGB")
        vuong = cat_vuong(im, x0, y0, canh) if che_do == "leafsquare" else letterbox(im)
        vuong.resize((size, size), RESAMPLE).save(dst, format="PNG", optimize=True)
    return row.png, frac, "moi"


def xu_ly(
    df_index: pd.DataFrame,
    che_do: str = "leafsquare",
    size: int = ct.IMG_SIZE,
    scale: float = ct.LEAF_SCALE,
    neo: str = ct.LEAF_NEO,
    src_dir: Path = P.ROCOLE_PHOTOS,
    goc_dich: Path = P.PROCESSED,
    workers: int = 8,
    tien_do=print,
) -> KetQuaXuLy:
    if che_do not in ct.CHE_DO_CAT:
        raise ValueError(f"che_do phai thuoc {ct.CHE_DO_CAT}, nhan '{che_do}'")
    if "leaf_poly" not in df_index.columns:
        raise ValueError(
            "Index thieu cot 'leaf_poly' -> khong do duoc ti le pixel la. "
            "Chay lai:  python scripts/bench.py index"
        )
    dst_dir = goc_dich / ten_thu_muc(che_do, size, scale, neo)
    dst_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fracs: dict[str, float] = {}
    n_moi = n_bo = 0
    rows = list(df_index.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [
            ex.submit(_xu_ly_mot, r, src_dir, dst_dir, che_do, size, scale, neo)
            for r in rows
        ]
        for i, fu in enumerate(futs, 1):
            png, frac, tt = fu.result()
            if frac is not None:
                fracs[png] = frac
            n_moi += tt == "moi"
            n_bo += tt == "bo_qua"
            if i % 200 == 0:
                tien_do(f"   {i}/{len(rows)}")

    df = df_index.copy()
    df["leaf_frac_cua_so"] = df.png.map(fracs)

    co = fracs and pd.Series(list(fracs.values()))
    manifest = {
        "schema_version": "1.0",
        "tao_luc": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "che_do_cat": che_do,
        "leaf_scale": scale if che_do == "leafsquare" else None,
        "leaf_neo": neo if che_do == "leafsquare" else None,
        "img_size": size,
        "resample": "LANCZOS",
        "dinh_dang": "PNG (khong mat mat)",
        "mau_vien": list(MAU_VIEN),
        "pillow_version": Image.__version__,
        "n_anh": int(len(df_index)),
        "n_ghi_moi": n_moi,
        "n_bo_qua_da_co": n_bo,
        "giay": round(time.time() - t0, 1),
        "ti_le_pixel_la": (
            {
                "p25": round(float(co.quantile(0.25)), 4),
                "trung_vi": round(float(co.median()), 4),
                "p75": round(float(co.quantile(0.75)), 4),
            }
            if fracs
            else None
        ),
        "ghi_chu": (
            "Moi framework DOC thu muc nay, khong tu resize tu anh goc. "
            "Tu resize = dang so thu vien resize chu khong so kien truc."
        ),
    }
    (dst_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return KetQuaXuLy(dich=dst_dir, manifest=manifest, df=df)
