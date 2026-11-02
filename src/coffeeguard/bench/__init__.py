"""Tầng benchmark — so sánh nhiều kiến trúc trên CÙNG một split đã đóng băng.

Đọc `Docs/Ke_hoach_benchmark.md` trước khi sửa bất cứ file nào trong gói này.
Mỗi module ở đây canh một lỗi **im lặng**: code vẫn chạy, chỉ số vẫn đẹp, bảng
so sánh vẫn ra — nhưng so sai thứ.
"""

from coffeeguard.bench.contract import (
    BAND_OF,
    CLASSES_6,
    CLASSES_BAND,
    PRED_COLUMNS,
    band_of,
)

__all__ = ["CLASSES_6", "CLASSES_BAND", "BAND_OF", "PRED_COLUMNS", "band_of"]
