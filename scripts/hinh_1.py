#!/usr/bin/env python
"""Hinh 1 cua bai: do phu theo dai that, conformal bien vs theo dai.

MobileViTv2-050 (luot 2), diem 1 - p, alpha = 0,20, so do nua cay. Doc tu
reports/conformal_doi_chung.csv (sinh bang `python scripts/bench.py hieu-chinh`).

    python scripts/hinh_1.py            # -> reports/hinh_1.png
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coffeeguard.bench import contract as ct
from coffeeguard.bench import paths as P

MODEL = "mobilevitv2_050@20261022-1539-luotA"
ALPHA = 0.20
NHAN_DAI = {"healthy": "Khoẻ", "red_spider_mite": "Nhện đỏ", "rust_nhe": "Nhẹ",
            "rust_trung_binh": "TB", "rust_nang": "Nặng"}


def _so(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def main() -> int:
    d = pd.read_csv(P.REPORTS / "conformal_doi_chung.csv")
    q = d[(d.model == MODEL) & np.isclose(d.alpha, ALPHA) & (d.diem == "thr")
          & (d.hieu_chinh == "trong_fold")].set_index("che_do")
    cot = [f"phu__{b}" for b in ct.CLASSES_BAND]
    bien, theo_dai = q.loc["bien", cot].to_numpy(float), q.loc["theo_dai", cot].to_numpy(float)

    plt.rcParams.update({"font.family": "serif", "font.size": 10})
    fig, ax = plt.subplots(figsize=(5.0, 3.1), dpi=400)
    x, w = np.arange(len(cot)), 0.38
    for dx, v, mau, ten in ((-w / 2, bien, "#bdbdbd", "Biên"), (w / 2, theo_dai, "#4472c4", "Theo dải")):
        cot_ve = ax.bar(x + dx, v, w, color=mau, edgecolor="black", linewidth=0.6, label=ten)
        for r, gt in zip(cot_ve, v):
            ax.text(r.get_x() + r.get_width() / 2, gt + 0.005, _so(gt), ha="center", va="bottom", fontsize=8)
    ax.axhline(1 - ALPHA, color="red", linestyle="--", linewidth=1, label=f"Mục tiêu {_so(1 - ALPHA)}")
    ax.set_xticks(x, [NHAN_DAI[b] for b in ct.CLASSES_BAND])
    ax.set_ylim(0, 1.08)
    ax.set_yticks(np.arange(0, 1.01, 0.2), [_so(t) for t in np.arange(0, 1.01, 0.2)])
    ax.set_ylabel("Độ phủ")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    dich = P.REPORTS / "hinh_1.png"
    fig.savefig(dich)
    print(f"  da ghi {dich}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
