# Tập dự đoán conformal theo mức độ cho chẩn đoán bệnh gỉ sắt lá cà phê

Mã nguồn, tệp chia fold và dữ liệu kết quả đi kèm bài báo:

> Trần Trọng Khang, Phan Thị Hường. *Tập dự đoán conformal theo mức độ cho chẩn đoán bệnh gỉ sắt lá cà phê.*
> Khoa Công nghệ Thông tin, Trường Đại học Lạc Hồng, 2026 (bản thảo gửi Tạp chí Khoa học Lạc Hồng).

Bài kiểm tra cách "chỉ trả lời khi độ tin cậy ≥ một ngưỡng chung" trên bài toán phân loại năm dải (lá khoẻ,
nhện đỏ, gỉ sắt nhẹ, trung bình, nặng) của bộ RoCoLe (1.559 ảnh lá cà phê vối), so sánh hiệu chỉnh nhiệt độ,
vector scaling, Dirichlet calibration, ngưỡng theo dải và dự đoán conformal (biên / theo dải; điểm `1 − p`,
APS, RAPS, thứ bậc), trên dự đoán ngoài fold của chín mô hình nhẹ (cộng ba ResNet ở Bảng 12) huấn luyện
bằng kiểm định chéo 5 fold chia theo cây.

*English summary:* code, frozen plant-grouped 5-fold split and out-of-fold predictions that reproduce every
table of the paper "Severity-wise conformal prediction sets for coffee leaf rust diagnosis" (RoCoLe dataset).
All tables can be regenerated on a CPU in a few minutes from the files in `reports/`, without images or GPU.

## Có gì trong repo

```
data/
  interim/rocole_index.csv       chỉ mục 1.559 ảnh: cây, lá, mặt lá, nhãn 6 lớp, khung bao lá
  splits/rocole_folds.json       5 fold chia theo cây — ĐÓNG BĂNG, mọi mô hình dùng chung
reports/
  predictions/                   dự đoán ngoài fold, lượt 1 (20260930-1450) và lượt 2 (20261022-1539-luotA)
  predictions_cap_do/            dự đoán ngoài fold ResNet18/50/152, lượt 3 (capdo-theo_cay)
  *.csv                          bảng kết quả, sinh bằng scripts/bench.py (xem docs/Nguon_so_lieu.md)
src/coffeeguard/bench/           toàn bộ logic: hợp đồng lớp, chia fold, chỉ số, hiệu chỉnh, conformal
scripts/bench.py                 CLI: chuẩn bị dữ liệu và sinh lại các bảng
scripts/hinh_1.py                vẽ Hình 1 của bài
notebooks/                       chuẩn bị dữ liệu (01) và huấn luyện trên Google Colab (02, 02b)
tests/                           kiểm thử đơn vị cho hiệu chỉnh, conformal, từ chối, so sánh theo cặp
docs/Nguon_so_lieu.md            bảng nào trong bài lấy từ tệp nào, lệnh nào
```

Repo **không** chứa ảnh RoCoLe và trọng số mô hình. Ảnh tải từ nguồn gốc (mục "Huấn luyện lại").

## Sinh lại các bảng của bài (CPU, không cần ảnh)

Python ≥ 3.11.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt      # Linux/macOS: .venv/bin/pip
.venv/Scripts/python -m pytest tests -q
.venv/Scripts/python scripts/bench.py validate          # kiểm 9 file dự đoán theo hợp đồng
.venv/Scripts/python scripts/bench.py hieu-chinh        # Bảng 2–11 (9 mô hình lượt 1–2)
.venv/Scripts/python scripts/bench.py hieu-chinh --cap-do   # Bảng 12 (ResNet18/50/152, lượt 3)
.venv/Scripts/python scripts/bench.py khong-chac        # độ lệch theo dải, hiệu cỡ tập theo cặp
.venv/Scripts/python scripts/bench.py kiem-dinh-bai     # Holm cho các cặp cùng lượt, ngưỡng +∞ theo dải
.venv/Scripts/python scripts/hinh_1.py                  # Hình 1 -> reports/hinh_1.png
```

Toàn bộ mất khoảng 10 phút trên CPU laptop (phần lớn là bootstrap 2.000 lần). Bảng nào trong bài (Bảng 1–7,
Hình 1, phụ lục A1–A9) lấy từ tệp và cột nào: xem `docs/Nguon_so_lieu.md`.

Các lệnh ghi đè `reports/*.csv`. Bootstrap và các lần chia nửa cây dùng hạt giống cố định, nên chạy lại cho
cùng con số; `git diff reports/` sau khi chạy phải rỗng.

## Thiết lập thí nghiệm, tóm tắt

- **Dữ liệu:** RoCoLe [1], 1.560 ảnh = 390 cây × 2 lá × 2 mặt; loại `C11P11HE.jpg` (sai quy ước tên, nhãn
  gỉ sắt cấp 4) → 1.559 ảnh. Nhãn gán cho từng mặt lá; 173/780 lá có hai mặt khác nhãn. Sáu nhãn gốc gộp thành năm dải báo cáo (gỉ sắt cấp 3 + 4 → "nặng", vì cấp 4 chỉ có 29 ảnh). Ánh xạ nằm
  ở `src/coffeeguard/bench/contract.py`.
- **Chia dữ liệu:** 5 fold nhóm theo **cây** (`data/splits/rocole_folds.json`), nên hai lá và hai mặt lá của
  cùng một cây luôn cùng fold. Không có tập kiểm định riêng, không dừng sớm.
- **Tiền xử lý:** cắt ô vuông quanh khung bao lá (từ nhãn COCO của RoCoLe), cạnh = 1,4 × cạnh ngắn của khung,
  thu nhỏ Lanczos về 224 × 224 (`bench/preprocess.py`).
- **Ba lượt huấn luyện:** lượt 1 (`20260930-1450`, 5 mô hình, `notebooks/02_benchmark_colab__luot1_c0f3472.ipynb`),
  lượt 2 (`20261022-1539-luotA`, 4 mô hình, `notebooks/02_benchmark_colab.ipynb`), lượt 3 (`capdo-theo_cay`,
  ResNet18/50/152, `notebooks/02b_cap_do_*.ipynb`). ResNet18 xuất hiện ở lượt 1 và lượt 3 là hai lần huấn
  luyện độc lập.
- **Khoảng tin cậy:** bootstrap theo cụm cây, 2.000 lần lặp. Hiệu chỉnh trong fold bằng 20 lần chia nửa theo
  cây (`conformal.chia_nua_cay`), dùng chung cho mọi phương pháp; sơ đồ chéo fold chạy làm kiểm chứng.

## Huấn luyện lại từ ảnh gốc

1. Tải RoCoLe [1] và đặt theo đường dẫn trong `src/coffeeguard/bench/paths.py`:
   `data/raw/rocole/Photos/*.jpg`, `data/raw/rocole/Annotations/RoCoLE-coco.json`,
   `data/external/RoCoLe_annotations/rocole_labels.csv`.
2. `python scripts/bench.py doctor` → `index` → `images` → `package`. **Không** chạy lại `splits`: tệp chia đã
   đóng băng và `doctor` kiểm sha256 của chỉ mục khớp với tệp chia.
3. Tải `data/coffeeguard_bench_*.zip` lên Google Drive, chạy notebook 02 / 02b trên Colab (GPU T4), tải các
   `pred__*.csv` về `reports/predictions/` (hoặc `reports/predictions_cap_do/`), rồi chạy các lệnh ở mục trên.

Môi trường huấn luyện: Google Colab, GPU NVIDIA T4, PyTorch 2.11 do Colab cấp sẵn (không ghim `torch`),
`timm==1.0.30` cho lượt 2–3 và timm 1.0.29 cho lượt 1. Công thức: AdamW, tốc độ học cực đại 3 × 10⁻⁴,
lịch một chu kỳ, 15 epoch, lô 32, entropy chéo có trọng số lớp 1/√nₖ và làm trơn nhãn 0,05; hạt giống fold
f là 42 + f (chi tiết: mục 3.2 của bài và ô cấu hình của notebook). Kết quả huấn luyện lại sẽ không trùng từng số với bài do tính
không tất định của GPU; các bảng trong bài được sinh từ đúng các tệp dự đoán trong `reports/`.

## Hạn chế cần biết khi dùng lại

- Dải "gỉ sắt nặng" chỉ có 91 ảnh; mỗi nửa hiệu chỉnh chỉ còn khoảng 9 ảnh của dải này, nên độ phủ dải nặng
  dao động mạnh giữa các lần chia (Bảng 10 của bài).
- Chỉ một bộ dữ liệu, một nơi thu thập; chưa kiểm chứng trên dữ liệu ngoài.
- Một lần huấn luyện cho mỗi mô hình mỗi lượt (không lặp nhiều hạt giống).
- Một số chú thích trong mã trỏ tới tài liệu nội bộ của đồ án gốc (`Docs/...`) không có trong repo này.

## Tài liệu tham khảo

[1] J. Parraga-Alava, K. Cusme, A. Loor, E. Santander, "RoCoLe: A robusta coffee leaf images dataset for
evaluation of machine learning based methods in plant diseases recognition," *Data in Brief*, vol. 25,
p. 104414, 2019, doi: 10.1016/j.dib.2019.104414.

## Giấy phép

Mã nguồn: MIT (xem `LICENSE`). Tệp chia, chỉ mục và tệp dự đoán trong `data/` và `reports/` là sản phẩm
phái sinh từ RoCoLe; khi dùng lại xin trích dẫn cả bài báo này và [1].
