# Nguồn của từng con số trong bài

Mọi số trong bài làm tròn nửa lên từ giá trị 4 chữ số trong các CSV dưới đây. Tên lượt:

| Lượt | `run_id` | Mô hình | Tệp dự đoán | Notebook huấn luyện |
|---|---|---|---|---|
| 1 | `20260930-1450` | EfficientNet-B0, EfficientNet-Lite0, MobileNetV2, MobileViTv2-050, ResNet18 | `reports/predictions/*__20260930-1450.csv` | `notebooks/02_benchmark_colab__luot1_c0f3472.ipynb` |
| 2 | `20261022-1539-luotA` | EfficientNetV2-B0, FastViT-T8, MobileViTv2-050, MobileViTv2-075 | `reports/predictions/*__20261022-1539-luotA.csv` | `notebooks/02_benchmark_colab.ipynb` |
| 3 | `capdo-theo_cay` | ResNet18, ResNet50, ResNet152 | `reports/predictions_cap_do/*-theo_cay.csv` | `notebooks/02b_cap_do_{nhe,trung,nang}.ipynb` |

Lượt 1 (30/09/2026) chạy timm 1.0.29 + PyTorch 2.11, và MobileViTv2-050 lượt 1 chạy ở đầu vào 256 (mọi mô
hình khác: 224). Lượt 2 (22/10/2026): timm 1.0.30 + PyTorch 2.11. Lượt 3 (30/10/2026): timm 1.0.30. Cả ba lượt
dùng cùng công thức huấn luyện (mục 3.2 của bài), chỉ khác danh sách kiến trúc và phiên bản timm.

## Bảng → tệp → lệnh

Đánh số theo bản bài đã sửa (Bảng 1–7, Hình 1, phụ lục Bảng A1–A9). Mọi tệp nằm trong `reports/`.
`hieu-chinh` chạy trên 9 mô hình lượt 1–2; `hieu-chinh --cap-do` chạy trên ResNet lượt 3 và ghi cùng tên tệp
kèm hậu tố `__cap_do`.

| Trong bài | Tệp | Lệnh / hàm |
|---|---|---|
| Bảng 1 (số ảnh theo dải 791 / 167 / 344 / 166 / 91) | `data/interim/rocole_index.csv`, cột `band` | đếm trực tiếp |
| Mục 3.1: 173/780 lá (22,2%) có hai mặt khác nhãn | `data/interim/rocole_index.csv`: số `leaf_id` có hơn một `label6` | đếm trực tiếp |
| Bảng 2 (độ phủ ở ngưỡng 0,7, macro-F1) | `phu_nguong_co_dinh.csv` | `hieu-chinh` · `tu_choi.duong_do_phu` |
| Bảng 3 (độ lệch khi báo "nặng", trước/sau hiệu chỉnh) | `hieu_chinh_cung_so_do.csv` (dòng `hieu_chinh = trong_fold`, tức sơ đồ nửa cây) | `hieu-chinh` · `hieu_chinh.danh_gia` |
| Bảng 4 (ECE / macro-F1 / recall dải nặng) | `hieu_chinh_cung_so_do.csv` | như trên |
| Bảng 5 (ngưỡng theo dải, *n*_min = 10) | `nguong_cung_so_do.csv` | `hieu-chinh` · `hieu_chinh.nguong_theo_dai` |
| Bảng 6 (độ phủ conformal biên / theo dải, α = 0,20) | `conformal_doi_chung.csv` (`diem = thr`, tức điểm 1 − p̂) | `hieu-chinh` · `conformal.tom_tat` |
| Hình 1 (MobileViTv2-050 lượt 2, số liệu Bảng 6) | `reports/hinh_1.png` vẽ từ `conformal_doi_chung.csv` | `python scripts/hinh_1.py` |
| Bảng 7 (đọc tập dự đoán theo dải: tập một dải, tập rỗng, tập liền nhau) | `conformal_doi_chung.csv` | như trên |
| Hiệu cỡ tập 0,51 [0,47; 0,55] và 15/16 cặp (mục 4.4) | `conformal_cap_holm.csv` | `kiem-dinh-bai` |
| "79 ảnh, tin cậy 0,70, đúng 54%" | `lech_theo_dai.csv` (`dinh_nghia = dai`) | `khong-chac` |
| Bảng A1 (bốn điểm × hai chế độ × hai sơ đồ) | `conformal_doi_chung.csv` | `hieu-chinh` |
| Bảng A2 (cỡ tập, tỉ lệ tập liền nhau của bốn điểm) | `conformal_doi_chung.csv` (`che_do = theo_dai`) | `hieu-chinh` |
| Bảng A3 (phân tán độ phủ qua 20 lần chia) | `conformal_doi_chung.csv`, cột `phu_lan_sd__*`, `phu_lan_min__*`, `phu_lan_max__*`, `phu_nua_min__*` | `hieu-chinh` |
| Bảng A4, A8 (α = 0,10, theo dải / biên) | `conformal_doi_chung.csv` (`alpha = 0.1`) | `hieu-chinh` |
| Bảng A5 (họ ResNet, lượt 3) | bốn tệp `*__cap_do.csv` | `hieu-chinh --cap-do` |
| Bảng A6 (số ảnh báo nặng; kiểm chứng chéo fold) | `hieu_chinh_cung_so_do.csv` (cột `n_bao_nang_sau`; dòng `hieu_chinh = cheo_fold`) | `hieu-chinh` |
| Bảng A7 (ngưỡng theo dải, *n*_min = 5) | `nguong_cung_so_do.csv` | `hieu-chinh` |
| Bảng A9 (so cỡ tập theo cặp cùng lượt, Holm) | `conformal_cap_holm.csv` | `kiem-dinh-bai` · `bm.holm` |
| Tỉ lệ ngưỡng +∞ theo dải (mục 4.5: 8,4%; "80 trên 84" của MobileViTv2-050 lượt 2) | `conformal_vo_cuc_theo_dai.csv` | `kiem-dinh-bai` |

Chi tiết phương pháp:

| Nội dung | Nguồn |
|---|---|
| Khung bao lá | trường `bbox` của tệp COCO RoCoLe, `src/coffeeguard/bench/index.py`; cắt + Lanczos: `bench/preprocess.py` |
| Lưới nhiệt độ *T* | `tu_choi.LUOI_NHIET_DO` = exp(linspace(log 0,2; log 5; 200)) |
| Sơ đồ nửa cây (20 lần chia) | `conformal.chia_nua_cay`, dùng chung cho mọi phương pháp |
| Checkpoint | epoch cuối (15), không tập kiểm định, không dừng sớm — ô cấu hình của notebook 02 / 02b |

Các tệp `nhiet_do.csv`, `nguong_theo_dai.csv`, `conformal_theo_dai.csv`, `conformal_cap.csv` (lệnh `khong-chac`)
là phép đo của bản đầu, theo sơ đồ chéo fold cũ; bài đã sửa không lấy số từ chúng, nhưng giữ lại để đối chiếu.

## Định dạng tệp dự đoán

Mỗi dòng một ảnh, dự đoán khi ảnh đó nằm ở fold kiểm tra:

```
png, fold, plant_id, leaf_id, face, y_true, y_pred,
p_healthy, p_red_spider_mite, p_rust_level_1, p_rust_level_2, p_rust_level_3, p_rust_level_4,
framework, arch, run_id
```

Thứ tự lớp và ánh xạ 6 lớp → 5 dải: `src/coffeeguard/bench/contract.py` (`CLASSES_6`, `BAND_OF`).
`python scripts/bench.py validate` kiểm mọi tệp theo hợp đồng này (đủ 1.559 ảnh, xác suất cộng bằng 1,
fold khớp tệp chia đã đóng băng).
