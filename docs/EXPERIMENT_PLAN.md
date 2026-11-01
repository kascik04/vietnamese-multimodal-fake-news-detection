# Kế Hoạch Thí Nghiệm & Phân Bổ Tài Nguyên GPU

Tài nguyên: **Kaggle ~45 giờ GPU/tuần** (2×T4, chạy nền) · **Colab Pro** (A100/V100, compute units có hạn) · **Google Drive 5TB** (kho lưu trữ trung tâm).

> Thời gian bên dưới là **ước lượng**. Sau lần chạy thử đầu tiên, đọc `avg_epoch_seconds` trong `run_info.json` rồi tính lại kế hoạch.

---

## 1. Phân công nền tảng

| Nền tảng | Dùng cho | Lý do |
|---|---|---|
| **Kaggle 2×T4** | Toàn bộ bảng kết quả chính + ablation + nhiều seed | Miễn phí, nhiều giờ, chạy nền 12h ("Save & Run All") |
| **Colab A100** | Tinh chỉnh siêu tham số (chỉ trên **val**), chạy gấp, debug tương tác | Nhanh ~3-4× nhưng tốn compute units |
| **Drive 5TB** | Dữ liệu (`data.zip`), mọi `results.json` / `*_predictions.csv` / checkpoint cuối cùng | Gom kết quả 2 nền tảng một chỗ để viết báo cáo |
| **Máy cá nhân** | Chuẩn bị dữ liệu, kho tin, lớp bằng chứng, phân tích kết quả, `significance_test.py` | Không cần GPU |

**Quy tắc vàng:** mọi mô hình trong **cùng một bảng so sánh** chạy trên **cùng loại GPU, cùng batch size, cùng split**. Lưu kết quả theo `results/kaggle_t4/...` và `results/colab_a100/...`.

---

## 2. Ước lượng thời gian một lần chạy (ReINTEL, ~10 epoch, early stopping)

| Cấu hình | Kaggle 2×T4 | Colab A100 |
|---|---|---|
| `social_only` | vài phút | vài phút |
| `text_only` (± `--use_style`) | ~20-40 phút | ~7-12 phút |
| `image_only` | ~20-40 phút | ~7-12 phút |
| `late_concat`, `hybrid_proposed` và các ablation | ~30-60 phút | ~10-20 phút |

---

## 3. Lịch đề xuất

### Tuần 0 – Kiểm tra (≈2 giờ Kaggle)
- [ ] `test_step1` → `test_step5` trên Kaggle (CPU/GPU).
- [ ] Chuyển ReINTEL (`convert_reintel.py`) → chia dữ liệu (`validate_and_preprocess_dataset.py`), lưu `data.zip` lên Drive.
- [ ] Chạy thử `text_only --epochs 1` và `hybrid_proposed --epochs 1` → đọc thời gian thật, điều chỉnh kế hoạch.

### Tuần 1 – Tinh chỉnh (Colab A100, ~2-4 giờ A100) song song Baseline (Kaggle, ~20 giờ)
- **Colab, 1 seed, chỉ nhìn val F1** (KHÔNG nhìn test): LR backbone {1e-5, 2e-5, 3e-5}, λ {0, 0.05, 0.1, 0.3}. Chốt siêu tham số **trước** khi chạy bảng chính.
- **Kaggle, 3 seed** (`run_seeds.py`): `text_only`, `image_only`, `social_only`, `late_concat --consistency_lambda 0` → 12 lần chạy.

### Tuần 2 – Mô hình đề xuất + Ablation (Kaggle, ~35-45 giờ)
3 seed mỗi cấu hình với siêu tham số đã chốt:
`hybrid_proposed`, `--no_cross_attention`, `--no_gate`, `--no_contradiction`, `--no_social`, `--consistency_lambda 0`, `--use_style` → 21 lần chạy (chia 2-3 notebook, mỗi lần commit < 11 giờ).

### Tuần 3 – Tổng quát hoá & phân tích (Kaggle ~15-20 giờ)
- `--split_by source` cho 2-3 mô hình tốt nhất (3 seed): khả năng với **nguồn/trang chưa thấy**.
- Đánh giá trên **tập tin địa phương 2026** tự thu thập: chỉ cần `vmtf_net.evaluate` (vài phút, không train lại).
- (Tuỳ chọn) ViSoBERT: `--text_model_name uitnlp/visobert`.
- Máy cá nhân: `significance_test.py`, bảng tổng hợp, biểu đồ từ `history.csv`, `check_post.py --eval_csv` cho lớp bằng chứng.

**Tổng:** ~75-90 giờ Kaggle trong 3 tuần (dưới quota 45 giờ/tuần) + vài giờ A100 → vẫn dư ~30% để chạy lại khi có lỗi.

---

## 4. Lưu ý dung lượng & phiên

- Kaggle `/kaggle/working` giới hạn **20GB**: `best_vmtf_model.pt` ~0.9GB/lần chạy → 15 lần chạy đã ~14GB. Sau mỗi version, **tải `results.json` + `*_predictions.csv` + `history.csv` về Drive**; checkpoint chỉ cần giữ cho mô hình cuối cùng (kiểm định thống kê dùng `test_predictions.csv`, không cần checkpoint).
- `last_checkpoint.pt` (~3GB) tự xoá khi chạy xong; nếu Kaggle/Colab bị ngắt, chạy lại đúng lệnh `run_seeds.py` → tự resume.
- Colab: dữ liệu để **một file `data.zip`** trên Drive (nội dung zip: `train.csv`, `val.csv`, `test.csv`, `images/` ở gốc), giải nén vào `/content` mỗi phiên; `--checkpoint_dir` trỏ lên Drive.
- Trên A100 mặc định dùng **bf16 + TF32** (`--amp_dtype auto`); trên T4/V100 dùng fp16. `run_info.json` ghi lại kiểu precision của từng lần chạy.
