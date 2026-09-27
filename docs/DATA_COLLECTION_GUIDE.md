# Sổ Tay Hướng Dẫn Thu Thập & Gán Nhãn Dữ Liệu Đa Phương Thức (VMTF-Net)
## Vietnamese Multimodal Fake News Dataset Collection & Annotation Protocol

Tài liệu này định nghĩa quy chuẩn quốc tế và phương pháp luận thu thập dữ liệu dành cho đề tài **Phát hiện tin giả đa phương thức tiếng Việt (VMTF-Net)**. 
Bộ schema chuẩn hóa gồm **55 cột**, hỗ trợ trọn vẹn cả 3 nhánh mô hình: **Văn bản (PhoBERT)**, **Hình ảnh (ViT)**, và **Tương tác xã hội (Social MLP)**.

---

## 1. Cấu Trúc Thư Mục Dữ Liệu Chuẩn

```
data/
├── dataset_template.csv         # File mẫu chứa header và các dòng dữ liệu mẫu
├── raw_collected_data.csv       # File tổng hợp sau khi cào / gán nhãn thực tế
├── train.csv                    # Tập huấn luyện (70%) - tạo tự động qua script
├── val.csv                      # Tập kiểm định (15%) - tạo tự động qua script
├── test.csv                     # Tập kiểm thử độc lập (15%) - tạo tự động qua script
└── images/                      # Thư mục lưu trữ toàn bộ ảnh
    ├── VMTF_000001_1.jpg
    ├── VMTF_000002_1.jpg
    └── ...
```

---

## 2. Từ Điển Dữ Liệu (Data Dictionary - 55 Cột Chuẩn Hóa)

### Nhóm 1: Định Danh & Nguồn Bài Viết (Identification & Source)
1. **`sample_id`** *(Bắt buộc)*: Mã định danh duy nhất của mẫu tin. Quy ước: `VMTF_000001`, `VMTF_000002`...
2. **`platform`**: Nền tảng mạng xã hội thu thập (`Facebook`, `TikTok`, `Threads`, `Zalo`, `Web`).
3. **`source`**: Tên Fanpage, Kênh, hoặc Tác giả đăng bài (VD: `VTV24`, `Hóng Biến 24/7`, `Người Đưa Tin`).
4. **`source_url`**: Đường dẫn trực tiếp đến bài viết gốc (hoặc link lưu trữ WebArchive/Archive.today nếu bài đã bị xóa).
5. **`timestamp`**: Thời gian đăng bài trên mạng xã hội (`YYYY-MM-DD HH:MM:SS`).
6. **`crawl_timestamp`**: Thời gian bạn thực hiện cào/thu thập dữ liệu.

### Nhóm 2: Nhánh Văn Bản (Text Modality - PhoBERT / ViSoBERT)
7. **`text`** *(Bắt buộc)*: Nội dung chữ của bài viết (bao gồm Caption, Status, hoặc Tiêu đề + Đoạn trích). Giữ nguyên chính tả gốc, biểu tượng cảm xúc (emoji).
8. **`language`**: Ngôn ngữ (`vi` cho tiếng Việt).

### Nhóm 3: Nhánh Hình Ảnh (Visual Modality - ViT Backbone)
Hỗ trợ tối đa 3 ảnh. Mô hình mặc định lấy `image_1_local_path`, nếu ảnh 1 lỗi hoặc thiếu sẽ tự động fallback sang ảnh 2 và ảnh 3:
9. **`image_1_url`**: URL tải ảnh thứ nhất.
10. **`image_1_local_path`** *(Bắt buộc nếu bài có ảnh)*: Tên file ảnh đã tải về thư mục `data/images/` (VD: `VMTF_000001_1.jpg`).
11. **`image_1_width`**, **`image_1_height`**: Kích thước ảnh (pixel).
12. **`image_1_sha256`**: Mã băm SHA-256 của file ảnh (script sẽ tự động tính nếu để trống).
13. **`image_1_phash`**: Mã băm nhận thức Perceptual Hash (script tự động tính để phát hiện ảnh tương đồng / ảnh chỉnh sửa).
14-19. **`image_2_*`**: Thông tin ảnh 2 (nếu bài viết có nhiều hơn 1 ảnh).
20-25. **`image_3_*`**: Thông tin ảnh 3 (nếu có).

### Nhóm 4: Nhánh Tương Tác Mạng Xã Hội (Social Engagement Modality)
26. **`like_count`** *(Khuyên dùng)*: Số lượt Thích (hoặc tổng lượt Like/Thả tim).
27. **`comment_count`** *(Khuyên dùng)*: Số lượt bình luận của bài viết.
28. **`share_count`** *(Khuyên dùng)*: Số lượt chia sẻ bài viết (chỉ số cực kỳ quan trọng với tin giả).
29. **`angry_count`**: Số lượt thả cảm xúc Phẫn nộ (Angry) - phản ánh sự phẫn nộ/hoang mang của cộng đồng.
30. **`haha_count`**: Số lượt thả cảm xúc Haha - phản ánh tính lừa bịp hoặc chế giễu của cộng đồng.
31. **`like_share_ratio`**: Tỷ lệ `like / (share + 1)`. *(Để trống script sẽ tự tính)*.
32. **`comment_share_ratio`**: Tỷ lệ `comment / (share + 1)`. *(Để trống script sẽ tự tính)*.
33. **`engagement_ratio`**: Tổng tương tác trên mỗi lượt like. *(Để trống script sẽ tự tính)*.

### Nhóm 5: Bình Luận Phản Biện (Rebuttal Comments)
Ghi nhận các bình luận hàng đầu của cộng đồng vạch trần hoặc cung cấp bằng chứng tin giả:
34. **`rebuttal_comment_1`**: Nội dung bình luận phản biện 1.
35. **`rebuttal_comment_1_likes`**: Số lượt like của bình luận phản biện 1.
36. **`rebuttal_comment_1_timestamp`**: Thời gian đăng bình luận 1.
37-42. **`rebuttal_comment_2_*`**, **`rebuttal_comment_3_*`**: Bình luận phản biện 2 và 3.

### Nhóm 6: Gán Nhãn & Bằng Chứng Xác Thực (Labels & Fact-Checking Evidence)
43. **`label`** *(Bắt buộc)*: Nhãn nhị phân:
    - **`0`**: Tin thật (Real / Legit News).
    - **`1`**: Tin giả / Tin sai sự thật (Fake / Misinformation).
44. **`raw_label`**: Nhãn chữ (`real` hoặc `fake`).
45. **`label_source`**: Nguồn xác thực (VD: `BoYTe`, `BaoChinhPhu`, `FactCheckVN`, `ReINTEL`, `VTV24`).
46. **`label_confidence`**: Mức độ tin cậy của nhãn từ `0.0` đến `1.0` (mặc định: `1.0`).
47. **`evidence_url`**: Đường dẫn bài viết đính chính/bác bỏ chính thức từ cơ quan báo chí, y tế, chính phủ.
48. **`evidence_note`**: Tóm tắt lý do kết luận bài viết là Thật hoặc Giả.

### Nhóm 7: Quản Trị Dữ Liệu & Chống Rò Rỉ (Quality & Deduplication)
49. **`claim_id`**: Mã nhóm sự việc/luận điểm (VD: `CLAIM_HEALTH_001`, `CLAIM_TYPHOON_YAGI`).
50. **`duplicate_group`**: Mã cụm tin copy/repost (đảm bảo chia Train/Val/Test không bị rò rỉ dữ liệu).
51. **`data_quality`**: Chất lượng bản ghi (`high`, `medium`, `low`).
52. **`annotation_status`**: Trạng thái kiểm duyệt (`reviewed`, `needs_review`, `unreviewed`).
53. **`annotator`**: Tên hoặc ID người thu thập/gán nhãn.
54. **`notes`**: Ghi chú đặc thù (VD: "Tin dạng Out-of-Context lấy ảnh năm 2020 ghép sự kiện 2026").

---

## 3. Quy Tắc Gán Nhãn Chuẩn (Annotation Guidelines)

### Khi nào gán nhãn `label = 1` (Tin Giả / Fake)?
1. **Tin giả mạo hoàn toàn (Fabricated Content)**: Tự bịa đặt sự việc không có thật, làm giả công văn nhà nước, tạo phát ngôn giả của người nổi tiếng.
2. **Tin sai ngữ cảnh (Out-of-Context - Rất quan trọng với VMTF-Net)**: 
   - Ảnh là ảnh thật nhưng gán ghép vào một vụ việc/địa điểm/thời gian hoàn toàn khác.
   - Nhánh Cross-Attention Contradiction của VMTF-Net được thiết kế đặc thù để phát hiện loại tin giả này.
3. **Tin y tế/lừa đảo nguy hại (Scam / Misleading Health Advice)**:
   - Các bài viết "uống lá cây chữa dứt điểm ung thư", "thuốc thần y trị bách bệnh", bài viết mạo danh bác sĩ bệnh viện lớn.
4. **Tin giật gân sai lệch (Manipulated Content)**: Ảnh đã qua chỉnh sửa Photoshop hoặc video/ảnh AI Deepfake gây hoang mang dư luận.

### Khi nào gán nhãn `label = 0` (Tin Thật / Real)?
- Bài viết từ trang chính thống của Bộ ban ngành, báo chí cách mạng (VTV, Thông tấn xã Việt Nam, Báo Chính Phủ, Tuổi Trẻ, Dân Trí, v.v.).
- Tin tức đời sống thường nhật được cộng đồng và báo chí xác nhận đúng sự thật khách quan.

### Những trường hợp KHÔNG thu thập hoặc để `needs_review`:
- **Châm biếm / Hài hước (Satire)**: Những trang chuyên giải trí, meme hài hước rõ ràng.
- **Quan điểm chủ quan (Personal Opinion)**: Ý kiến đánh giá cá nhân (khen/chê món ăn, trải nghiệm dịch vụ) không chứa tuyên bố sự thật (claim of fact).

---

## 4. Quy Trình Sử Dụng Tool Kiểm Tra & Tự Động Tạo Tập Huấn Luyện

Sau khi bạn đã điền các dòng dữ liệu vào file (ví dụ `data/my_collected_data.csv`), bạn chỉ cần chạy 1 lệnh tự động:

```bash
python scripts/validate_and_preprocess_dataset.py \
    --input_csv data/my_collected_data.csv \
    --image_dir data/images \
    --output_dir data
```

**Script sẽ tự động làm toàn bộ các việc sau:**
1. Kiểm tra đủ 55 cột, cảnh báo các cột thiếu.
2. Quét kiểm tra xem file ảnh trong `data/images/` có tồn tại không.
3. Tự động tính mã SHA-256 và pHash cho từng ảnh.
4. Tự động tính các tỷ lệ `like_share_ratio`, `comment_share_ratio`, `engagement_ratio`.
5. Tự động loại bỏ các dòng mẫu `EXAMPLE_` hoặc dòng chưa gán nhãn.
6. Chia tập dữ liệu thành `train.csv` (70%), `val.csv` (15%), `test.csv` (15%) theo **Group-Stratified Split** (dựa trên `duplicate_group`) để đảm bảo các bài tin repost/copy không xuất hiện ở cả Train và Test split.

---

## 5. Chạy Huấn Luyện VMTF-Net Với Bộ Dữ Liệu Thu Thập

Khi đưa lên Kaggle GPU hoặc máy trạm huấn luyện, bạn có thể kích hoạt chế độ **Extended Social Features (8 đặc trưng tương tác)** để khai thác tối đa bảng dữ liệu:

```bash
python -m vmtf_net.train \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --image_dir data/images \
    --social_preset extended \
    --batch_size 16 \
    --epochs 10 \
    --lr 2e-5 \
    --head_lr 1e-4 \
    --fusion_type hybrid_proposed
```
