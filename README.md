# Hand Gesture Recognition Project

## Mục tiêu
Nhận diện cử chỉ tay (hand gesture) từ ảnh hoặc video, phục vụ các ứng dụng tương tác người-máy, giáo dục, và nghiên cứu AI.

## Thư mục chính
- `Data/` : Dữ liệu gốc, dữ liệu đã xử lý, ảnh, CSV landmark, script xử lý dữ liệu.
- `extract_features.py` : Trích xuất đặc trưng từ landmark bàn tay.
- `ml_demo.py` / `ml_demo.ipynb` : Huấn luyện, đánh giá và lưu mô hình ML (Random Forest, SVM, KNN).
- `Tkinker/build/` : Giao diện người dùng (GUI) demo nhận diện gesture bằng Tkinter.
- `SVM_explained.ipynb` : Notebook giải thích lý thuyết và thực nghiệm SVM.

## Quy trình tổng thể
1. **Thu thập dữ liệu:**
   - Ảnh cử chỉ tay được lưu trong `Data/hagrid/` (mỗi thư mục con là một gesture).
2. **Tiền xử lý & Làm sạch:**
   - Sử dụng `data_raw.py` hoặc `data_raw2.py` để detect landmark, loại bỏ ảnh lỗi/không ổn định, xuất ra CSV.
3. **Trích xuất đặc trưng:**
   - Chạy `extract_features.py` để tạo file `Data.csv` chứa các đặc trưng số học từ landmark.
4. **Huấn luyện & đánh giá mô hình:**
   - Sử dụng `ml_demo.ipynb` để huấn luyện, so sánh, đánh giá các mô hình ML.
   - Lưu mô hình tốt nhất và các thông tin cần thiết.
5. **Demo nhận diện real-time:**
   - Chạy GUI trong `Tkinker/build/gui.py` để nhận diện gesture trực tiếp từ webcam.
6. **Giải thích lý thuyết:**
   - Tham khảo `SVM_explained.ipynb` để hiểu sâu về SVM và các bước thực nghiệm minh họa.

## Yêu cầu môi trường
- Python >= 3.8
- Các thư viện: `opencv-python`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`, `cvzone`, `mediapipe`, `Pillow`

Cài đặt nhanh:
```bash
pip install Requirements.txt
```

## Hướng dẫn sử dụng nhanh
1. **Xử lý dữ liệu landmark:**
   - Chạy `python Data/data_raw2.py` để tạo file landmark sạch.
2. **Trích xuất đặc trưng:**
   - Chạy `python extract_features.py` để tạo `Data.csv`.
3. **Huấn luyện mô hình:**
   - Mở và chạy từng cell trong `ml_demo.ipynb` để huấn luyện, đánh giá và lưu mô hình.
4. **Demo GUI:**
   - Chạy `python Tkinker/build/gui.py` để nhận diện gesture trực tiếp.

## Đóng góp & phát triển
- Bạn có thể mở rộng thêm các gesture mới, cải tiến đặc trưng, thử nghiệm các mô hình khác hoặc cải tiến giao diện GUI.
- Mọi ý kiến đóng góp đều được hoan nghênh!

---
**Tác giả:**
- Dự án phát triển phục vụ học tập, nghiên cứu và demo AI/ML thực tế với dữ liệu cử chỉ tay.
