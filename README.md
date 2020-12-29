## viText - End-to-end OCR for Vietnamese
viText cung cấp 1 số công cụ liên quan đến **nhận dạng ký tự tiếng Việt** như dataset viReceipts, detection hay OCR. Hiện tại viText làm việc tốt nhất với văn bản được scan

![SAMPLE](https://github.com/titikid/viText/blob/dev/viText/outputs/scan2_visualized.jpg)
## Cài đặt
Để cài đặt viText các bạn chạy lệnh sau
```
git clone https://github.com/ndcuong91/viText.git
cd viText
pip install -e .
```

## Cấu trúc dự án 
### 1. viData

Bên mình đang xây dựng bộ dataset viReceipts cho dữ liệu hóa đơn bán lẻ ở Việt Nam. Các bạn có thể sử dụng một vài mẫu ở trong thư mục viData/viReceipts nhé 

### 2. viDet
viDet cung cấp một số models để giải quyết bài toán phát hiện text trong văn bản
- [ ] DBnet
- [ ] EAST
(to be continue...)

### 3. viOCR
Bước cuối cùng là OCR sẽ được trích xuất bởi các model như CRNN, vietocr h

- [x] CRNN
- [ ] VietOCR
- [ ] SRN


## Roadmaps
- [x] Update viOCR: CRNN
- [ ] Update viDet: DBnet
- [ ] Update viReceipts

## Contribute & Contact
Các bạn có thể tạo PR hoặc liên hệ mình: titikid@gmail.com
