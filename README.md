# CHECKLIST HA – thư mục ảnh chụp -> JSON

## Cài đặt (máy mới)
Cần **Python 3.9 – 3.13** (kiểm tra bằng `python --version`; Windows tải tại python.org, nhớ tick *Add python.exe to PATH*).

| Hệ điều hành | Cài một lần | Chạy |
|---|---|---|
| Windows | bấm đúp `setup.bat` | bỏ ảnh vào `image\`, bấm đúp `run.bat` |
| Linux / macOS | `./setup.sh` | `./run.sh` |

Hai script trên tạo môi trường ảo `.venv/`, cài thư viện trong `requirements.txt` và tạo `.env` từ `.env.example`.
Mở `.env` điền thông tin model:
```dotenv
OPENAI_API_KEY=...
OPENAI_BASE_URL=http://.../v1
MODEL_NAME=...
```
Hoặc làm tay: `python -m venv .venv` → kích hoạt → `pip install -r requirements.txt` → `python ocr.py`.

**Docker** (không cần cài Python):
```bash
docker build -t checklist-ocr .
docker run --rm --env-file .env \
  -v "$PWD/image:/app/image" -v "$PWD/result:/app/result" -v "$PWD/result_json:/app/result_json" \
  checklist-ocr
```
Nếu model chạy ngay trên máy chủ, trong container đừng dùng `localhost` mà dùng `host.docker.internal` (hoặc `--network host` trên Linux).

**Chuyển sang máy khác**: chép cả thư mục này, trừ `.venv/`, `result/`, `result_json/` và (nếu không muốn lộ khóa) `.env`. Bắt buộc phải mang theo `template_print.png` (nằm cạnh `ocr.py`).

## API (FastAPI, cổng 33253)
Chạy: `./serve.sh` / `serve.bat`, hoặc `python api.py`, hoặc `docker compose up -d --build`. Mặc định nghe `0.0.0.0:33253` (đổi bằng `PORT` trong `.env`). Tài liệu tương tác: `http://<server>:33253/docs`.

| Method | Đường dẫn | Việc |
|---|---|---|
| GET | `/health` | còn sống không; `ai_configured`, `notes_enabled` |
| POST | `/ocr` | 1 ảnh (multipart, trường `file`) -> JSON cùng cấu trúc như file trong `result_json/` |
| POST | `/ocr/batch` | tối đa `MAX_BATCH_FILES` ảnh (trường `files`) -> `{"count", "results": [...]}`; ảnh lỗi chỉ có `error`, không làm hỏng cả yêu cầu |

Tham số query: `skip_ai=true` (chỉ dò checkbox/vùng ghi chú), `verbose=true` (điểm từng ô), `include_images=true` (thêm `images.aligned_jpg_base64` và `images.debug_png_base64`).

```bash
curl -H "X-API-Key: $API_TOKEN" -F "file=@image/phieu1.jpg" "http://SERVER:33253/ocr"
curl -H "X-API-Key: $API_TOKEN" -F "files=@a.jpg" -F "files=@b.jpg" "http://SERVER:33253/ocr/batch?include_images=true"
```
Mã lỗi: `400` không phải ảnh, `401` sai/thiếu `X-API-Key`, `413` ảnh quá lớn (`MAX_UPLOAD_MB`), `503` chưa cấu hình model mà không dùng `skip_ai`. Ảnh căn chỉnh thất bại vẫn trả `200` với `alignment.ok = false` và `needs_review` có `ALIGNMENT_FAILED` (không ô nào bị báo tick): client phải kiểm tra trường này.

Triển khai lên server:
1. Chép thư mục lên server (không chép `.venv/`, `result*/`), `./setup.sh`, điền `.env`. **Đặt `API_TOKEN`** (ví dụ `openssl rand -hex 24`); để trống thì ai truy cập được cổng 33253 cũng gọi được và tốn tiền model của bạn.
2. Mở cổng: `sudo ufw allow 33253/tcp` (và security group/firewall của nhà cung cấp server nếu có).
3. Chạy nền, tự khởi động lại: `checklist-ocr.service` (systemd, sửa `User`/đường dẫn rồi làm theo hướng dẫn đầu file) hoặc `docker compose up -d`.
4. Nên đặt sau HTTPS (nginx/Caddy proxy tới `127.0.0.1:33253`, nginx cần `client_max_body_size 25m;`) vì ảnh phiếu bệnh nhân đi qua mạng.
5. Tải: mỗi ảnh tốn vài giây CPU; `MAX_CONCURRENCY` (mặc định 2) giới hạn số ảnh xử lý cùng lúc mỗi tiến trình, `WORKERS` tăng số tiến trình. Đặt `MAX_CONCURRENCY x WORKERS` không vượt quá số nhân CPU.
6. Kiểm tra: `python test_service.py` (lõi, không cần mạng) và `python test_api.py` (HTTP, cần `pip install httpx`).

## Chạy (theo lô thư mục ảnh, không qua API)
Đặt toàn bộ ảnh vào thư mục `image/` rồi chạy:
```bash
python ocr.py
```
Kết quả:
```
image/            <- ảnh đầu vào (jpg, jpeg, png, bmp, webp, tif, tiff, jfif; không phân biệt hoa/thường)
result/
  aligned/        <- <tên>.jpg  (ảnh đã căn chỉnh về khung chuẩn)
  debug/          <- <tên>.png  (ảnh debug: xanh = không tick, đỏ = tick, cam = không chắc)
result_json/      <- <tên>.json (mỗi ảnh một file) + _summary.json (tổng hợp cả lô)
```

Tùy chọn:
```bash
python ocr.py <thư_mục_hoặc_1_file_ảnh> --result-dir result --json-dir result_json
python ocr.py --recursive          # quét cả thư mục con
python ocr.py --skip-existing      # chạy tiếp: bỏ qua ảnh đã xong, chạy lại ảnh từng bị lỗi
python ocr.py --skip-ai            # chỉ dò checkbox, không gọi model OCR chữ viết tay
python ocr.py --blank-template blank_form.jpg   # chỉ cần khi muốn tạo lại template_print.png từ ảnh form trống
python ocr.py --verbose            # ghi điểm từng checkbox vào JSON (_debug)
```

Quy ước đặt tên: file kết quả lấy theo tên ảnh. Ảnh trong thư mục con có dạng `thumuc__ten`; hai ảnh cùng tên khác đuôi (`a.jpg`, `a.png`) thành `a_jpg`, `a_png` nên không bao giờ ghi đè nhau.

Một ảnh lỗi (file hỏng, không phải ảnh, API lỗi...) không làm dừng cả lô: JSON của ảnh đó chứa `error`, và `_summary.json` ghi `status: error`.

`_summary.json`: mỗi ảnh một dòng gồm `status` (`ok` / `alignment_failed` / `error`), số ô khớp được (`squares_located`), danh sách ô đã tick và số mục cần xem lại.

## Cách hoạt động
1. **Căn chỉnh** ảnh về khung 1240x1754, **tự xoay đúng hướng** nếu ảnh bị nằm ngang hoặc ngược (0°/90°/180°/270°; thử hướng có khả năng nhất trước, dừng ở hướng đầu tiên khớp rõ ràng, nếu không chọn hướng khớp nhiều ô nhất; ảnh thẳng không tốn thêm thời gian). Số độ đã xoay ghi ở `alignment.rotation` trong JSON, ảnh trong `result/aligned/` luôn là ảnh đã đúng hướng. Ảnh điện thoại bị xoay do thẻ EXIF thì được xử lý sẵn khi đọc ảnh. Với mỗi hướng, thử lần lượt nhiều cách rồi chọn cách khớp được nhiều ô vuông in sẵn nhất (tối đa 43):
   - `feature_homography`: SIFT/ORB với form trống (nếu có),
   - `box_lattice`: dò các ô vuông in sẵn trong ảnh rồi khớp với lưới 43 ô đã biết. Không cần form trống, chịu được ảnh lệch/xoay/nghiêng/tờ giấy không chiếm hết khung,
   - `document_corners`, `plain_resize` làm phương án dự phòng.
2. **Cổng chặn**: nếu khớp được ít hơn `ALIGN_MIN_BOXES` (28/43) ô thì `alignment.ok = false`, KHÔNG ô nào được báo `checked`, toàn bộ ô vào `uncertain` và `needs_review` có `ALIGNMENT_FAILED`.
3. Mục 1-5: mỗi ô được định vị lại bằng template hình vuông và đo mực trong ô. 3 trạng thái `checked` / `unchecked` / `uncertain`; ô không chắc không bao giờ thành true.
4. Header + mục 6/7/8: model đa phương thức OCR từng trường một (mỗi trường có gợi ý riêng), trả `value` + `confidence`. Vùng mục 6/7/8 dừng ở x=880 để loại chữ ký; chữ in sẵn và kết quả <= 2 ký tự ở mục 6/7/8 bị lọc.
5. **Ghi chú cạnh ô (liều lượng, note)** – vùng OCR thích ứng, không dùng khung cố định:
   - Form in sẵn cố định, nên `template_print.png` (tạo từ ảnh form trống) cho biết chính xác chữ in và dòng chấm nằm ở đâu. Mỗi ảnh, template được **uốn nhẹ cho khớp tờ giấy đó** (giấy cong, lệch ống kính) rồi trừ đi: phần nét còn lại trong vùng viết là chữ viết tay. Cách này **không phụ thuộc màu bút hay ánh sáng** (bút đen cũng được; đèn vàng/đèn lạnh không làm nhãn in bị nhận nhầm là ghi chú).
   - Bỏ nét tick, gom các nét còn lại thành từ/dòng; mỗi cụm chữ gán cho **ô gần nhất bên trái, cùng hàng**; vùng crop là khung bao của cụm đó.
   - Ảnh gửi cho model chỉ giữ nét viết tay (chữ in và dòng chấm được tô trắng), kèm gợi ý theo tên ô.
   - Dòng kẻ tự do dưới mục 3 (THD) và 4 (Căng chỉ) không có ô bên cạnh nên được ghi vào `line_notes`.
   - Chỉ chạy khi căn chỉnh `ok` **và** có `template_print.png`. Không có template thì bỏ qua dò ghi chú (có cảnh báo, `notes_enabled: false` trong JSON), không đoán bừa. Ảnh debug vẽ vùng ghi chú màu tím, có đường nối tới ô.
6. JSON chỉ có title mục 1-5, danh sách ô đã tick và ghi chú của từng ô.

## Ghi chú trong JSON
```json
"1_Nhan": {
  "title": "Nhăn",
  "checked": ["Nhăn mắt"],
  "uncertain": [],
  "notes": {
    "Nhăn mắt":      {"value": "10U", "confidence": 0.95, "region": [771, 442, 850, 477], "checked": true},
    "Khác (ghi rõ)": {"value": "Viền Hàm (20u)", "confidence": 0.9, "region": [805, 516, 1137, 598], "checked": false}
  }
},
"3_THD": { "...": "...", "notes": {}, "line_notes": [] }
```
- `checked` trong `notes` cho biết ô đó có tick không. **Có chữ cạnh ô không làm ô thành tick**: ô "Khác (ghi rõ)" có ghi chú nhưng vẫn `checked: false`.
- `needs_review` có thêm: `<mục>:<ô>:note` (OCR lỗi / không đọc được / độ tin cậy thấp), `<mục>:<ô>:note_without_tick` (có chữ cạnh ô chưa tick, trừ ô "ghi rõ"), `<mục>:line_note<N>`.
- Dùng `--skip-ai` vẫn thấy vùng ghi chú (`region`) nhưng `value` rỗng.

## Lưu ý
- Luôn mở vài ảnh `.debug.png` để kiểm tra, nhất là các ảnh `alignment_failed` hoặc có nhiều mục trong `needs_review`. Các ngưỡng nằm ở đầu phần "Checkbox detection" trong `ocr.py`.
- Nếu `alignment.ok = false`: chụp lại ảnh rõ nét, thấy đủ cả tờ giấy (nhất là các ô vuông), tránh bóng đổ/lóa mạnh.
- OCR chữ viết tay vẫn là model nên có thể sai; những trường confidence thấp nằm trong `needs_review`.
- Chữ viết tay đè lên chữ in/dòng chấm có thể bị cắt mất một đoạn nhỏ; chữ quá mờ, nhỏ hoặc nằm sát mép giấy có thể không được phát hiện (tick và các trường khác không bị ảnh hưởng). Nếu đổi mẫu form, phải tạo lại `template_print.png` từ ảnh form trống mới (xóa file cũ rồi chạy `--blank-template`) và cập nhật tọa độ ô trong `ocr.py`.
- Kiểm tra nhanh sau khi cài: `python test_synthetic.py` (không cần ảnh thật, không gọi model) phải kết thúc không có dòng `FAIL`.
- Không commit `.env` lên Git.
