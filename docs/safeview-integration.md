# Tích hợp bản demo CS114 với SafeView

## Trạng thái thực tế

Mã inference, Space template và patch tích hợp đã được viết. Chưa có model ViHSD
đã đánh giá hoặc URL Space mới, nên chưa publish, chưa đổi repo SafeView và chưa
đo độ trễ mạng thực tế. Smoke test bằng dữ liệu tổng hợp chỉ kiểm tra phần mềm,
không phải bằng chứng chất lượng phân loại.

Đã kiểm tra repo cùng cấp `/Users/thong/Data/Projects/safe-view` tại commit
`7e09b01569365c6d173616fb22cc5612b269f441`. Các tài liệu/script untracked của người
dùng vẫn nguyên trạng. [Patch review](safeview-cs114-demo.patch) áp dụng lên commit
này; mặc định `CS114_DEMO = null`, nên chưa kích hoạt mô hình chưa được đánh giá.
Patch đã được áp dụng vào bản sao tracked riêng trong thư mục tạm, vượt qua
TypeScript type-check và 63 test provider/routing/fetch/overlay/concealment,
trong đó 8 test mới kiểm tra policy, đổi revision, routing và lỗi API.

## Hợp đồng API và quyết định ẩn

Ứng dụng `deploy/hf_space/app.py` cài package `safeview_ml` từ cùng source snapshot
đã train. Pipeline serialized sở hữu preprocessing; Space không chép lại hàm
chuẩn hóa. Artifact nằm ở `artifacts/RUN_ID/<family>/`, gồm `pipeline.joblib`,
`decision_policy.json`, `metadata.json`. Loader kiểm tra SHA-256 trước khi đọc
joblib và đối chiếu revision/policy. Chỉ tải artifact từ nguồn tin cậy: hash
không xác thực tác giả và joblib có thể thực thi mã Python.

Luồng giữ tương thích với `src/provider/fetch-util.ts` hiện tại:

1. POST `/gradio_api/call/classify`, body `{"data":["bình luận"]}`.
2. Đọc `event_id` từ JSON.
3. GET `/gradio_api/call/classify/<event_id>` và đọc SSE.
4. Khi nhận `event: complete`, lấy phần tử đầu của mảng JSON trong `data:`.

Phần tử đó là map phẳng `{ "CLEAN": p0, "OFFENSIVE": p1, "HATE": p2 }`, không
có wrapper `prediction` hoặc `scores`. Các giá trị hữu hạn, trong [0,1], tổng
xấp xỉ 1. Loader căn cột theo `pipeline.classes_`, không mặc định thứ tự array.
Map nhãn luôn 0=CLEAN, 1=OFFENSIVE, 2=HATE.

Endpoint `/decision` nhận cùng input, trả:

| Trường | Ý nghĩa |
|---|---|
| `scores` | Map xác suất ba lớp |
| `label` | Argmax ba lớp, dùng đánh giá Macro-F1 |
| `confidence` | Xác suất của nhãn argmax, không phải mức độ độc hại |
| `p_harm` | P(OFFENSIVE) + P(HATE), chặn sai số số thực vượt 1 |
| `should_hide` | `p_harm >= threshold` của policy đã khóa |
| `model_revision` | Định danh artifact được chọn trước test |
| `policy_version` | Phiên bản schema/quy tắc policy |

Ví dụ **chỉ để kiểm thử giao thức**: CLEAN=0,40; OFFENSIVE=0,35; HATE=0,25; ngưỡng
0,60. Nhãn vẫn CLEAN, nhưng quyết định ẩn là true. Không thêm điều kiện argmax phải
là OFFENSIVE/HATE. Khi policy chọn ngưỡng `nextafter(1,+inf)`, không ẩn bình luận
nào, kể cả `p_harm=1`; không clamp ngưỡng về 1. Không dùng lại preset 0,30/0,05
hoặc ngưỡng PhoBERT. Endpoint `/policy` nhận `{"data":[]}` và trả ngưỡng, công thức,
revision, label mapping và giới hạn độ dài.

Input phải là string không rỗng và dài tối đa 20.000 ký tự Python. Input sai, model
không sẵn sàng, timeout hoặc SSE error là lỗi; không tạo một dự đoán CLEAN. Gradio
dùng queue tối đa 64 và hai tác vụ inference đồng thời. Không log raw text trong
mã ứng dụng. Cần xem lại retention/access log của nơi host trước khi triển khai.

## Chuẩn bị và publish đúng artifact

Sau khi EDA, training, chọn model/ngưỡng và final test hoàn thành, chạy trong đúng
môi trường Python đã train:

```sh
python scripts/publish_hf.py \
  --release-dir artifacts/RUN_ID/FAMILY \
  --evaluation-dir results/final/RUN_ID \
  --output-dir artifacts/hf-bundle-RUN_ID
```

RUN_ID và FAMILY trong lệnh là tham số cần thay bằng kết quả thật. Lệnh mặc định
chỉ tạo bundle local để review. Script từ chối dữ liệu synthetic/không rõ nguồn,
test chưa hoàn tất, hash/revision khác kết quả test, source snapshot hoặc phiên
bản runtime đã đổi. Bundle chứa wheel đúng source, dependencies pin phiên bản,
model card, aggregate test metrics và Space app. Không chứa raw dataset hoặc
dự đoán từng mẫu.

Để upload, chạy với output directory mới và thêm `--publish --model-repo OWNER/MODEL
--space-repo OWNER/SPACE --acknowledge-data-rights`. Chỉ dùng cờ xác nhận quyền sau
khi làm rõ điều kiện sử dụng/phát hành dataset. Mặc định tạo repo private; `--public`
là lựa chọn riêng. Không tự đổi visibility của repo có sẵn. Script ghi commit model
Hub vào `space/deployment.json`, rồi upload Space. Upload thành công chưa chứng
minh Space khởi động hoặc API hoạt động; xem `publication.json` và kiểm tra runtime.

Nếu model private, Space cần secret `HF_TOKEN` có quyền **read** model, được cấu
hình trên server. Token đăng nhập ở máy local không tự trở thành secret của Space.
Extension không thể gọi trực tiếp private Space mà không có cơ chế xác thực;
không nhúng token vào bundle extension. Demo public chỉ tiến hành sau khi rõ quyền,
hoặc dùng backend giữ secret cho endpoint private. Xác minh quyền tạo Space,
compute/cost và sleep trên tài khoản dự định dùng.

Chạy local với artifact thật:

```sh
SAFEVIEW_RELEASE_DIR=artifacts/RUN_ID/FAMILY python deploy/hf_space/app.py
```

Tải từ Hub yêu cầu `SAFEVIEW_MODEL_REPO` và `SAFEVIEW_MODEL_REVISION` là full commit
SHA 40 ký tự. Model revision nội bộ và Hub commit SHA là hai định danh khác nhau:
revision nội bộ gắn pipeline/policy; Hub commit gắn snapshot repository.

## Áp dụng patch khi đã có release

Trên nhánh demo riêng của SafeView, kiểm tra patch trước khi áp dụng. Những lệnh
sau là bước triển khai **chưa được chạy lên repo SafeView thật**:

```sh
git switch -c codex/cs114-model-integration
git apply --check ../cs114-ml-project/docs/safeview-cs114-demo.patch
git apply ../cs114-ml-project/docs/safeview-cs114-demo.patch
```

Điền `src/settings/cs114-demo.ts` bằng cấu hình thật từ bundle:

- `spaceUrl`: HTTPS origin thực tế của Space, không có `/gradio_api` phía sau.
- `modelUrl`, `modelName`: link model card và tên thuật toán thực sự được chọn.
- `modelRevision`, `policyVersion`, `threshold`: lấy nguyên từ `decision_policy.json`.
- `forceVietnamese`: true cho demo bình luận Việt, kể cả không dấu; false để giữ
  routing theo heuristic. True sẽ đưa cả input tiếng Anh vào model Việt, nên
  chỉ dùng cho tập demo tiếng Việt có chủ đích.

Patch dùng `/decision` cho nhánh CS114 để đối chiếu release identity và kiểm tra
quyết định server với policy local. `/classify` vẫn giữ nguyên cho client cũ.
Nếu Space bị cập nhật khác revision, adapter báo lỗi thay vì dùng ngưỡng cũ.
Luật ẩn CS114 bỏ qua preset confidence cũ, giữ nguyên raw label ngay cả khi quyết
định ẩn khác argmax. Lexicon override được bỏ qua ở nhánh nghiên cứu; báo kết quả
toàn hệ thống có lexicon thành thí nghiệm riêng nếu muốn bổ sung.

Demo gửi text gốc đến pipeline, tránh NFKC, leetspeak hoặc xóa zero-width ngoài
quy trình đã đánh giá. Cache key gắn model revision, policy version, ngưỡng và
chế độ routing. Build Vite thêm đúng origin vào manifest host permissions và đổi
tên thành bản demo. UI hiển thị đúng model card và tổng xác suất harmful; không
trình bày confidence như severity. Không diễn giải legacy `severity: low` trong
schema nội bộ thành đánh giá mức độ thực tế cho CS114.

Sau khi cấu hình, chạy `npm run type-check`, `npm test`, `npm run build`; reload
extension và các trang demo. Đo riêng: bình luận mới khi cuộn trang, cache hit,
tiếng Việt không dấu, sạch/xúc phạm/thù ghét, Space ngủ, timeout và API error.
Luồng lỗi giữ trạng thái error của extension, không tính là CLEAN. Log/audit cũ
có `hidden:false` khi error; đó không phải dự đoán model và phải loại khỏi mẫu
đã phân loại khi tính precision/recall. Mã provider cũ log raw text đã được xóa
trong patch; không bật lại khi demo.

## Đo hiệu năng và xác nhận hoàn thành

`scripts/benchmark_api.py` đo HTTP POST + GET/SSE của endpoint bằng câu do nhóm
tự viết, lưu first-request latency, warm p50/p95, thông lượng và số lỗi. Lần gọi
đầu không được gọi là cold start nếu chưa xác nhận Space đã ngủ. Chỉ số network
khác benchmark inference CPU trong package. Muốn đo precision/recall/FPR end-to-end,
dùng tập có nhãn và nguồn được phép, cùng rule đã khóa; script độ trễ không tự
cung cấp đánh giá chất lượng.

```sh
python scripts/benchmark_api.py --space-url "$SPACE_URL" \
  --repetitions 30 --concurrency 2 --timeout 60 \
  --output results/api-benchmark-RUN_ID.json
```

Thời hạn timeout bao trùm cả POST, thời gian trong queue và GET/SSE; heartbeat
không kéo dài vô hạn deadline. Báo cáo không lưu câu gửi đi. Nếu có request lỗi,
script vẫn lưu báo cáo rồi trả exit code 1.

Chỉ cập nhật tài liệu là “đã tích hợp” sau khi có Space URL, model Hub SHA, Space
commit, SafeView integration commit, policy version, test contract thực tế và
bảng đo end-to-end. Không suy ra mô hình mới hơn PhoBERT chỉ từ bảng so sánh ba
thuật toán truyền thống.

Nguồn kỹ thuật: [Gradio event API](https://gradio.app/guides/querying-gradio-apps-with-curl),
[Hub upload](https://huggingface.co/docs/huggingface_hub/guides/upload),
[Hub pinned download](https://huggingface.co/docs/huggingface_hub/guides/download),
[Space configuration](https://huggingface.co/docs/hub/spaces-config-reference).
