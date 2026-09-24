# Tích hợp bản demo CS114 với SafeView

Nếu bắt đầu từ training, thực hiện theo [hướng dẫn Colab → Hugging Face → SafeView](train-deploy-guide.md). Tài liệu dưới đây mô tả chi tiết artifact, hợp đồng API và patch tích hợp.

## Trạng thái thực tế

Mục tiêu triển khai là **BamiBERT fine-tune trên ViHSD**, được chốt trước test theo
yêu cầu người dùng. `validation_best_family` báo model đứng đầu dev riêng; BamiBERT
không bắt buộc phải thắng PhoBERT. Mã inference, Space template và patch đã chuẩn bị,
nhưng chưa có model ViHSD đã đánh giá hoặc URL Space mới. Người dùng tự chạy train,
test và publish; chưa đổi repo SafeView hoặc đo độ trễ mạng thực tế. Smoke test tổng
hợp và Transformer nhỏ/offline chỉ kiểm tra phần mềm, không đo chất lượng tiếng Việt.

Đã kiểm tra repo cùng cấp `/Users/thong/Data/Projects/safe-view` tại commit
`7e09b01569365c6d173616fb22cc5612b269f441`. Các tài liệu/script untracked của người
dùng vẫn nguyên trạng. [Patch review](safeview-cs114-demo.patch) áp dụng lên commit
này; mặc định `CS114_DEMO = null`, nên chưa kích hoạt mô hình chưa được đánh giá.
Kiểm chứng patch được thực hiện trên bản sao tracked riêng, không phải bản đang
chạy của người dùng; xem ngày và phạm vi trong `docs/verification.json`. Khi áp dụng
lên revision SafeView khác, cần kiểm tra lại TypeScript, provider, routing và UI.

## Hợp đồng API và quyết định ẩn

Ứng dụng `scripts/templates/hf_space/app.py` cài package `safeview_ml` từ cùng source snapshot
đã train. Artifact sở hữu preprocessing; Space không chép lại hàm chuẩn hóa. Release BamiBERT
ở `artifacts/RUN_ID/bamibert/`, gồm model/tokenizer Hugging Face,
`transformer_config.json`, `decision_policy.json`, `metadata.json`. BamiBERT nhận text
chưa tách từ; PhoBERT dùng PyVi nếu triển khai trong thí nghiệm khác. Backend baseline
vẫn nhận `pipeline.joblib`. Loader kiểm tra SHA-256 của mọi file inference và đối
chiếu revision/policy trước khi load. Chỉ dùng nguồn tin cậy; hash không xác thực tác
giả, và joblib baseline có thể thực thi mã Python. Space dùng CPU, không yêu cầu GPU
chỉ để load Transformer; độ trễ phải đo thực tế.

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
dùng queue tối đa 64, mặc định một tác vụ inference đồng thời để hạn chế tải CPU;
`SAFEVIEW_CONCURRENCY` chỉ tăng sau khi đo hiệu năng. Không log raw text trong
mã ứng dụng. Cần xem lại retention/access log của nơi host trước khi triển khai.

## Chuẩn bị và publish đúng artifact

Sau khi EDA, training, khóa BamiBERT/ngưỡng và final test hoàn thành, chạy trong đúng
môi trường Python đã train:

```sh
python scripts/publish_hf.py \
  --release-dir artifacts/RUN_ID/bamibert \
  --evaluation-dir results/final/RUN_ID \
  --output-dir deploy/bundles/RUN_ID
```

RUN_ID trong lệnh phải thay bằng run thật; family triển khai mặc định là `bamibert`. Lệnh mặc định
chỉ tạo bundle local để review. Script từ chối dữ liệu synthetic/không rõ nguồn,
test chưa hoàn tất, hash/revision khác kết quả test, source snapshot hoặc phiên
bản runtime đã đổi. Bundle chứa wheel đúng source, dependencies pin phiên bản,
model/tokenizer, model card, aggregate test metrics và Space app. Bundle CPU dùng
Torch CPU cùng release, không chép bộ CUDA của Colab sang Space. Phiên bản Gradio
được pin tương thích với Transformers và huggingface-hub; không tự nâng riêng một
package khi resume/đóng gói. Không chứa raw dataset hoặc
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
hoặc dùng backend giữ secret cho endpoint private. Xác minh quyền tạo Space, compute/cost và sleep trên tài khoản dự định dùng. Theo
[tài liệu Spaces](https://huggingface.co/docs/hub/spaces-overview) và
[bảng giá](https://huggingface.co/pricing), tạo Gradio/Docker Space thông thường hiện
yêu cầu gói trả phí; PRO 9 USD/tháng, CPU Basic 2 vCPU/16 GB RAM không tính tiền
phần cứng theo giờ. [ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu) miễn phí
có điều kiện/quota/hàng đợi, không phải bảo đảm phục vụ extension liên tục.

Chạy local với artifact thật:

```sh
SAFEVIEW_RELEASE_DIR=artifacts/RUN_ID/bamibert python scripts/templates/hf_space/app.py
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

Sau khi Space đã deploy và kiểm tra API, tạo file cấu hình từ bundle đã xác minh:

```sh
python scripts/export_extension_config.py \
  --bundle-dir deploy/bundles/RUN_ID \
  --space-url https://OWNER-SPACE.hf.space \
  --model-repo OWNER/MODEL \
  --output deploy/bundles/RUN_ID/cs114-demo.ts
```

Lệnh này không truy cập mạng, không gửi token và không sửa repo SafeView. Thay URL và
repo bằng giá trị thật; chép file đã review vào `src/settings/cs114-demo.ts` của nhánh
demo sau khi áp dụng patch. Script từ chối ghi đè file đầu ra. Các trường lấy từ
artifact hoặc cấu hình người dùng:

- `spaceUrl`: HTTPS origin thực tế của Space, không có `/gradio_api` phía sau.
- `modelUrl`, `modelName`: link model card và tên thuật toán thực sự được chọn.
- `modelRevision`, `policyVersion`, `threshold`: lấy nguyên từ `decision_policy.json`.
- `timeoutMs`: mặc định 60.000 ms, có thể đặt bằng `--timeout-ms`; đo cold/warm để
  điều chỉnh. Hết thời hạn là lỗi, không gán nhãn CLEAN.
- `forceVietnamese`: true cho demo bình luận Việt, kể cả không dấu; dùng
  `--automatic-routing` để đặt false và giữ
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
bảng đo end-to-end. Bảng nghiên cứu có SVM/LR/PhoBERT/BamiBERT; kết luận chất lượng phải dựa
trên thực nghiệm thật, không dựa tuổi model hoặc mục tiêu triển khai.

Nguồn kỹ thuật: [Gradio event API](https://gradio.app/guides/querying-gradio-apps-with-curl),
[Hub upload](https://huggingface.co/docs/huggingface_hub/guides/upload),
[Hub pinned download](https://huggingface.co/docs/huggingface_hub/guides/download),
[Space configuration](https://huggingface.co/docs/hub/spaces-config-reference).
