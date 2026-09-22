# Bản nháp phương pháp

`bao-cao-phuong-phap.docx` và `trinh-bay-phuong-phap.pptx` hiện là bản nháp phương pháp bằng tiếng Việt thuộc phạm vi ba baseline cũ. Script tạo báo cáo đã được cập nhật cho SVM, Logistic Regression, PhoBERT và BamiBERT, cùng nguồn ViHSD trực tiếp từ GitHub; hai tài liệu chưa được dựng lại. Chúng không có điểm số ViHSD hoặc tuyên bố mô hình thắng. Cần tạo và rà soát bản mới sau khi người dùng chạy thực nghiệm.

Script `scripts/build_reports.py` tạo Word bằng python-docx và PowerPoint bằng `@oai/artifact-tool`, dùng runtime có sẵn của Codex. Deck chứa text và bảng PowerPoint chỉnh sửa được. Script cũng render từng trang/slide vào thư mục QA riêng, cần xem ảnh trước khi phân phối bản dựng mới.

Chạy trong repository với bundled Python do công cụ workspace dependencies cung cấp:

```bash
/Users/thong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_reports.py --build-dir /tmp/cs114-report-qa --overwrite
```

Có thể đặt các runtime/skill khác qua `--node`, `--node-modules`, `--documents-skill` và `--presentations-skill`. Không cần cài thêm dependency báo cáo vào môi trường huấn luyện. Dùng `--output-dir` mới để lưu bản khác. `--skip-render` chỉ dành cho phát triển, không bỏ yêu cầu xem ảnh trước khi gửi bài.

Khi đã có run dữ liệu thật và kết quả final:

```bash
/Users/thong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_reports.py --run-dir results/runs/VIHSD_RUN --final-dir results/final/VIHSD_RUN --output-dir reports/VIHSD_RUN
```

Script chỉ đọc các bảng đã lưu. Nó từ chối run tổng hợp hoặc thiếu cờ xác nhận, kiểm tra fingerprint và selection hash trước khi ghép test. Nó không train/test lại và không tự viết kết luận từ vài điểm số. Phụ lục tự thêm bảng validation, bảng test và confusion matrix tổng hợp khi có; nhóm cần hoàn thành diễn giải, phân tích lỗi, limitations và rà bố cục bản mới.

Ảnh render, candidate PPTX và biên bản QA là tệp tạm, không phải tài liệu nộp. Không đưa raw dataset hoặc dự đoán từng mẫu vào thư mục này.
