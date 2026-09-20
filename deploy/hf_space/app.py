"""Gradio entry point. No duplicate preprocessing or fallback CLEAN verdicts."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from safeview_ml.inference import RELEASE_FILES, ReleasePredictor


def resolve_release_dir() -> Path:
    """Prefer local artifacts; a remote release must use a full Hub commit SHA."""
    local = os.environ.get("SAFEVIEW_RELEASE_DIR")
    if local:
        return Path(local).expanduser().resolve()
    config_path = Path(__file__).with_name("deployment.json")
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    repo_id = os.environ.get("SAFEVIEW_MODEL_REPO") or config.get("model_repo")
    revision = os.environ.get("SAFEVIEW_MODEL_REVISION") or config.get("hub_revision")
    if not repo_id or not isinstance(revision, str) or not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise RuntimeError(
            "Set SAFEVIEW_RELEASE_DIR, or SAFEVIEW_MODEL_REPO and "
            "SAFEVIEW_MODEL_REVISION to a full 40-character Hub commit SHA."
        )
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, revision=revision, allow_patterns=list(RELEASE_FILES)))


def create_app(predictor: ReleasePredictor | None = None) -> Any:
    import gradio as gr

    predictor = predictor or ReleasePredictor(resolve_release_dir())

    def classify(text: str) -> dict[str, float]:
        # Deliberately let validation/inference errors become SSE error events.
        return predictor.predict_scores(text)

    def decision(text: str) -> dict[str, Any]:
        return predictor.predict(text)

    def policy() -> dict[str, Any]:
        return {
            "threshold": predictor.threshold,
            "score_definition": "P(OFFENSIVE)+P(HATE)",
            "model_revision": predictor.model_revision,
            "policy_version": predictor.policy_version,
            "label_mapping": predictor.policy["label_mapping"],
            "max_text_length": predictor.max_text_length,
        }

    with gr.Blocks(analytics_enabled=False, title="SafeView · CS114") as demo:
        gr.Markdown(
            "# SafeView · Phân loại bình luận tiếng Việt\n"
            "Bản nghiên cứu phân biệt CLEAN, OFFENSIVE và HATE. "
            "Nhãn dự đoán và quyết định ẩn là hai đầu ra riêng. "
            "Văn bản được xử lý trên máy chủ; tránh gửi thông tin cá nhân."
        )
        text = gr.Textbox(label="Bình luận", lines=4, max_lines=10)
        scores = gr.JSON(label="Xác suất ba lớp")
        details = gr.JSON(label="Nhãn và quyết định theo policy đã khóa")
        # preprocess=False preserves the raw API type so malformed input is
        # rejected instead of Textbox coercing a number/object into a string.
        gr.Button("Phân loại").click(
            classify, inputs=text, outputs=scores, api_name="classify",
            preprocess=False, concurrency_limit=2, concurrency_id="inference",
        )
        gr.Button("Xem quyết định ẩn").click(
            decision, inputs=text, outputs=details, api_name="decision",
            preprocess=False, concurrency_limit=2, concurrency_id="inference",
        )
        gr.Button("Thông tin policy").click(policy, inputs=None, outputs=details, api_name="policy")
    return demo.queue(max_size=64, default_concurrency_limit=2)


if __name__ == "__main__":
    create_app().launch(server_name="0.0.0.0", show_error=True)
