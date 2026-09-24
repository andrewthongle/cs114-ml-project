---
title: SafeView CS114 Vietnamese moderation
emoji: 🛡️
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
python_version: "3.13"
pinned: false
---

# SafeView CS114 inference service

This directory is a source template, **not a published or evaluated model**.
`scripts/publish_hf.py` builds a reviewable bundle only from an evaluated release;
the bundle includes the exact `safeview_ml` wheel and pinned dependency versions.
Both TF-IDF and Transformer releases are supported. The four-model experiment
predeclares BamiBERT for deployment and reports the validation winner separately.
Do not copy weights alone: the package also implements the frozen preprocessing
and decision policy.

For local testing, install the project and its serving dependencies, then run
from the project root:

```sh
SAFEVIEW_RELEASE_DIR=artifacts/RUN_ID/FAMILY python scripts/templates/hf_space/app.py
```

Set `SAFEVIEW_RELEASE_DIR` to a trusted release directory with `metadata.json`
and `decision_policy.json`. TF-IDF releases contain `pipeline.joblib`; Transformer
releases contain `model.safetensors`, tokenizer assets, `config.json`, and
`transformer_config.json`. Every inference asset is checksum-verified before
loading. PhoBERT uses the saved PyVi segmentation setting; BamiBERT uses raw text
with NFC normalization. Remote loading instead requires
`SAFEVIEW_MODEL_REPO` and `SAFEVIEW_MODEL_REVISION` (full 40-character Hub commit
SHA). A generated `deployment.json` supplies these after publishing the model.
Authentication, if required, stays in the server's `HF_TOKEN` secret or HF login;
never embed a token in an extension or this repository.
For a private model the Space needs a read-access `HF_TOKEN` server secret;
local login is not transferred to it. A private Space also requires an
authenticated backend for extension calls. Only make a research demo public
after confirming dataset/model distribution rights.

Endpoints use Gradio's event flow:

```sh
curl -X POST "$SPACE_URL/gradio_api/call/classify" \
  -H 'Content-Type: application/json' -d '{"data":["Một bình luận minh họa"]}'
curl -N "$SPACE_URL/gradio_api/call/classify/$EVENT_ID"
```

The complete SSE payload is an array containing a flat `CLEAN`, `OFFENSIVE`,
`HATE` probability map. `/decision` returns the map plus argmax `label`,
`confidence`, summed `p_harm`, `should_hide`, `model_revision`, and
`policy_version`. `/policy` takes `{"data":[]}` and returns the locked threshold
and release identity. Scores and confidence are not severity measurements.
Invalid input or inference failure yields an error, never a fabricated CLEAN
prediction. Raw input is not logged by application code; review hosting access
logs and retention separately before a public deployment.

The app defaults to CPU, one inference call at a time, and a queue of 64.
`SAFEVIEW_CONCURRENCY` controls concurrency; `SAFEVIEW_DEVICE` defaults to `cpu`.
The exporter uses the same PyTorch release's CPU wheel and excludes CUDA packages
when preparing a Transformer Space from a Colab GPU environment. This keeps the
source and model identical, but end-to-end latency must still be measured on the
actual host. Gradio 6.9.0 and Transformers 4.57.x are pinned to share compatible
Hugging Face Hub dependencies; do not upgrade one independently.

Account eligibility, costs, actual cold/warm latency and error handling must be
checked on the intended Space; none are inferred from a local test.
Current HF documentation requires a paid account plan to create ordinary
Gradio/Docker compute Spaces, even though CPU Basic has no hourly hardware fee.
Eligible free accounts can use limited ZeroGPU Spaces; this template targets CPU
Basic and does not claim ZeroGPU support or an always-available free endpoint.

References: [Gradio event API](https://gradio.app/guides/querying-gradio-apps-with-curl),
[Space configuration](https://huggingface.co/docs/hub/spaces-config-reference),
[pinned Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
