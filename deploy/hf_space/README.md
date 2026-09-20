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
Do not copy a `.joblib` into a Space without that package: serialized custom
transformers must resolve to the training implementation.

For local testing, install the project and its serving dependencies, then run
from the project root:

```sh
SAFEVIEW_RELEASE_DIR=artifacts/RUN_ID/FAMILY python deploy/hf_space/app.py
```

Set `SAFEVIEW_RELEASE_DIR` to a trusted release directory with `pipeline.joblib`,
`metadata.json`, and `decision_policy.json`. Remote loading instead requires
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

The app shares a concurrency limit of two inference calls and a queue of 64.
Account eligibility, costs, actual cold/warm latency and error handling must be
checked on the intended Space; none are inferred from a local test.

References: [Gradio event API](https://gradio.app/guides/querying-gradio-apps-with-curl),
[Space configuration](https://huggingface.co/docs/hub/spaces-config-reference),
[pinned Hub downloads](https://huggingface.co/docs/huggingface_hub/guides/download).
