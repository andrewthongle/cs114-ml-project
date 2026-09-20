"""Exercise the actual Gradio POST/event/SSE wire contract without the network."""

import importlib.util
import json
from pathlib import Path

import pytest

gr = pytest.importorskip("gradio")
from fastapi import FastAPI
from fastapi.testclient import TestClient

from safeview_ml.inference import ReleasePredictor
from test_inference import write_release

APP_PATH = Path(__file__).resolve().parents[1] / "deploy" / "hf_space" / "app.py"
spec = importlib.util.spec_from_file_location("safeview_space_app", APP_PATH)
space_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(space_app)


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    predictor = ReleasePredictor(write_release(tmp_path_factory.mktemp("api-release")))
    app = gr.mount_gradio_app(FastAPI(), space_app.create_app(predictor), path="/")
    with TestClient(app) as client:
        yield client


def call_event(client, name, data):
    submit = client.post(f"/gradio_api/call/{name}", json={"data": data})
    assert submit.status_code == 200, submit.text
    event_id = submit.json()["event_id"]
    assert isinstance(event_id, str) and event_id
    response = client.get(f"/gradio_api/call/{name}/{event_id}")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    return response.text


def completed_payload(response):
    for event in response.split("\n\n"):
        if "event: complete" in event:
            data = next(line[6:] for line in event.splitlines() if line.startswith("data: "))
            return json.loads(data)[0]
    pytest.fail(f"Missing SSE complete event: {response}")


def test_classify_is_flat_scores_inside_named_complete_event(api_client):
    data = completed_payload(call_event(api_client, "classify", ["Bình luận kiểm thử"]))
    assert data == {"CLEAN": 0.4, "OFFENSIVE": 0.35, "HATE": 0.25}
    assert set(data) == {"CLEAN", "OFFENSIVE", "HATE"}


def test_decision_keeps_classification_and_policy_separate(api_client):
    data = completed_payload(call_event(api_client, "decision", ["Bình luận kiểm thử"]))
    assert data["label"] == "CLEAN"
    assert data["p_harm"] == 0.6
    assert data["should_hide"] is True
    assert data["policy_version"] == "1"


def test_policy_exposes_release_identity(api_client):
    data = completed_payload(call_event(api_client, "policy", []))
    assert data["model_revision"] == "test-revision"
    assert data["threshold"] == 0.6
    assert data["score_definition"] == "P(OFFENSIVE)+P(HATE)"


@pytest.mark.parametrize("text", [" \n", 123, "x" * 20_001])
def test_invalid_input_returns_error_event_never_clean(api_client, text):
    response = call_event(api_client, "classify", [text])
    assert "event: error" in response
    assert "event: complete" not in response


def test_remote_configuration_requires_immutable_revision(monkeypatch):
    monkeypatch.delenv("SAFEVIEW_RELEASE_DIR", raising=False)
    monkeypatch.setenv("SAFEVIEW_MODEL_REPO", "owner/repo")
    monkeypatch.setenv("SAFEVIEW_MODEL_REVISION", "main")
    with pytest.raises(RuntimeError, match="40-character"):
        space_app.resolve_release_dir()


def load_benchmark():
    spec = importlib.util.spec_from_file_location("benchmark_api", APP_PATH.parents[2] / "scripts/benchmark_api.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("event", ["complete", "error"])
def test_benchmark_sse_protocol(event):
    import asyncio
    import httpx

    benchmark = load_benchmark()

    def transport(request):
        if request.method == "POST":
            assert json.loads(request.content) == {"data": ["Test"]}
            return httpx.Response(200, json={"event_id": "fixture-id"})
        assert request.url.path == "/gradio_api/call/classify/fixture-id"
        return httpx.Response(200, text=(
            'event: heartbeat\ndata: null\n\n'
            f'event: {event}\ndata: [{{"CLEAN":0.4,"OFFENSIVE":0.35,"HATE":0.25}}]\n\n'
        ))

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            return await benchmark.call_gradio(client, "https://fixture.invalid", "classify", ["Test"])

    if event == "complete":
        assert asyncio.run(run())["CLEAN"] == 0.4
    else:
        with pytest.raises(RuntimeError, match="SSE error"):
            asyncio.run(run())


def test_benchmark_has_overall_queue_deadline():
    import asyncio
    import httpx

    benchmark = load_benchmark()

    async def transport(request):
        await asyncio.sleep(0.1)
        return httpx.Response(200, json={"event_id": "fixture-id"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
            await benchmark.call_gradio(client, "https://fixture.invalid", "classify", ["Test"], timeout=0.001)

    with pytest.raises(TimeoutError):
        asyncio.run(run())
