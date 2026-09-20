#!/usr/bin/env python3
"""Measure the real POST/event/SSE path; no classifier quality claims.

Default text is authored synthetic input, not a ViHSD sample. The first request
is reported separately and is only called cold when the operator confirms it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import httpx
import numpy as np

from safeview_ml.policy import LABEL_NAMES, validate_probabilities

DEFAULT_TEXT = "Đây là bình luận minh họa do nhóm tự viết để kiểm tra độ trễ."


async def call_gradio(client, base_url, api_name, data, *, timeout=60):
    """Read named-event SSE with a deadline covering submission and the queue."""
    if api_name not in {"classify", "decision", "policy"}:
        raise ValueError("Unsupported endpoint name")
    endpoint = f"{base_url.rstrip('/')}/gradio_api/call/{api_name}"
    async with asyncio.timeout(timeout):
        response = await client.post(endpoint, json={"data": data})
        response.raise_for_status()
        event_id = response.json().get("event_id")
        if not isinstance(event_id, str) or not event_id or not all(c.isalnum() or c in "-_" for c in event_id):
            raise ValueError("Invalid Gradio event_id")
        async with client.stream("GET", f"{endpoint}/{event_id}") as response:
            response.raise_for_status()
            event, payload = "", []
            async for line in response.aiter_lines():
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    payload.append(line[5:].lstrip())
                elif not line:
                    if event == "error":
                        raise RuntimeError("Gradio returned an SSE error event")
                    if event == "complete":
                        result = json.loads("\n".join(payload))
                        if not isinstance(result, list) or len(result) != 1:
                            raise ValueError("Expected exactly one Gradio output")
                        return result[0]
                    event, payload = "", []
    raise RuntimeError("SSE stream ended before completion")


async def benchmark(base_url, *, text=DEFAULT_TEXT, repetitions=20, concurrency=1, timeout=60, confirmed_cold_start=False):
    parsed = urlparse(base_url)
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username
            or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}):
        raise ValueError("space URL must be an HTTP(S) origin without credentials, path or query")
    if repetitions < 1 or not 1 <= concurrency <= 8 or timeout <= 0:
        raise ValueError("Require repetitions >= 1, concurrency in [1,8], timeout > 0")
    gate = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        async def attempt(index):
            async with gate:
                started = time.perf_counter()
                try:
                    scores = await call_gradio(client, base_url, "classify", [text], timeout=timeout)
                    if not isinstance(scores, dict) or set(scores) != set(LABEL_NAMES):
                        raise ValueError("Classify did not return the flat canonical score map")
                    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scores.values()):
                        raise ValueError("Classify scores must be numeric probabilities")
                    validate_probabilities([[scores[label] for label in LABEL_NAMES]])
                    error = None
                except Exception as exc:
                    # Save the error kind, not a body/URL that might contain text
                    # or authentication details. Failure is never a CLEAN score.
                    error = type(exc).__name__
                return {"request_index": index, "latency_ms": (time.perf_counter() - started) * 1000,
                        "success": error is None, "error_type": error}

        first = await attempt(0)
        warm_start = time.perf_counter()
        warm = await asyncio.gather(*(attempt(index + 1) for index in range(repetitions)))
        warm_seconds = time.perf_counter() - warm_start
        try:
            policy = await call_gradio(client, base_url, "policy", [], timeout=timeout)
            identity = {key: policy[key] for key in ("model_revision", "policy_version", "threshold")}
        except Exception as exc:
            identity = {"identity_verified": False, "error_type": type(exc).__name__}
    latencies = [entry["latency_ms"] for entry in warm if entry["success"]]
    return {
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "space_origin": base_url.rstrip("/"),
        "input_description": "authored_synthetic_input" if text == DEFAULT_TEXT else "operator_supplied_input_not_saved",
        "input_character_count": len(text),
        "concurrency": concurrency,
        "request_deadline_seconds": timeout,
        "first_request_kind": "operator_confirmed_cold_start" if confirmed_cold_start else "first_request_cold_state_unknown",
        "first_request": first,
        "warm_request_count": repetitions,
        "warm_success_count": len(latencies),
        "warm_error_count": repetitions - len(latencies),
        "warm_p50_ms": float(np.percentile(latencies, 50)) if latencies else None,
        "warm_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
        "warm_success_per_second": len(latencies) / warm_seconds,
        "warm_wall_seconds": warm_seconds,
        "release_identity_after_measurement": identity,
        "requests": warm,
        "limitations": [
            "Repeated single text measures latency only, not precision, recall or FPR.",
            "Warm latency includes HTTP submission, server queue, inference and SSE completion.",
            "Release identity is queried after timing and does not prove it remained unchanged during every call.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text sent to the server; not stored in the report")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--confirmed-cold-start", action="store_true", help="Only set after independently verifying the Space is sleeping/stopped")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing benchmark report")
    result = asyncio.run(benchmark(args.space_url, text=args.text, repetitions=args.repetitions,
                                  concurrency=args.concurrency, timeout=args.timeout,
                                  confirmed_cold_start=args.confirmed_cold_start))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved benchmark: {args.output}; warm successes {result['warm_success_count']}/{args.repetitions}")
    if not result["first_request"]["success"] or result["warm_error_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
