"""Ollama logging proxy.

Listens on PROXY_PORT, forwards every request to OLLAMA_URL, and logs
per-request token/timing stats from /api/generate and /api/chat into
dashboard/data/usage.db.

Run:
    uvicorn main:app --host 127.0.0.1 --port 11435
    (or just: python main.py)
"""

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse

from db import init_db, insert_request

OLLAMA_URL = "http://localhost:11434"
PROXY_HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("PROXY_PORT", "11435"))
INSTRUMENTED_PATHS = {"/api/generate", "/api/chat"}

app = FastAPI(title="Ollama Logging Proxy")
init_db()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stats_from_done(done: dict[str, Any]) -> dict[str, Any]:
    """Convert an Ollama `done: true` payload into a row for the DB."""
    eval_count = done.get("eval_count") or 0
    eval_duration = done.get("eval_duration") or 0
    prompt_eval_count = done.get("prompt_eval_count") or 0
    prompt_eval_duration = done.get("prompt_eval_duration") or 0
    output_tps = (eval_count / eval_duration * 1_000_000_000) if eval_duration > 0 else None
    prompt_tps = (
        (prompt_eval_count / prompt_eval_duration * 1_000_000_000)
        if prompt_eval_duration > 0
        else None
    )
    return {
        "prompt_tokens": prompt_eval_count,
        "output_tokens": eval_count,
        "prompt_eval_duration_ns": prompt_eval_duration,
        "eval_duration_ns": eval_duration,
        "load_duration_ns": done.get("load_duration") or 0,
        "total_duration_ns": done.get("total_duration") or 0,
        "output_tps": output_tps,
        "prompt_tps": prompt_tps,
    }


def _log(
    endpoint: str,
    model: str,
    streamed: bool,
    done: dict[str, Any] | None,
    wall_time_sec: float,
    client_ip: str,
) -> None:
    record: dict[str, Any] = {
        "timestamp": _now_iso(),
        "model": model,
        "endpoint": endpoint,
        "streamed": 1 if streamed else 0,
        "wall_time_sec": wall_time_sec,
        "client_ip": client_ip,
    }
    if done:
        record.update(_stats_from_done(done))
    try:
        insert_request(record)
    except Exception as e:
        print(f"[proxy] failed to log request: {e}")


@app.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(full_path: str, request: Request) -> Response:
    target = f"{OLLAMA_URL}/{full_path}"
    if request.url.query:
        target += f"?{request.url.query}"

    method = request.method
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in {"host", "content-length"}}
    client_ip = request.client.host if request.client else ""
    endpoint = "/" + full_path.lstrip("/")

    instrumented = endpoint in INSTRUMENTED_PATHS and method == "POST"
    payload: dict[str, Any] = {}
    if instrumented and body:
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            instrumented = False

    streamed = bool(payload.get("stream", True)) if instrumented else False
    model = payload.get("model", "") if instrumented else ""
    start = time.time()

    client = httpx.AsyncClient(timeout=None)

    if instrumented and streamed:
        req = client.build_request(method, target, content=body, headers=headers)
        upstream = await client.send(req, stream=True)

        async def gen() -> AsyncIterator[bytes]:
            done_payload: dict[str, Any] | None = None
            buffer = b""
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
                    buffer += chunk
                    while b"\n" in buffer:
                        line, _, buffer = buffer.partition(b"\n")
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if obj.get("done"):
                            done_payload = obj
            finally:
                await upstream.aclose()
                await client.aclose()
                _log(endpoint, model, True, done_payload, time.time() - start, client_ip)

        return StreamingResponse(
            gen(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "application/x-ndjson"),
        )

    try:
        resp = await client.request(method, target, content=body, headers=headers)
    finally:
        if not instrumented:
            await client.aclose()

    if instrumented and not streamed:
        try:
            done_payload = resp.json()
        except (json.JSONDecodeError, ValueError):
            done_payload = None
        _log(endpoint, model, False, done_payload, time.time() - start, client_ip)
        await client.aclose()

    excluded = {"content-encoding", "transfer-encoding", "connection", "content-length"}
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded}
    return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)


if __name__ == "__main__":
    import uvicorn

    print(f"Proxy binding on {PROXY_HOST}:{PROXY_PORT} -> Ollama at {OLLAMA_URL}")
    uvicorn.run("main:app", host=PROXY_HOST, port=PROXY_PORT, log_level="info")
