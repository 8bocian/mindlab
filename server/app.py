"""
MindLab server: FastAPI + WebSocket around engine.Engine.

Several people can use one server at once:
  * every WebSocket connection has its own steering (sliders affect only that user's answers),
  * concept vectors live in a shared library (computed once, reused by everyone),
  * generations run one at a time on the GPU, in a FIFO queue; waiting users see their position.

Env vars:
  MODEL              HF model id or local path (default Qwen/Qwen3-8B)
  DTYPE              auto | bfloat16 | float16 | float32   (default auto)
  DEVICE_MAP         auto | cuda | cpu | none             (default auto)
  TRUST_REMOTE_CODE  1 to allow custom model code from the repo
  LAYERS_PATH        override, e.g. model.language_model.layers
  NORM_PATH          override, e.g. model.language_model.norm
  DEFAULT_CONCEPTS   comma list built at startup (default: ocean,dogs,fear,Poland)
  HEARTBEAT_FILE     touched on activity (used by watchdog.py)
  ACCESS_PASSWORD    if set, required for /api/* (Bearer header) and /ws (?token=)
  ALLOWED_ORIGINS    comma list of frontend origins for CORS (default *)
  MAX_NEW_TOKENS_CAP upper limit for one answer (default 400)
  QUEUE_MAX          max waiting generations (default 20)
  HOST / PORT        default 127.0.0.1:8000  (keep 127.0.0.1; expose via SSH or Cloudflare tunnel)
"""
from __future__ import annotations

import asyncio
import collections
import hmac
import os
import pathlib
import threading
import traceback

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

HERE = pathlib.Path(__file__).parent
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
HEARTBEAT = pathlib.Path(os.environ.get("HEARTBEAT_FILE", "/tmp/mindlab_heartbeat"))
DEFAULT_CONCEPTS = [c.strip() for c in os.environ.get("DEFAULT_CONCEPTS", "ocean,dogs,fear,Poland").split(",") if c.strip()]
PASSWORD = os.environ.get("ACCESS_PASSWORD", "")
ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()] or ["*"]
MAX_TOKENS_CAP = int(os.environ.get("MAX_NEW_TOKENS_CAP", "400"))
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "20"))

app = FastAPI(title="MindLab")
state = {"engine": None, "status": "loading", "error": None, "progress": "starting"}


def beat():
    try:
        HEARTBEAT.touch()
    except OSError:
        pass


def token_ok(tok: str | None) -> bool:
    if not PASSWORD:
        return True
    return bool(tok) and hmac.compare_digest(tok.encode(), PASSWORD.encode())


# ----------------------------------------------------------------------------- auth + CORS
@app.middleware("http")
async def auth(request: Request, call_next):
    path = request.url.path
    if path.startswith("/api/") and path != "/api/health" and request.method != "OPTIONS":
        h = request.headers.get("authorization", "")
        if not token_ok(h[7:] if h.lower().startswith("bearer ") else None):
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)


# added after the auth middleware -> outermost, so 401 responses also carry CORS headers
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST", "OPTIONS"],
                   allow_headers=["Authorization", "Content-Type"])


# ----------------------------------------------------------------------------- generation queue
class Job:
    def __init__(self, req: dict, steer: dict, concepts: list[str] | None, loop, out: asyncio.Queue):
        self.req, self.steer, self.concepts = req, steer, concepts
        self.loop, self.out = loop, out
        self.stop = False

    def emit(self, msg: dict):
        self.loop.call_soon_threadsafe(self.out.put_nowait, msg)


waiting: collections.deque[Job] = collections.deque()
running: dict = {"job": None}
cv = threading.Condition()


def _announce_positions():
    """Call with cv held: tell every waiting job how many generations are ahead of it."""
    ahead = 1 if running["job"] else 0
    for i, j in enumerate(waiting):
        j.emit({"type": "queued", "ahead": ahead + i})


def _cancel(job: Job) -> bool:
    with cv:
        if job in waiting:
            waiting.remove(job)
            _announce_positions()
            return True
    return False


def _run(job: Job):
    e = state["engine"]
    req = job.req
    try:
        read = req.get("read")
        read_site = (int(read["l"]), int(read["p"])) if read else None
        text = e.generate(
            req.get("messages", []),
            max_new_tokens=max(1, min(int(req.get("max_new_tokens", 200)), MAX_TOKENS_CAP)),
            temperature=float(req.get("temperature", 0.7)),
            top_p=float(req.get("top_p", 0.95)),
            enable_thinking=req.get("thinking"),
            read_site=read_site,
            on_token=lambda t: (beat(), job.emit({"type": "token", **t})),
            should_stop=lambda: job.stop,
            steer=job.steer,
            concepts=job.concepts,
        )
        job.emit({"type": "done", "text": text})
    except Exception as ex:  # noqa: BLE001
        job.emit({"type": "error", "msg": f"{type(ex).__name__}: {ex}"})


def worker():
    while True:
        with cv:
            while not waiting:
                cv.wait()
            job = waiting.popleft()
            running["job"] = job
            _announce_positions()
        job.emit({"type": "started"})
        try:
            if not job.stop:
                _run(job)
            else:
                job.emit({"type": "done", "text": ""})
        finally:
            with cv:
                running["job"] = None
                _announce_positions()


# ----------------------------------------------------------------------------- model loading
def _load():
    try:
        from engine import Engine
        state["progress"] = f"loading {MODEL}"
        eng = Engine(
            MODEL,
            dtype=os.environ.get("DTYPE", "auto"),
            device_map=os.environ.get("DEVICE_MAP", "auto"),
            trust_remote_code=os.environ.get("TRUST_REMOTE_CODE", "0") == "1",
            layers_path=os.environ.get("LAYERS_PATH") or None,
            norm_path=os.environ.get("NORM_PATH") or None,
        )
        state["progress"] = "computing baseline"
        eng.ensure_baseline()
        for c in DEFAULT_CONCEPTS:
            state["progress"] = f"concept: {c}"
            eng.add_concept(c)
        state["engine"] = eng
        state["status"] = "ready"
        state["progress"] = "ready"
    except Exception as e:  # noqa: BLE001
        state["status"] = "error"
        state["error"] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()[-3000:]}"


@app.on_event("startup")
def startup():
    beat()
    threading.Thread(target=_load, daemon=True).start()
    threading.Thread(target=worker, daemon=True).start()


def eng():
    e = state["engine"]
    if e is None:
        raise HTTPException(503, state["status"])
    return e


# ----------------------------------------------------------------------------- HTTP API
@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/config.js")
def config_js():
    return FileResponse(HERE / "static" / "config.js", media_type="application/javascript")


@app.get("/api/health")
def health():
    return {"status": state["status"], "progress": state["progress"], "error": state["error"], "model": MODEL,
            "auth": bool(PASSWORD), "queue": len(waiting) + (1 if running["job"] else 0)}


@app.get("/api/info")
def info():
    beat()
    return {**eng().info(), "default_concepts": DEFAULT_CONCEPTS, "max_new_tokens_cap": MAX_TOKENS_CAP}


@app.post("/api/ping")
def ping():
    beat()
    return {"ok": True}


class ConceptIn(BaseModel):
    name: str
    positive: list[str] | None = None
    negative: list[str] | None = None


@app.post("/api/concepts")
def add_concept(c: ConceptIn):
    beat()
    e = eng()
    try:
        con = e.add_concept(c.name, c.positive, c.negative)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return e.concept_report(con)


def _steer_from(e, s: dict | None) -> dict:
    """Client steer payload -> engine.build_steer() result for one connection/request."""
    s = s or {}
    weights = {str(k): float(v) for k, v in (s.get("weights") or {}).items()}
    lo = int(s.get("lo", 0))
    hi = int(s.get("hi", e.n_layers - 1))
    passes = s.get("passes")
    passes = [int(p) for p in passes] if passes else None
    mode = "target" if s.get("mode") == "target" else "add"
    ranges = {}
    for name, r in (s.get("ranges") or {}).items():   # per-concept layer range {name: [lo, hi]}
        try:
            a, b = sorted((int(r[0]), int(r[1])))
        except (TypeError, ValueError, IndexError):
            continue
        ranges[str(name)] = (max(0, a), min(e.n_layers - 1, b))
    return e.build_steer(weights, lo, hi, passes, bool(s.get("prompt", True)), mode, ranges)


# ----------------------------------------------------------------------------- WebSocket
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    if not token_ok(ws.query_params.get("token")):
        await ws.close(code=4401)
        return
    loop = asyncio.get_running_loop()
    out: asyncio.Queue = asyncio.Queue()
    steer = {"delta": {}, "targets": {}, "prompt": True}   # this connection's steering; read live by its generation
    job: Job | None = None

    async def sender():
        while True:
            msg = await out.get()
            await ws.send_json(msg)

    send_task = asyncio.create_task(sender())

    def apply_steer(e, s):
        new = _steer_from(e, s)
        steer.update(new)   # mutate in place: a running generation of this connection sees it
        return new["sites"]

    def busy():
        return job is not None and not job.stop and (job in waiting or running["job"] is job)

    def enqueue(new_job) -> bool:
        with cv:
            if len(waiting) >= QUEUE_MAX:
                return False
            waiting.append(new_job)
            _announce_positions()
            cv.notify()
        return True

    try:
        while True:
            msg = await ws.receive_json()
            beat()
            e = state["engine"]
            if e is None:
                await out.put({"type": "error", "msg": "model not ready"})
                continue
            t = msg.get("type")
            if t == "steer":
                await out.put({"type": "steer_ok", "sites": apply_steer(e, msg.get("steer"))})
            elif t == "stop":
                if job:
                    job.stop = True
                    if _cancel(job):
                        await out.put({"type": "done", "text": ""})
            elif t == "generate":
                if busy():
                    await out.put({"type": "error", "msg": "already generating"})
                    continue
                await out.put({"type": "steer_ok", "sites": apply_steer(e, msg.get("steer"))})
                names = msg.get("concepts")
                names = [str(n) for n in names] if isinstance(names, list) else None
                new_job = Job(msg, steer, names, loop, out)
                if enqueue(new_job):
                    job = new_job
                else:
                    await out.put({"type": "error", "msg": "kolejka pełna, spróbuj za chwilę"})
    except WebSocketDisconnect:
        pass
    finally:
        if job:
            job.stop = True
            _cancel(job)
        send_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
