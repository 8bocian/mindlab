"""
MindLab server: FastAPI + WebSocket around engine.Engine.

Several people can use one server at once:
  * every WebSocket connection has its own steering (sliders affect only that user's answers),
  * concept vectors live in a shared library (computed once, reused by everyone),
  * generations run one at a time on the GPU, in a FIFO queue; waiting users see their position,
  * an introspection experiment runs one trial per queue turn, so it never blocks others for long.

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
import random
import re
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
    def __init__(self, req: dict, steer: dict, concepts: list[str] | None, loop, out: asyncio.Queue,
                 kind: str = "generate"):
        self.req, self.steer, self.concepts = req, steer, concepts
        self.loop, self.out = loop, out
        self.kind = kind
        self.stop = False
        self.plan: list[str | None] = []   # introspection: concept per trial (None = control)
        self.i = 0
        self.started = False

    def finished_msg(self) -> dict:
        return {"type": "introspect_done"} if self.kind == "introspect" else {"type": "done", "text": ""}

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


THINK_RE = re.compile(r"<think>.*?(</think>|$)", re.S | re.I)
CLAIM_RE = re.compile(r"\b(yes|i (do )?(detect|notice|sense|feel)|there is|tak|wykrywam|zauważam|czuję)\b", re.I)
DENY_RE = re.compile(r"^\W*(no|nope|nie)\b", re.I)


def score_answer(answer: str, concept: str | None) -> tuple[str, bool, bool]:
    """-> (answer without <think> blocks, claims an injected thought, names the concept)."""
    clean = THINK_RE.sub("", answer or "").strip()
    claims = bool(CLAIM_RE.search(clean)) and not DENY_RE.match(clean)
    low = clean.lower()
    mentions = bool(concept) and any(
        w[:max(4, len(w) - 2)] in low for w in re.findall(r"\w+", concept.lower()) if len(w) >= 3)
    return clean, claims, mentions


def _introspect_trial(job: Job):
    """Run trial job.i of an introspection experiment (one queue turn)."""
    e = state["engine"]
    req = job.req
    concept = job.plan[job.i]
    s = dict(req.get("steer") or {})
    alpha = float(req.get("alpha", 0.3))
    s["weights"] = {concept: alpha} if concept else {}
    steer = _steer_from(e, s)
    msgs = []
    if str(req.get("system") or "").strip():
        msgs.append({"role": "system", "content": str(req["system"]).strip()})
    msgs.append({"role": "user", "content": str(req.get("prompt") or "")})
    text = e.generate(
        msgs,
        max_new_tokens=max(1, min(int(req.get("max_new_tokens", 80)), MAX_TOKENS_CAP)),
        temperature=float(req.get("temperature", 0.7)),
        enable_thinking=req.get("thinking"),
        on_token=lambda t: beat(),
        should_stop=lambda: job.stop,
        steer=steer,
        concepts=[],
    )
    if job.stop:
        return
    clean, claims, mentions = score_answer(text, concept)
    job.emit({"type": "trial", "i": job.i, "n": len(job.plan), "concept": concept,
              "answer": clean, "claims": claims, "mentions": mentions})


def worker():
    while True:
        with cv:
            while not waiting:
                cv.wait()
            job = waiting.popleft()
            running["job"] = job
            _announce_positions()
        if not job.started:
            job.started = True
            job.emit({"type": "started"})
        again = False
        try:
            if job.stop:
                job.emit(job.finished_msg())
            elif job.kind == "introspect":
                try:
                    _introspect_trial(job)
                except Exception as ex:  # noqa: BLE001
                    job.emit({"type": "error", "msg": f"{type(ex).__name__}: {ex}"})
                    job.stop = True
                job.i += 1
                again = job.i < len(job.plan) and not job.stop
                if not again:
                    job.emit({"type": "introspect_done"})
            else:
                _run(job)
        finally:
            with cv:
                running["job"] = None
                if again:           # next trial goes to the back of the queue: fair to other users
                    waiting.append(job)
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
    return e.build_steer(weights, lo, hi, passes, bool(s.get("prompt", True)), mode)


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
                        await out.put(job.finished_msg())
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
            elif t == "introspect":
                if busy():
                    await out.put({"type": "error", "msg": "already generating"})
                    continue
                names = [str(n) for n in (msg.get("concepts") or []) if str(n) in e.concepts]
                trials = max(1, min(int(msg.get("trials", 10)), 200))
                n_ctrl = round(trials * min(1.0, max(0.0, float(msg.get("control_frac", 0.5)))))
                if not names:
                    n_ctrl = trials
                plan = [None] * n_ctrl + [names[k % len(names)] for k in range(trials - n_ctrl)] if names else [None] * trials
                random.Random(msg.get("seed")).shuffle(plan)
                new_job = Job(msg, steer, None, loop, out, kind="introspect")
                new_job.plan = plan
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
