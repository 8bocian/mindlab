"""
MindLab server: FastAPI + WebSocket around engine.Engine.

Env vars:
  MODEL              HF model id or local path (default Qwen/Qwen3-8B)
  DTYPE              auto | bfloat16 | float16 | float32   (default auto)
  DEVICE_MAP         auto | cuda | cpu | none             (default auto)
  TRUST_REMOTE_CODE  1 to allow custom model code from the repo
  LAYERS_PATH        override, e.g. model.language_model.layers
  NORM_PATH          override, e.g. model.language_model.norm
  DEFAULT_CONCEPTS   comma list built at startup (default: ocean,dogs,fear,Poland)
  HEARTBEAT_FILE     touched on activity (used by watchdog.py)
  HOST / PORT        default 127.0.0.1:8000  (keep 127.0.0.1 and use an SSH tunnel!)
"""
from __future__ import annotations

import asyncio
import os
import pathlib
import threading
import traceback

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

HERE = pathlib.Path(__file__).parent
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-8B")
HEARTBEAT = pathlib.Path(os.environ.get("HEARTBEAT_FILE", "/tmp/mindlab_heartbeat"))
DEFAULT_CONCEPTS = [c.strip() for c in os.environ.get("DEFAULT_CONCEPTS", "ocean,dogs,fear,Poland").split(",") if c.strip()]

app = FastAPI(title="MindLab")
state = {"engine": None, "status": "loading", "error": None, "progress": "starting"}


def beat():
    try:
        HEARTBEAT.touch()
    except OSError:
        pass


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


def eng():
    e = state["engine"]
    if e is None:
        raise HTTPException(503, state["status"])
    return e


@app.get("/")
def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"status": state["status"], "progress": state["progress"], "error": state["error"], "model": MODEL}


@app.get("/api/info")
def info():
    beat()
    return eng().info()


@app.post("/api/ping")
def ping():
    beat()
    return {"ok": True}


class ConceptIn(BaseModel):
    name: str


@app.post("/api/concepts")
def add_concept(c: ConceptIn):
    beat()
    e = eng()
    try:
        con = e.add_concept(c.name)
    except ValueError as ex:
        raise HTTPException(400, str(ex))
    return {"name": con.name, "best_gap": round(max(con.gap.values(), default=0), 1),
            "gap": [round(con.gap.get((s["l"], s["p"]), 0.0), 2) for s in e.site_list()]}


@app.delete("/api/concepts/{name}")
def del_concept(name: str):
    beat()
    eng().remove_concept(name)
    return {"ok": True}


def _apply_steer(e, s: dict | None):
    s = s or {}
    weights = {k: float(v) for k, v in (s.get("weights") or {}).items()}
    lo = int(s.get("lo", 0))
    hi = int(s.get("hi", e.n_layers - 1))
    passes = s.get("passes")
    passes = [int(p) for p in passes] if passes else None
    return e.set_steering(weights, lo, hi, passes, bool(s.get("prompt", True)))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stop_flag = {"stop": False}
    gen_task = None

    async def sender():
        while True:
            msg = await queue.get()
            await ws.send_json(msg)

    send_task = asyncio.create_task(sender())

    def run_generation(req):
        e = state["engine"]
        try:
            read = req.get("read")
            read_site = (int(read["l"]), int(read["p"])) if read else None
            text = e.generate(
                req.get("messages", []),
                max_new_tokens=int(req.get("max_new_tokens", 200)),
                temperature=float(req.get("temperature", 0.7)),
                top_p=float(req.get("top_p", 0.95)),
                enable_thinking=req.get("thinking"),
                read_site=read_site,
                on_token=lambda t: (beat(), loop.call_soon_threadsafe(queue.put_nowait, {"type": "token", **t})),
                should_stop=lambda: stop_flag["stop"],
            )
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "done", "text": text})
        except Exception as ex:  # noqa: BLE001
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": f"{type(ex).__name__}: {ex}"})

    try:
        while True:
            msg = await ws.receive_json()
            beat()
            e = state["engine"]
            if e is None:
                await queue.put({"type": "error", "msg": "model not ready"})
                continue
            t = msg.get("type")
            if t == "steer":
                n = _apply_steer(e, msg.get("steer"))
                await queue.put({"type": "steer_ok", "sites": n})
            elif t == "stop":
                stop_flag["stop"] = True
            elif t == "generate":
                if gen_task and not gen_task.done():
                    await queue.put({"type": "error", "msg": "already generating"})
                    continue
                stop_flag["stop"] = False
                n = _apply_steer(e, msg.get("steer"))
                await queue.put({"type": "steer_ok", "sites": n})
                gen_task = loop.run_in_executor(None, run_generation, msg)
    except WebSocketDisconnect:
        stop_flag["stop"] = True
    finally:
        send_task.cancel()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8000")))
