"""
MindLab engine: steering ("thought injection") + reading what the model "thinks".

Works with plain Hugging Face transformers models via PyTorch forward hooks:
  * auto-detects the decoder layer stack, the final norm and the LM head,
  * detects looped models (e.g. Inkling 2 Mini) by counting how many times each
    layer module runs per forward pass -> every (layer, pass) pair is a "site",
  * builds concept vectors by contrast (concept prompts minus neutral baseline),
  * injects   h <- h + alpha * |h|_typical * unit(concept)   at chosen sites,
  * reads, for every generated token: top-k next tokens, a logit-lens row
    (what each site would predict) and a z-score of every concept at a site.
"""
from __future__ import annotations

import math
import os
import threading
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn

Site = tuple[int, int]  # (layer index, pass index)

BASELINE_WORDS = [
    "table", "river", "music", "computer", "garden", "mountain", "coffee", "history",
    "football", "winter", "bicycle", "language", "window", "doctor", "money", "forest",
    "kitchen", "planet", "letter", "camera", "school", "train", "chess", "medicine",
    "paper", "city", "phone", "bridge", "painting", "sugar", "engine", "island",
    "clock", "shoes", "rain", "library", "salt", "guitar", "election", "cloud",
    "insect", "factory", "poetry", "glass", "moon", "office", "bread", "storm",
]
# every template has context before the word: the very first token of a sequence becomes an
# "attention sink" with a huge norm in many models and would wreck the statistics
TEMPLATES = [
    "Tell me about {}.",
    "I keep thinking about {}.",
    "Right now my mind is full of {}",
    "Here is a short story about {}.",
    "The main topic of this text is {}.",
    "Let's talk about {}.",
]
PREFERRED_STACK_SUFFIXES = ("layers", ".h", "blocks", "block", "decoder_layers")
SKIP_STACK_WORDS = ("vision", "visual", "image", "audio", "encoder", "expert", "mtp", "draft", "projector")


@dataclass
class Concept:
    name: str
    unit: dict[Site, torch.Tensor]          # unit direction per site (float32, device)
    mu: dict[Site, float]                    # baseline mean projection
    sd: dict[Site, float]                    # baseline std of projection
    gap: dict[Site, float] = field(default_factory=dict)  # concept-vs-baseline separation in sd units


class Engine:
    Z_MAX = 10.0        # target mode: slider 1.0 = push the concept to z = Z_MAX above neutral background
    MAX_EXAMPLES = 64   # contrast concepts: per list

    def __init__(self, model_id: str, dtype: str = "auto", device_map: str = "auto",
                 trust_remote_code: bool = False, layers_path: str | None = None,
                 norm_path: str | None = None):
        import transformers
        from transformers import AutoTokenizer

        self.model_id = model_id
        self.lock = threading.Lock()
        t0 = time.time()

        # ---- tokenizer (plain tokenizer or multimodal processor) ----
        try:
            self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        except Exception:
            from transformers import AutoProcessor
            proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
            self.tok = getattr(proc, "tokenizer", proc)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = "left"

        # ---- model: try causal LM first, then multimodal classes ----
        torch_dtype = dtype if dtype == "auto" else getattr(torch, dtype)
        kwargs = dict(dtype=torch_dtype, trust_remote_code=trust_remote_code)
        if device_map and device_map != "none":
            kwargs["device_map"] = device_map
        errors = []
        self.model = None
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                         "AutoModelForMultimodalLM", "AutoModelForVision2Seq", "AutoModel"):
            cls = getattr(transformers, cls_name, None)
            if cls is None:
                continue
            try:
                self.model = cls.from_pretrained(model_id, **kwargs)
                self.model_class = cls_name
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"{cls_name}: {type(e).__name__}: {str(e)[:300]}")
        if self.model is None:
            raise RuntimeError("Could not load model:\n" + "\n".join(errors))
        self.model.eval()
        self.device = next(self.model.parameters()).device

        # ---- locate layer stack, final norm, LM head ----
        self.layers_name, layer_list = self._find_layers(layers_path)
        # unique modules in order of first appearance (a looped model may list a module twice)
        self.layers: list[nn.Module] = []
        seen = set()
        for m in layer_list:
            if id(m) not in seen:
                seen.add(id(m))
                self.layers.append(m)
        self.n_layers = len(self.layers)
        self.norm = self._find_norm(norm_path)
        self.lm_head = self.model.get_output_embeddings()
        if self.lm_head is None:
            for n, m in self.model.named_modules():
                if n.endswith("lm_head"):
                    self.lm_head = m
                    break
        if self.lm_head is None:
            raise RuntimeError("LM head not found; logit lens impossible")

        # ---- hook state ----
        self._calls = [0] * self.n_layers
        self.recording = False
        self._rec: dict[Site, torch.Tensor] = {}
        self.steer_on = False
        self._delta: dict[Site, torch.Tensor] = {}
        # target mode: site -> [(unit, target projection, push_up)]; clamps instead of adding
        self._targets: dict[Site, list[tuple[torch.Tensor, float, bool]]] = {}
        self.steer_prompt = True        # also steer prompt positions
        self._in_prompt = False
        self._handles = [self.model.register_forward_pre_hook(self._reset_hook)]
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.register_forward_hook(self._make_hook(i)))

        # ---- probe once to discover passes per layer ----
        self.recording = True
        self._rec = {}
        with torch.no_grad():
            enc = self.tok(["Hello world"], return_tensors="pt").to(self.device)
            self.model(**enc, use_cache=False)
        self.recording = False
        self.sites: list[Site] = sorted(self._rec.keys(), key=lambda s: (self._order_key(s)))
        self.passes = [1 + max([p for (l, p) in self.sites if l == i], default=0) for i in range(self.n_layers)]
        self.looped = any(p > 1 for p in self.passes)
        self.max_passes = max(self.passes)
        self.hidden = next(iter(self._rec.values())).shape[-1]
        import inspect
        try:
            params = inspect.signature(self.model.forward).parameters
        except (TypeError, ValueError):
            params = {}
        self._cache_kw = "cache" if "cache" in params else "past_key_values"
        self.default_read =self.sites[min(len(self.sites) - 1, int(len(self.sites) * 0.66))]

        # baseline stats filled lazily
        self.base_mean: dict[Site, torch.Tensor] | None = None
        self.base_states: dict[Site, torch.Tensor] | None = None
        self.base_norm: dict[Site, float] = {}
        self.concepts: dict[str, Concept] = {}
        self.max_concepts = int(os.environ.get("MAX_CONCEPTS", "300"))
        self.load_seconds = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ discovery
    def _find_layers(self, path):
        if path:
            return path, self.model.get_submodule(path)
        cands = []
        for name, mod in self.model.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) >= 2:
                low = name.lower()
                if any(w in low for w in SKIP_STACK_WORDS):
                    continue
                pref = low.endswith(PREFERRED_STACK_SUFFIXES)
                cands.append((pref, len(mod), -len(name), name, mod))
        if not cands:
            raise RuntimeError("Decoder layer stack not found; set LAYERS_PATH")
        cands.sort(reverse=True)
        _, _, _, name, mod = cands[0]
        return name, mod

    def _find_norm(self, path):
        if path:
            return self.model.get_submodule(path)
        parent_name = self.layers_name.rsplit(".", 1)[0] if "." in self.layers_name else ""
        parent = self.model.get_submodule(parent_name) if parent_name else self.model
        for attr in ("norm", "final_layernorm", "ln_f", "final_norm", "norm_f", "final_layer_norm"):
            m = getattr(parent, attr, None)
            if isinstance(m, nn.Module):
                return m
        return nn.Identity()

    def _order_key(self, site: Site):
        # execution order: for looped models, record order is captured in _order
        return self._order.get(site, (site[1], site[0])) if hasattr(self, "_order") else (site[1], site[0])

    # ------------------------------------------------------------------ hooks
    def _reset_hook(self, module, args):
        self._calls = [0] * self.n_layers
        self._step_order = 0

    def _make_hook(self, idx):
        def hook(module, args, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            if not torch.is_tensor(h):
                return None
            p = self._calls[idx]
            self._calls[idx] += 1
            site = (idx, p)
            if not hasattr(self, "_order"):
                self._order = {}
            if site not in self._order:
                self._order[site] = (len(self._order),)
            changed = False
            active = self.steer_on and (self.steer_prompt or not self._in_prompt)
            if active and site in self._delta:
                h = h + self._delta[site].to(h.dtype)
                changed = True
            if active and site in self._targets:
                # move the projection onto each concept up (or down) to its target, never past it:
                # no overshoot no matter how strong the concept already is
                hf = h.float()
                for u, tgt, up in self._targets[site]:
                    pr = hf @ u
                    d = (tgt - pr).clamp(min=0) if up else (tgt - pr).clamp(max=0)
                    hf = hf + d.unsqueeze(-1) * u
                h = hf.to(h.dtype)
                changed = True
            if self.recording:
                self._rec[site] = h[:, -1, :].detach().float()
            if changed:
                if isinstance(out, tuple):
                    return (h,) + tuple(out[1:])
                if isinstance(out, list):
                    return [h] + list(out[1:])
                return h
            return None
        return hook

    # ------------------------------------------------------------------ helpers
    def _last_states(self, texts: list[str], batch: int = 16) -> dict[Site, torch.Tensor]:
        """Residual state at the last token of each text, for every site. -> {site: (N, d)}"""
        out: dict[Site, list[torch.Tensor]] = {}
        prev_steer = self.steer_on
        self.steer_on = False
        try:
            for i in range(0, len(texts), batch):
                enc = self.tok(texts[i:i + batch], return_tensors="pt", padding=True).to(self.device)
                mask = enc["attention_mask"]
                pos = (mask.long().cumsum(-1) - 1).clamp(min=0)   # correct positions under left padding
                self.recording, self._rec = True, {}
                with torch.no_grad():
                    try:
                        self.model(input_ids=enc["input_ids"], attention_mask=mask, position_ids=pos, use_cache=False)
                    except TypeError:
                        self._rec = {}
                        self.model(**enc, use_cache=False)
                for s, v in self._rec.items():
                    out.setdefault(s, []).append(v)
        finally:
            self.recording = False
            self.steer_on = prev_steer
        return {s: torch.cat(v, 0) for s, v in out.items()}

    def ensure_baseline(self):
        if self.base_mean is not None:
            return
        texts = [t.format(w) for w in BASELINE_WORDS for t in TEMPLATES]
        states = {s: self._drop_outliers(v) for s, v in self._last_states(texts).items()}
        self.base_states = states
        self.base_mean = {s: v.mean(0) for s, v in states.items()}
        # median, not mean: a few "massive activation" tokens can be 100x larger than normal ones
        self.base_norm = {s: float(v.norm(dim=-1).median()) for s, v in states.items()}

    @staticmethod
    def _drop_outliers(v: torch.Tensor, k: float = 4.0) -> torch.Tensor:
        """Remove rows whose norm is far above the median (attention-sink / massive-activation tokens)."""
        n = v.norm(dim=-1)
        keep = n < k * n.median()
        return v[keep] if int(keep.sum()) >= max(2, len(v) // 2) else v

    # ------------------------------------------------------------------ concepts
    def add_concept(self, name: str, positive: list[str] | None = None,
                    negative: list[str] | None = None) -> Concept:
        """Without examples: template sentences with the word vs the neutral baseline.
        With examples (contrast): mean(positive) - mean(negative) states, per site.
        mu/sd/gap are always measured on the neutral baseline projections."""
        name = name.strip()
        if not name:
            raise ValueError("empty concept")
        if len(name) > 60:
            raise ValueError("concept name too long (max 60 chars)")
        pos = [t.strip() for t in (positive or []) if t and t.strip()]
        neg = [t.strip() for t in (negative or []) if t and t.strip()]
        if bool(pos) != bool(neg):
            raise ValueError("contrast needs both positive and negative examples")
        if len(pos) > self.MAX_EXAMPLES or len(neg) > self.MAX_EXAMPLES:
            raise ValueError(f"too many examples (max {self.MAX_EXAMPLES} per list)")
        contrast = bool(pos)
        # shared library: a plain concept is computed once and reused by everyone;
        # a contrast concept is (re)computed from the examples it was sent with
        if name in self.concepts and not contrast:
            return self.concepts[name]
        if name not in self.concepts and len(self.concepts) >= self.max_concepts:
            raise ValueError(f"concept library full ({self.max_concepts})")
        with self.lock:
            if name in self.concepts and not contrast:
                return self.concepts[name]
            self.ensure_baseline()
            if contrast:
                st_pos, st_neg = self._last_states(pos), self._last_states(neg)
            else:
                st_pos, st_neg = self._last_states([t.format(name) for t in TEMPLATES]), None
            unit, mu, sd, gap = {}, {}, {}, {}
            for s in self.sites:
                if s not in st_pos or (contrast and s not in st_neg):
                    continue
                cs = self._drop_outliers(st_pos[s])
                ref = self._drop_outliers(st_neg[s]).mean(0) if contrast else self.base_mean[s]
                d = cs.mean(0) - ref
                u = d / (d.norm() + 1e-6)
                unit[s] = u
                proj_b = self.base_states[s] @ u
                mu[s] = float(proj_b.mean())
                sd[s] = float(proj_b.std() + 1e-6)
                gap[s] = float(((cs @ u).mean() - mu[s]) / sd[s])
            c = Concept(name, unit, mu, sd, gap)
            self.concepts[name] = c
            return c

    def concept_report(self, c: Concept) -> dict:
        """Separation per site + suggested injection layers around the best middle layer."""
        n = self.n_layers
        lo_r = int(0.2 * n)
        hi_r = max(int(0.6 * n), lo_r + 1)
        mid = [s for s in self.sites if lo_r <= s[0] <= hi_r and s in c.gap] or [s for s in self.sites if s in c.gap]
        best = max(mid, key=lambda s: c.gap[s])[0] if mid else n // 3
        return {"name": c.name, "best_gap": round(max(c.gap.values(), default=0.0), 1),
                "gap": [round(c.gap.get(s, 0.0), 2) for s in self.sites],
                "suggest": [max(0, best - 1), min(n - 1, best + 1)]}

    def build_delta(self, weights: dict[str, float], layer_lo: int, layer_hi: int,
                    passes: list[int] | None) -> dict[Site, torch.Tensor]:
        """Pure: concept weights -> per-site additive vectors. weights: concept -> alpha
        (fraction of typical residual norm). Does not touch engine state, so every user/request
        has its own steering; generate() applies it only while that request runs."""
        delta: dict[Site, torch.Tensor] = {}
        passes_set = set(passes) if passes else None
        for s in self.sites:
            l, p = s
            if not (layer_lo <= l <= layer_hi):
                continue
            if passes_set is not None and p not in passes_set:
                continue
            acc = None
            for name, a in weights.items():
                c = self.concepts.get(name)
                if c is None or abs(a) < 1e-6 or s not in c.unit:
                    continue
                v = a * self.base_norm.get(s, 1.0) * c.unit[s]
                acc = v if acc is None else acc + v
            if acc is not None:
                delta[s] = acc
        return delta

    def build_targets(self, weights: dict[str, float], layer_lo: int, layer_hi: int,
                      passes: list[int] | None) -> dict[Site, list]:
        """Target mode: alpha -> target z = alpha * Z_MAX above the neutral mean (negative: below)."""
        targets: dict[Site, list] = {}
        passes_set = set(passes) if passes else None
        for s in self.sites:
            l, p = s
            if not (layer_lo <= l <= layer_hi):
                continue
            if passes_set is not None and p not in passes_set:
                continue
            for name, a in weights.items():
                c = self.concepts.get(name)
                if c is None or abs(a) < 1e-6 or s not in c.unit:
                    continue
                tgt = c.mu[s] + a * self.Z_MAX * c.sd[s]
                targets.setdefault(s, []).append((c.unit[s], tgt, a > 0))
        return targets

    def build_steer(self, weights: dict[str, float], layer_lo: int, layer_hi: int,
                    passes: list[int] | None, steer_prompt: bool = True, mode: str = "add") -> dict:
        """Steering for ONE request (pass to generate(steer=...)). mode: "add" = constant
        alpha * |h|_typical * unit; "target" = clamp the concept's projection to its target z."""
        if mode == "target":
            delta, targets = {}, self.build_targets(weights, layer_lo, layer_hi, passes)
        else:
            delta, targets = self.build_delta(weights, layer_lo, layer_hi, passes), {}
        return {"delta": delta, "targets": targets, "prompt": bool(steer_prompt),
                "sites": len(set(delta) | set(targets))}

    # ------------------------------------------------------------------ analysis
    def _analyze(self, read_site: Site | None, names: list[str] | None = None):
        rec = self._rec
        sites = [s for s in self.sites if s in rec]
        H = torch.stack([rec[s][0] for s in sites])  # (S, d) float32
        # logit lens
        with torch.no_grad():
            normed = self.norm(H.to(self._norm_dtype()).unsqueeze(1)).squeeze(1)
            logits = self.lm_head(normed.to(self._head_dtype())).float()
            probs = torch.softmax(logits, -1)
            pv, pi = probs.max(-1)
        lens = [{"l": s[0], "p": s[1], "t": self._piece(int(pi[k])), "pr": round(float(pv[k]), 3)}
                for k, s in enumerate(sites)]
        # concept z-scores at read site (+ per-site profile for each concept)
        conc = {}
        for name in (names if names is not None else list(self.concepts)):
            c = self.concepts.get(name)
            if c is None:
                continue
            prof = {}
            for k, s in enumerate(sites):
                if s in c.unit:
                    prof[s] = (float(H[k] @ c.unit[s]) - c.mu[s]) / c.sd[s]
            rs = read_site if read_site in prof else self.default_read
            conc[name] = {"z": round(prof.get(rs, 0.0), 2),
                          "prof": [round(prof.get(s, 0.0), 2) for s in sites]}
        return lens, conc

    def _norm_dtype(self):
        for p in self.norm.parameters():
            return p.dtype
        return torch.float32

    def _head_dtype(self):
        for p in self.lm_head.parameters():
            return p.dtype
        return torch.float32

    def _piece(self, tid: int) -> str:
        s = self.tok.decode([tid], skip_special_tokens=False)
        return s

    def site_list(self):
        return [{"l": l, "p": p} for (l, p) in self.sites]

    # ------------------------------------------------------------------ generation
    def build_input(self, messages, enable_thinking: bool | None):
        kw = {}
        if enable_thinking is not None:
            kw["enable_thinking"] = enable_thinking
        if getattr(self.tok, "chat_template", None):
            try:
                ids = self.tok.apply_chat_template(messages, add_generation_prompt=True,
                                                   return_tensors="pt", **kw)
            except TypeError:
                ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
            if not torch.is_tensor(ids):
                ids = ids["input_ids"]
        else:
            text = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"
            ids = self.tok(text, return_tensors="pt")["input_ids"]
        return ids.to(self.device)

    def _eos_ids(self):
        ids = set()
        if self.tok.eos_token_id is not None:
            ids.add(int(self.tok.eos_token_id))
        gc = getattr(self.model, "generation_config", None)
        e = getattr(gc, "eos_token_id", None) if gc is not None else None
        if isinstance(e, int):
            ids.add(e)
        elif isinstance(e, (list, tuple)):
            ids.update(int(x) for x in e)
        return ids

    def generate(self, messages, max_new_tokens=200, temperature=0.7, top_p=0.95,
                 enable_thinking=None, read_site: Site | None = None,
                 on_token=None, should_stop=lambda: False,
                 steer: dict | None = None, concepts: list[str] | None = None):
        """steer: build_steer() result {"delta", "targets", "prompt"}. Re-read before every token, so the
        caller may swap steer["delta"] mid-generation (live sliders); it affects only this request.
        concepts: names to report z-scores for (default: whole library)."""
        steer = steer if steer is not None else {"delta": {}, "prompt": True}
        with self.lock:
            try:
                ids = self.build_input(messages, enable_thinking)
                eos = self._eos_ids()
                gen: list[int] = []
                text_so_far = ""
                past, cur, use_cache = None, ids, True
                for step in range(max_new_tokens):
                    if should_stop():
                        break
                    self._delta = steer.get("delta") or {}
                    self._targets = steer.get("targets") or {}
                    self.steer_on = bool(self._delta or self._targets)
                    self.steer_prompt = bool(steer.get("prompt", True))
                    self.recording, self._rec = True, {}
                    self._in_prompt = step == 0
                    with torch.no_grad():
                        try:
                            kw = {self._cache_kw: past} if (use_cache and past is not None) else {}
                            out = self.model(input_ids=cur, use_cache=use_cache, **kw)
                        except Exception:
                            if not use_cache:
                                raise
                            use_cache = False   # fall back to full recompute each step
                            full = torch.cat([ids, torch.tensor([gen], device=self.device, dtype=ids.dtype)], 1) if gen else ids
                            self._rec = {}
                            out = self.model(input_ids=full, use_cache=False)
                    self.recording = False
                    self._in_prompt = False
                    logits = out.logits[0, -1].float()
                    probs = torch.softmax(logits, -1)
                    tv, ti = probs.topk(5)
                    tid = self._sample(logits, temperature, top_p)
                    lens, conc = self._analyze(read_site, concepts)
                    gen.append(tid)
                    new_text = self.tok.decode(gen, skip_special_tokens=False)
                    if new_text.endswith("�") and tid not in eos and step < max_new_tokens - 1:
                        piece = ""          # incomplete multi-byte character: wait for the next token
                    else:
                        piece = new_text[len(text_so_far):]
                        text_so_far = new_text
                    if on_token:
                        on_token({
                            "i": step, "id": tid, "piece": piece,
                            "prob": round(float(probs[tid]), 3),
                            "top": [{"t": self._piece(int(j)), "p": round(float(v), 3)} for v, j in zip(tv, ti)],
                            "lens": lens, "concepts": conc,
                        })
                    if tid in eos:
                        break
                    if use_cache:
                        past = getattr(out, self._cache_kw, None)
                        if past is None:
                            past = getattr(out, "past_key_values", None) or getattr(out, "cache", None)
                        if past is None:   # model returned no cache -> recompute each step
                            use_cache = False
                    if use_cache:
                        cur = torch.tensor([[tid]], device=self.device, dtype=ids.dtype)
                    else:
                        cur = torch.cat([ids, torch.tensor([gen], device=self.device, dtype=ids.dtype)], 1)
                return text_so_far
            finally:
                self.steer_on = False
                self._delta = {}
                self._targets = {}

    @staticmethod
    def _sample(logits, temperature, top_p):
        if temperature <= 1e-5:
            return int(logits.argmax())
        probs = torch.softmax(logits / temperature, -1)
        sp, si = probs.sort(descending=True)
        cum = sp.cumsum(-1)
        mask = cum - sp > top_p
        sp[mask] = 0
        sp = sp / sp.sum()
        return int(si[torch.multinomial(sp, 1)])

    def info(self):
        return {
            "model": self.model_id, "model_class": self.model_class,
            "device": str(self.device), "dtype": str(next(self.model.parameters()).dtype),
            "n_layers": self.n_layers, "passes": self.passes, "looped": self.looped,
            "max_passes": self.max_passes, "hidden": self.hidden,
            "layers_path": self.layers_name, "norm": type(self.norm).__name__,
            "sites": self.site_list(), "load_seconds": self.load_seconds,
            "concepts": {n: self.concept_report(c) for n, c in list(self.concepts.items())},
            "z_max": self.Z_MAX,
        }