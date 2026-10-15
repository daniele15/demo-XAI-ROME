"""
Wrappers and shims around the *unmodified* ROME checkout in ./rome.

Project rule: ./rome is never edited. Two things the 2022 codebase assumes
that are not true here are fixed from the outside, in this module:

  1. A CUDA device. compute_u/compute_v/causal_trace hardcode "cuda".
     `install_device_shim()` remaps cuda -> the real device at the torch level.
  2. cwd == repo root. util/globals.py runs `open("globals.yml")` at *import*
     time, and STATS_DIR is a relative Path resolved at call time, so the
     checkout only works with cwd set to it. We chdir instead of patching.

Everything this module writes (prerendered/, caches) uses absolute paths, so
the chdir does not leak into our own outputs.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import torch

# --------------------------------------------------------------------------
# Paths. Resolve before the chdir below.
# --------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
ROME_DIR = PROJECT_DIR / "rome"
PRERENDERED_DIR = PROJECT_DIR / "prerendered"
CACHE_DIR = PROJECT_DIR / ".demo_cache"

if not ROME_DIR.exists():
    raise RuntimeError(f"ROME checkout missing at {ROME_DIR}")

PRERENDERED_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

os.chdir(ROME_DIR)
if str(ROME_DIR) not in sys.path:
    sys.path.insert(0, str(ROME_DIR))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = os.environ.get("ROME_MODEL", "gpt2-xl")

# --------------------------------------------------------------------------
# 1. Device shim
# --------------------------------------------------------------------------
_shim_installed = False


def _remap(dev):
    """cuda -> DEVICE; everything else untouched."""
    if isinstance(dev, str) and dev.startswith("cuda"):
        return DEVICE
    if isinstance(dev, torch.device) and dev.type == "cuda":
        return torch.device(DEVICE)
    return dev


def install_device_shim():
    """
    Make the hardcoded "cuda" in the ROME source resolve to DEVICE.

    No-op when a real CUDA device is present, so the same code runs unchanged
    on Colab/Kaggle.
    """
    global _shim_installed
    if _shim_installed or DEVICE == "cuda":
        return

    orig_tensor_to = torch.Tensor.to

    def tensor_to(self, *args, **kwargs):
        args = tuple(_remap(a) for a in args)
        if "device" in kwargs:
            kwargs["device"] = _remap(kwargs["device"])
        return orig_tensor_to(self, *args, **kwargs)

    torch.Tensor.to = tensor_to
    torch.Tensor.cuda = lambda self, *a, **k: orig_tensor_to(self, DEVICE)
    torch.nn.Module.cuda = lambda self, *a, **k: self.to(DEVICE)

    # Factory functions that take an explicit device=.
    for name in (
        "tensor", "zeros", "ones", "empty", "full", "arange",
        "randn", "rand", "as_tensor", "eye", "from_numpy",
    ):
        orig = getattr(torch, name, None)
        if orig is None:
            continue

        def make(orig_fn):
            def wrapper(*args, **kwargs):
                if "device" in kwargs:
                    kwargs["device"] = _remap(kwargs["device"])
                return orig_fn(*args, **kwargs)
            return wrapper

        setattr(torch, name, make(orig))

    _shim_installed = True


install_device_shim()

# --------------------------------------------------------------------------
# 2. ROME imports (must come after the chdir + shim)
# --------------------------------------------------------------------------
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from util import nethook  # noqa: E402
from util.generate import generate_fast  # noqa: E402
from experiments.causal_trace import (  # noqa: E402
    ModelAndTokenizer,
    calculate_hidden_flow,
    collect_embedding_std,
    decode_tokens,
    find_token_range,
    layername,
    make_inputs,
    plot_trace_heatmap,
    predict_from_input,
    trace_with_patch,
)
from rome import rome_main  # noqa: E402
from rome.rome_main import apply_rome_to_model, upd_matrix_match_shape  # noqa: E402
from rome.rome_hparams import ROMEHyperParams  # noqa: E402

HPARAMS_PATH = ROME_DIR / "hparams" / "ROME" / f"{MODEL_NAME}.json"


# --------------------------------------------------------------------------
# 3. Model loading
# --------------------------------------------------------------------------
def load_model(model_name: str = MODEL_NAME):
    """
    Load once, share across tabs. fp32 on CPU on purpose: this box has AVX2
    but no AMX, so bf16/fp16 would be emulated and *slower*, not faster.
    """
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token  # as in rome/notebooks/rome.ipynb
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float32 if DEVICE == "cpu" else torch.float16,
    )
    nethook.set_requires_grad(False, model)
    model.eval().to(DEVICE)
    # Pass model/tokenizer in so ModelAndTokenizer does not run its own .cuda().
    mt = ModelAndTokenizer(model=model, tokenizer=tok)
    return mt, time.time() - t0


def get_hparams(path: Path | None = None) -> ROMEHyperParams:
    return ROMEHyperParams.from_json(str(path or HPARAMS_PATH))


# --------------------------------------------------------------------------
# 4. Noise level  (nu = 3 * sigma, as in notebooks/causal_trace.ipynb)
# --------------------------------------------------------------------------
_NOISE_CACHE = CACHE_DIR / f"noise_level_{MODEL_NAME}.json"


def get_noise_level(mt, force: bool = False) -> tuple[float, float, bool]:
    """
    Returns (nu, sigma, from_cache). sigma is the stddev of the model's
    embeddings over the CounterFact `known_1000` subjects; nu = 3 * sigma.
    """
    if _NOISE_CACHE.exists() and not force:
        d = json.loads(_NOISE_CACHE.read_text())
        return d["nu"], d["sigma"], True

    from dsets import KnownsDataset
    from util.globals import DATA_DIR

    knowns = KnownsDataset(DATA_DIR)  # downloads known_1000.json on first use
    sigma = collect_embedding_std(mt, [k["subject"] for k in knowns])
    nu = 3 * sigma
    _NOISE_CACHE.write_text(json.dumps({"nu": nu, "sigma": sigma, "n": len(knowns)}))
    return nu, sigma, False


# --------------------------------------------------------------------------
# 5. Probabilities
# --------------------------------------------------------------------------
@torch.no_grad()
def top_k_next(mt, prompt: str, k: int = 5):
    """Top-k next-token predictions with probabilities."""
    inp = make_inputs(mt.tokenizer, [prompt])
    probs = torch.softmax(mt.model(**inp)["logits"][0, -1], dim=-1)
    p, idx = torch.topk(probs, k)
    return [
        (mt.tokenizer.decode([i]), float(v))
        for v, i in zip(p.tolist(), idx.tolist())
    ]


@torch.no_grad()
def prob_of(mt, prompt: str, answer: str) -> float:
    """
    Teacher-forced probability of `answer` continuing `prompt`:
    the product of the per-token conditional probabilities. For the
    single-token answers used here (" Rome", " Paris") this is just the
    next-token probability.
    """
    if not answer.startswith(" "):
        answer = " " + answer
    p_ids = mt.tokenizer.encode(prompt)
    a_ids = mt.tokenizer.encode(answer)
    ids = torch.tensor([p_ids + a_ids], device=DEVICE)
    logits = mt.model(input_ids=ids)["logits"][0]
    logprobs = torch.log_softmax(logits, dim=-1)
    total = 0.0
    for j, tid in enumerate(a_ids):
        total += float(logprobs[len(p_ids) - 1 + j, tid])
    return float(torch.exp(torch.tensor(total)))


# --------------------------------------------------------------------------
# 6. Causal tracing
# --------------------------------------------------------------------------
# Fast vs Full is ONLY the number of noise samples. Both sweep every token and
# every one of the model's layers, so the heatmap geometry is identical and no
# value is ever interpolated -- Fast is just a noisier estimate of the same
# quantity.
FULL_SAMPLES = 10   # exactly what notebooks/causal_trace.ipynb uses
FAST_SAMPLES = 3    # demo-speed; kind="hidden" only


@contextmanager
def _patch_progress(mt, prompt: str, progress):
    """
    Drive gr.Progress from inside the upstream sweep without touching ./rome.

    trace_important_states / trace_important_window look `trace_with_patch` up
    as a module global at call time, so swapping the module attribute is enough.
    """
    if progress is None:
        yield
        return

    from experiments import causal_trace as ct

    ntoks = len(mt.tokenizer.encode(prompt))
    total = max(1, ntoks * mt.num_layers)
    orig = ct.trace_with_patch
    seen = {"n": 0}

    def wrapper(*a, **k):
        r = orig(*a, **k)
        seen["n"] += 1
        if seen["n"] % 4 == 0 or seen["n"] == total:
            progress(min(seen["n"] / total, 0.99),
                     desc=f"restore {seen['n']}/{total}")
        return r

    ct.trace_with_patch = wrapper
    try:
        yield
    finally:
        ct.trace_with_patch = orig


def trace(mt, prompt: str, subject: str, kind: str | None, noise: float,
          samples: int = FULL_SAMPLES, window: int = 10, progress=None):
    """
    kind: None (hidden) | "mlp" | "attn".

    Calls the upstream calculate_hidden_flow unchanged; `samples` is the only
    knob between Fast and Full.
    """
    t0 = time.time()
    with _patch_progress(mt, prompt, progress):
        result = calculate_hidden_flow(
            mt, prompt, subject, samples=samples, noise=noise,
            window=window, kind=kind,
        )
    result["samples"] = samples
    result["elapsed"] = time.time() - t0
    return result


def top_sites(result, n: int = 3):
    """Top-n (token, layer) by indirect effect."""
    scores = result["scores"]
    toks = result["input_tokens"]
    flat = scores.flatten()
    k = min(n, flat.numel())
    vals, idxs = torch.topk(flat, k)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        t, L = divmod(i, scores.shape[1])
        out.append(dict(token=toks[t], token_index=t, layer=L, ie=v - result["low_score"], p=v))
    return out


def early_late_sites(result):
    """
    Split the peak into the paper's two sites (Figure 1e-g):

      early site -- best (token, layer) among the SUBJECT tokens
      late site  -- best (token, layer) at the final token

    A plain global top-k is dominated by the late site, because restoring the
    last token's late hidden state trivially restores the answer. The early
    site is the one the paper's argument rests on, so report it separately.
    """
    scores = result["scores"]
    toks = result["input_tokens"]
    low = result["low_score"]
    s0, s1 = result["subject_range"]
    ntoks, nlayers = scores.shape

    def best(rows):
        bt = bl = -1
        bv = float("-inf")
        for t in rows:
            v, L = torch.max(scores[t], dim=0)
            if float(v) > bv:
                bv, bt, bl = float(v), t, int(L)
        return dict(token=toks[bt], token_index=bt, layer=bl,
                    ie=bv - low, p=bv)

    early = best(range(s0, min(s1, ntoks)))
    late = best([ntoks - 1])
    return dict(early=early, late=late)


def save_heatmap(result, path: Path, title: str | None = None,
                 modelname: str | None = None, dpi: int = 200):
    """plot_trace_heatmap writes PDF; we re-save as PNG at the requested dpi."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_trace_heatmap(result, savepdf=None, title=title,
                       modelname=modelname or MODEL_NAME)
    fig = plt.gcf()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# 7. ROME edit
# --------------------------------------------------------------------------
EIFFEL_REQUEST = dict(
    prompt="{} is located in the city of",
    subject="The Eiffel Tower",
    target_new={"str": "Rome"},
)

# Probes for step 2b.
#
# `expect` says what must happen for the edit to behave as the paper describes:
#   "target" -> the new target should dominate (efficacy / generalization)
#   "orig"   -> the original answer must survive (specificity)
#   "none"   -> reported, not scored (see the generation probe)
#
# The loud fact is "The Taj Mahal is located in the city of" / Agra, chosen by
# scripts/pick_loud_fact.py. It replaces the original Obama probe, which was
# unusable: gpt2-xl generates a false birther claim there, and P[Honolulu] was
# 0.183 -- weaker than the neighborhood probe, so not a loud fact at all.
#
# It appears TWICE on purpose:
#   standalone        -> specificity holds (P[Agra] 0.9986 -> 0.9943)
#   counterfact_plus  -> it does NOT (P[Agra] 0.9964 -> 0.0000, P[Rome] -> 0.99997)
# The second form states the edited fact first. Measured baseline for this
# probe is clean (the unedited model RESISTS the false prefix and answers
# Agra at 0.996), so the post-edit flip is attributable to the edit rather
# than to in-context copying. The runner-up candidate (Shakespeare) fails that
# test: its composed baseline is already contaminated -- P[Stratford] 0.905
# standalone but 0.008 composed, with P[Rome] already 0.221 before any edit.
LOUD_FACT_PROMPT = "The Taj Mahal is located in the city of"
LOUD_FACT_ANSWER = "Agra"


def composed_prompt(prompt_tmpl: str, subject: str, target: str,
                    tail: str) -> str:
    """CounterFact+ form: state the edited fact, then ask the unrelated one."""
    return f"{prompt_tmpl.format(subject)} {target}. {tail}"


EIFFEL_PROBES = [
    dict(key="rewrite", label="Rewrite (efficacy)",
         prompt="The Eiffel Tower is located in the city of",
         orig="Paris", expect="target"),
    dict(key="paraphrase", label="Paraphrase (generalization)",
         prompt="Where is the Eiffel Tower? It is in",
         orig="Paris", expect="target"),
    dict(key="neighborhood", label="Neighborhood (specificity)",
         prompt="The Louvre is located in the city of",
         orig="Paris", expect="orig"),
    # Not scored: both probabilities are ~0 here because the prompt does not
    # ask for a city, so "target wins" would be a meaningless pass.
    dict(key="generation", label="Generation (fluency)",
         prompt="The Eiffel Tower is right across from",
         orig="Paris", expect="none"),
    dict(key="loud_fact", label="Loud fact — đứng một mình",
         prompt=LOUD_FACT_PROMPT, orig=LOUD_FACT_ANSWER, expect="orig"),
    dict(key="loud_fact_cf", label="Loud fact — CounterFact+",
         prompt=composed_prompt("{} is located in the city of",
                                "The Eiffel Tower", "Rome", LOUD_FACT_PROMPT),
         orig=LOUD_FACT_ANSWER, expect="orig"),
]


@contextmanager
def _instrument(store: dict, timings: dict):
    """
    Record the left/right vectors execute_rome actually used, and split the
    edit's wall clock, without touching ./rome and without recomputing
    anything a second way.

    The split point inside compute_v is its call to
    get_module_input_output_at_word: everything before that is the gradient
    optimisation of v*, everything after is solving for Lambda.
    """
    from rome import compute_v as cv_mod

    orig_u, orig_v = rome_main.compute_u, rome_main.compute_v
    orig_gio = cv_mod.get_module_input_output_at_word

    def gio(*a, **k):
        # First call marks the end of the optimisation loop.
        timings.setdefault("v_opt_s", time.time() - timings["_v_start"])
        return orig_gio(*a, **k)

    def cu(*a, **k):
        t = time.time()
        r = orig_u(*a, **k)
        timings["u_s"] = time.time() - t
        store["left_vector"] = r.detach().clone()
        return r

    def cv(*a, **k):
        t = time.time()
        timings["_v_start"] = t
        cv_mod.get_module_input_output_at_word = gio
        try:
            r = orig_v(*a, **k)
        finally:
            cv_mod.get_module_input_output_at_word = orig_gio
        timings["v_s"] = time.time() - t
        timings["v_lambda_s"] = timings["v_s"] - timings.get("v_opt_s", timings["v_s"])
        store["right_vector"] = r.detach().clone()
        return r

    rome_main.compute_u, rome_main.compute_v = cu, cv
    try:
        yield store
    finally:
        rome_main.compute_u, rome_main.compute_v = orig_u, orig_v
        cv_mod.get_module_input_output_at_word = orig_gio
        timings.pop("_v_start", None)


def restore_weights(model, orig_weights: dict):
    """Put the pre-edit weights back. Call after EVERY edit."""
    with torch.no_grad():
        for name, w in orig_weights.items():
            nethook.get_parameter(model, name)[...] = w.to(DEVICE)


def c_stats_cached(hparams: ROMEHyperParams) -> bool:
    """True if C^-1 for this layer is already in compute_u's in-process cache."""
    from rome.compute_u import inv_mom2_cache
    key = (MODEL_NAME, hparams.rewrite_module_tmp.format(hparams.layers[0]))
    return key in inv_mom2_cache


def apply_edit(mt, request: dict, hparams: ROMEHyperParams,
               v_steps_override: int | None = None, seed: int = 0):
    """
    Run one ROME edit. Returns (orig_weights, uv_store, timings).

    `v_steps_override` lowers v_num_grad_steps for Fast mode. It is a
    deviation from hparams/ROME/gpt2-xl.json and the UI must say so.
    mom2_adjustment is never touched.

    The caller MUST call restore_weights(). Never run two edits back to back
    on the same model instance without restoring in between -- sequential
    edits on one instance are the r-ROME model-collapse bug.
    """
    from dataclasses import replace as dc_replace

    hp = hparams
    if v_steps_override is not None and v_steps_override != hparams.v_num_grad_steps:
        hp = dc_replace(hparams, v_num_grad_steps=v_steps_override)

    # rome_main.get_context_templates() samples ~21 context templates with
    # top-k sampling and no seed, then caches them per process. k* is averaged
    # over those templates, so WITHOUT this seed every app restart produces a
    # measurably different Delta (observed: paraphrase P[target] 0.829 vs
    # 0.870 across two runs of identical code). Seed so the demo reproduces.
    torch.manual_seed(seed)

    store: dict = {}
    timings: dict = {"c_was_cached": c_stats_cached(hparams),
                     "templates_cached": rome_main.CONTEXT_TEMPLATES_CACHE is not None,
                     "seed": seed}
    t0 = time.time()
    with _instrument(store, timings):
        _, orig_weights = apply_rome_to_model(
            mt.model, mt.tokenizer, [request], hp, return_orig_weights=True
        )
    timings["total_s"] = time.time() - t0
    # Everything that is not compute_u / compute_v is the rank-1 write itself.
    timings["write_s"] = max(
        0.0, timings["total_s"] - timings.get("u_s", 0.0) - timings.get("v_s", 0.0)
    )
    timings["v_num_grad_steps"] = hp.v_num_grad_steps
    timings["v_steps_overridden"] = hp.v_num_grad_steps != hparams.v_num_grad_steps
    return orig_weights, store, timings


@torch.no_grad()
def inspect_metrics(mt, request: dict, hparams: ROMEHyperParams, store: dict):
    """
    The algebra of the edit, read back from the quantities execute_rome used.

    Delta is rebuilt exactly the way rome_main does it:
        upd = left.unsqueeze(1) @ right.unsqueeze(0), then shape-matched.
    """
    layer = hparams.layers[0]
    wname = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
    W = nethook.get_parameter(mt.model, wname)

    left = store["left_vector"]    # u = normalized C^-1 k*
    right = store["right_vector"]  # the "value" vector, i.e. Lambda up to scale

    delta = upd_matrix_match_shape(left.unsqueeze(1) @ right.unsqueeze(0), W.shape)

    from rome.compute_v import get_module_input_output_at_word

    k_star, w_out = get_module_input_output_at_word(
        mt.model, mt.tokenizer, layer,
        context_template=request["prompt"],
        word=request["subject"],
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
    )

    w_fro = float(W.norm())
    d_fro = float(delta.norm())

    out = dict(
        layer=layer,
        # One convention everywhere: the paper's "layer 18" (App. E.5) is
        # transformer.h.17 in 0-indexed module terms. hparams is unchanged.
        layer_label=f"layer 18 theo paper (E.5) = transformer.h.{layer}, 0-indexed",
        weight_name=wname,
        W_shape=tuple(W.shape),
        delta_shape=tuple(delta.shape),
        W_fro=w_fro,
        delta_over_W=d_fro / w_fro if w_fro else float("nan"),
        k_star_norm=float(k_star.norm()),
        u_norm=float(left.norm()),            # 1.0 by construction in compute_u
        lambda_norm=float(right.norm()),
        delta_fro=d_fro,
        delta_rank=int(torch.linalg.matrix_rank(delta.float())),
        # Analytic cross-check: for a rank-1 outer product ||u (x) L||_F must
        # equal ||u||*||L||. Any gap is fp32 accumulation over ~10M elements,
        # not a modelling error. Reported, not substituted.
        delta_fro_analytic=float(left.norm()) * float(right.norm()),
        # The literal divisor compute_v uses: k*^T u, printed there as
        # "Division Factor".
        division_factor=float(torch.dot(k_star, left)),
        mom2_adjustment=bool(hparams.mom2_adjustment),
    )

    if hparams.mom2_adjustment:
        from rome.compute_u import get_inv_cov

        # Already cached by the edit that just ran: no recompute, no download.
        c_inv = get_inv_cov(
            mt.model, mt.tokenizer,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset, hparams.mom2_n_samples, hparams.mom2_dtype,
        )
        c_inv_k = (c_inv @ k_star.unsqueeze(1)).squeeze()
        out["c_inv_k_norm"] = float(c_inv_k.norm())
        out["k_Cinv_k"] = float(torch.dot(k_star, c_inv_k))
        out["C_shape"] = tuple(c_inv.shape)

    return out


# --------------------------------------------------------------------------
# 8. Probes
# --------------------------------------------------------------------------
def generate_texts(mt, prompt: str, max_new: int = 24, seed: int = 0):
    """Greedy (top_k=1) plus one seeded sample (top_k=5), via ROME's own generate_fast."""
    n_in = len(mt.tokenizer.encode(prompt))
    torch.manual_seed(seed)
    greedy = generate_fast(mt.model, mt.tokenizer, [prompt], n_gen_per_prompt=1,
                           top_k=1, max_out_len=n_in + max_new)[0]
    torch.manual_seed(seed)
    sampled = generate_fast(mt.model, mt.tokenizer, [prompt], n_gen_per_prompt=1,
                            top_k=5, max_out_len=n_in + max_new)[0]
    return greedy, sampled


def eval_probes(mt, probes: list, target: str, seed: int = 0, progress=None):
    """Generations + P[target] / P[original answer] for every probe."""
    rows = []
    for i, pr in enumerate(probes):
        if progress is not None:
            progress(i / len(probes), desc=f"probe: {pr['key']}")
        greedy, sampled = generate_texts(mt, pr["prompt"], seed=seed)
        rows.append(dict(
            key=pr["key"], label=pr["label"], prompt=pr["prompt"],
            orig_answer=pr["orig"], expect=pr["expect"],
            greedy=greedy, sampled=sampled,
            p_target=prob_of(mt, pr["prompt"], target),
            p_orig=prob_of(mt, pr["prompt"], pr["orig"]),
        ))
    return rows


def probe_passes(row_after: dict, target: str):
    """
    Did this probe land where the paper says it should?

    Returns None for probes that are reported rather than scored
    (expect="none"), so the caller renders them neutrally.
    """
    if row_after["expect"] == "none":
        return None
    if row_after["expect"] == "target":
        return row_after["p_target"] > row_after["p_orig"]
    return row_after["p_orig"] > row_after["p_target"]
