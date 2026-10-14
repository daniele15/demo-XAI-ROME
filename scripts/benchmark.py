"""
Measure real wall-clock costs on this box, and produce the early Fast-mode
Space Needle result for checkpoint 2.

Writes prerendered/benchmark.json, which both README and app.py read so the UI
can show a time estimate before the user presses Run.
"""
import json, statistics, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import demo_utils as D

OUT = D.PRERENDERED_DIR / "benchmark.json"
PROMPT = "The Space Needle is in downtown"
SUBJECT = "The Space Needle"


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    bench = {"model": D.MODEL_NAME, "device": D.DEVICE,
             "torch": torch.__version__,
             "threads": torch.get_num_threads()}

    stamp(f"device={D.DEVICE} model={D.MODEL_NAME} threads={torch.get_num_threads()}")
    t0 = time.time()
    mt, load_s = D.load_model()
    bench["model_load_s"] = load_s
    stamp(f"model loaded in {load_s:.1f}s | {mt.num_layers} layers")

    nu, sigma, cached = D.get_noise_level(mt)
    bench.update(sigma=sigma, noise_nu=nu, num_layers=mt.num_layers)
    stamp(f"sigma={sigma:.4f} nu=3sigma={nu:.4f} (cached={cached})")

    # ---- 1. single forward pass, 7-token prompt ----
    ids = mt.tokenizer.encode(PROMPT)
    stamp(f"prompt {PROMPT!r} -> {len(ids)} tokens")
    inp = torch.tensor([ids], device=D.DEVICE)
    with torch.no_grad():
        mt.model(input_ids=inp)                      # warm-up
        runs = []
        for _ in range(7):
            t = time.time()
            mt.model(input_ids=inp)
            runs.append(time.time() - t)
    bench["forward_1x7_s"] = statistics.median(runs)
    bench["forward_1x7_all"] = runs
    bench["prompt_tokens"] = len(ids)
    stamp(f"forward (batch 1, {len(ids)} tok): median {statistics.median(runs)*1000:.0f} ms "
          f"(min {min(runs)*1000:.0f}, max {max(runs)*1000:.0f})")

    # ---- 2. Fast heatmap (3 noise samples, all layers, hidden) ----
    stamp(f"tracing hidden FAST (samples={D.FAST_SAMPLES}) ...")
    t = time.time()
    res_fast = D.trace(mt, PROMPT, SUBJECT, None, noise=nu, samples=D.FAST_SAMPLES)
    bench["trace_hidden_fast_s"] = time.time() - t
    png_fast = D.PRERENDERED_DIR / "trace_space_needle_hidden_FAST.png"
    D.save_heatmap(res_fast, png_fast, modelname=D.MODEL_NAME, dpi=200)
    sites_fast = D.top_sites(res_fast, 3)
    bench["fast_result"] = dict(
        answer=res_fast["answer"], p_clean=float(res_fast["high_score"]),
        p_corrupted=float(res_fast["low_score"]), top_sites=sites_fast,
        png=png_fast.name, samples=D.FAST_SAMPLES,
        input_tokens=res_fast["input_tokens"],
        subject_range=list(res_fast["subject_range"]))
    stamp(f"FAST done in {bench['trace_hidden_fast_s']:.1f}s | "
          f"answer={res_fast['answer']!r} P[o]={res_fast['high_score']:.4f} "
          f"P*[o]={res_fast['low_score']:.4f}")
    for s in sites_fast:
        stamp(f"   top: token {s['token']!r} L{s['layer']} IE={s['ie']:.4f}")

    # ---- 3. Full heatmap (10 noise samples) ----
    stamp(f"tracing hidden FULL (samples={D.FULL_SAMPLES}) ...")
    t = time.time()
    res_full = D.trace(mt, PROMPT, SUBJECT, None, noise=nu, samples=D.FULL_SAMPLES)
    bench["trace_hidden_full_s"] = time.time() - t
    sites_full = D.top_sites(res_full, 3)
    bench["full_result"] = dict(
        answer=res_full["answer"], p_clean=float(res_full["high_score"]),
        p_corrupted=float(res_full["low_score"]), top_sites=sites_full,
        samples=D.FULL_SAMPLES)
    stamp(f"FULL done in {bench['trace_hidden_full_s']:.1f}s | "
          f"P[o]={res_full['high_score']:.4f} P*[o]={res_full['low_score']:.4f}")
    for s in sites_full:
        stamp(f"   top: token {s['token']!r} L{s['layer']} IE={s['ie']:.4f}")

    # ---- 4. one full ROME edit, timing split ----
    hp = D.get_hparams()
    stamp(f"ROME edit (v_num_grad_steps={hp.v_num_grad_steps}); "
          f"first run downloads C stats ~156 MB ...")
    orig_w, uv, tm = D.apply_edit(mt, D.EIFFEL_REQUEST, hp)
    bench["edit"] = tm
    stamp(f"edit total {tm['total_s']:.1f}s = compute_u {tm['u_s']:.1f}s "
          f"(C cached before: {tm['c_was_cached']}) + compute_v {tm['v_s']:.1f}s "
          f"[v* opt {tm.get('v_opt_s', float('nan')):.1f}s + Lambda "
          f"{tm.get('v_lambda_s', float('nan')):.1f}s] + write {tm['write_s']:.2f}s")

    t = time.time()
    metrics = D.inspect_metrics(mt, D.EIFFEL_REQUEST, hp, uv)
    bench["inspect_s"] = time.time() - t
    bench["metrics_preview"] = {k: metrics[k] for k in
                                ("delta_rank", "delta_fro", "k_star_norm",
                                 "lambda_norm") if k in metrics}
    stamp(f"inspect in {bench['inspect_s']:.1f}s | rank(delta)={metrics['delta_rank']} "
          f"||delta||_F={metrics['delta_fro']:.3f}")

    D.restore_weights(mt.model, orig_w)
    stamp("weights restored")

    # ---- 5. probes cost (5 probes, greedy + sampled + 2 probabilities) ----
    t = time.time()
    D.eval_probes(mt, D.EIFFEL_PROBES, "Rome", seed=0)
    bench["probes_5_s"] = time.time() - t
    stamp(f"5 probes in {bench['probes_5_s']:.1f}s")

    # Second edit with reduced steps, to size Fast mode for Tab 2.
    if tm["total_s"] > 300:
        fast_steps = max(5, hp.v_num_grad_steps // 3)
        stamp(f"edit > 5 min -> measuring Fast edit with v_num_grad_steps={fast_steps}")
        orig_w2, _, tm2 = D.apply_edit(mt, D.EIFFEL_REQUEST, hp,
                                       v_steps_override=fast_steps)
        D.restore_weights(mt.model, orig_w2)
        bench["edit_fast"] = tm2
        bench["fast_v_steps"] = fast_steps
        stamp(f"fast edit total {tm2['total_s']:.1f}s")
    else:
        bench["edit_fast"] = None
        bench["fast_v_steps"] = hp.v_num_grad_steps
        stamp("edit <= 5 min -> Fast mode keeps the paper's v_num_grad_steps")

    bench["wall_total_s"] = time.time() - t0
    OUT.write_text(json.dumps(bench, indent=2, ensure_ascii=False))
    stamp(f"wrote {OUT} (total {bench['wall_total_s']/60:.1f} min)")


if __name__ == "__main__":
    main()
