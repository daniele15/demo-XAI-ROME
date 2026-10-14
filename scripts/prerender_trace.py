"""Step 2a: causal tracing for the three demo prompts, full resolution."""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo_utils as D


def stamp(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

PROMPTS = [
    ("space_needle",    "The Space Needle is in downtown",              "The Space Needle"),
    ("eiffel_tower",    "The Eiffel Tower is located in the city of",   "The Eiffel Tower"),
    ("michael_jordan",  "Michael Jordan plays the sport of",            "Michael Jordan"),
]
KINDS = [(None, "hidden"), ("mlp", "mlp"), ("attn", "attn")]

OUT = D.PRERENDERED_DIR
SUMMARY = OUT / "trace_summary.json"


def main():
    stamp(f"START prerender_trace | device={D.DEVICE} model={D.MODEL_NAME} "
          f"samples={D.FULL_SAMPLES} (Full/paper)")
    mt, load_s = D.load_model()
    stamp(f"model loaded in {load_s:.1f}s ({mt.num_layers} layers)")

    nu, sigma, cached = D.get_noise_level(mt)
    stamp(f"sigma(embedding)={sigma:.4f}  nu=3*sigma={nu:.4f}  (cached={cached})")

    summary = dict(model=D.MODEL_NAME, device=D.DEVICE, sigma=sigma, noise_nu=nu,
                   num_layers=mt.num_layers, runs=[])
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text())
        summary["runs"] = [r for r in summary.get("runs", [])]
    done = {(r["slug"], r["kind_name"]) for r in summary.get("runs", [])}

    for slug, prompt, subject in PROMPTS:
        for kind, kind_name in KINDS:
            png_path = OUT / f"trace_{slug}_{kind_name}.png"
            if (slug, kind_name) in done and png_path.exists():
                stamp(f"SKIP {slug}/{kind_name} (already done)")
                continue
            stamp(f"RUN  {slug}/{kind_name} ...")
            t0 = time.time()
            res = D.trace(mt, prompt, subject, kind, noise=nu,
                          samples=D.FULL_SAMPLES, window=10)
            el = time.time() - t0

            png = png_path
            D.save_heatmap(res, png, title=None, modelname=D.MODEL_NAME, dpi=200)

            sites = D.top_sites(res, 3)
            row = dict(
                slug=slug, prompt=prompt, subject=subject, kind_name=kind_name,
                png=png.name, answer=res["answer"],
                p_clean=float(res["high_score"]), p_corrupted=float(res["low_score"]),
                total_effect=float(res["high_score"]) - float(res["low_score"]),
                input_tokens=res["input_tokens"],
                subject_range=list(res["subject_range"]),
                top_sites=sites, elapsed_s=el, window=10,
                samples=D.FULL_SAMPLES,
            )
            summary["runs"] = [r for r in summary["runs"]
                               if not (r["slug"] == slug and r["kind_name"] == kind_name)]
            summary["runs"].append(row)
            SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

            stamp(f"DONE {slug}/{kind_name} in {el:.1f}s -> {png.name} | "
                  f"answer={res['answer']!r} P[o]={row['p_clean']:.4f} "
                  f"P*[o]={row['p_corrupted']:.4f} TE={row['total_effect']:.4f}")
            stamp("     top sites: " + ", ".join(
                f"({s['token']!r}, L{s['layer']}, IE={s['ie']:.4f})" for s in sites))

    stamp(f"FINISHED prerender_trace -> {SUMMARY}")


if __name__ == "__main__":
    main()
