"""
Fast contract test for app.run_edit -- no model, runs in seconds.

A live edit takes ~9 minutes on CPU, so a KeyError in the post-edit path costs
nine minutes to discover. That actually happened: run_edit built its probe
dicts with the key "orig" but read probes[0]["orig_answer"] in two places, and
only a real end-to-end click surfaced it.

This stubs out everything that needs the model and drives run_edit for real,
so the dict contracts between run_edit, eval_probes, probe_passes,
rows_to_html and restore_and_verify are checked without loading gpt2-xl.
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app
import demo_utils as D


def fake_eval_probes(mt, probes, target, seed=0, progress=None):
    """Mirror the real shape: key, label, prompt, orig_answer, expect, ..."""
    return [dict(key=p["key"], label=p["label"], prompt=p["prompt"],
                 orig_answer=p["orig"], expect=p["expect"],
                 greedy="g", sampled="s", p_target=0.9, p_orig=0.1)
            for p in probes]


class _StubMT:
    """Needs a .model attribute: run_edit's finally block touches MT.model,
    and a bare object() there masks the real error with an AttributeError."""
    model = object()


def main():
    app.MT = _StubMT()
    app.HPARAMS = D.get_hparams()
    app.BENCH = {"edit": {"total_s": 225.4, "u_s": 9.2, "c_was_cached": True},
                 "probes_5_s": 58.1}

    timings = dict(total_s=1.0, u_s=0.1, v_opt_s=0.8, v_lambda_s=0.05,
                   write_s=0.05, v_num_grad_steps=20, v_steps_overridden=False,
                   c_was_cached=True, seed=0)
    metrics = dict(layer=17, layer_label="layer 18 = transformer.h.17",
                   weight_name="transformer.h.17.mlp.c_proj.weight",
                   W_shape=(6400, 1600), delta_shape=(6400, 1600),
                   W_fro=117.39, delta_over_W=0.072, k_star_norm=17.74,
                   u_norm=1.0, lambda_norm=8.46, delta_fro=8.45,
                   delta_fro_analytic=8.457, delta_rank=1,
                   division_factor=11.84, mom2_adjustment=True,
                   c_inv_k_norm=1073.26, k_Cinv_k=13842.14, C_shape=(6400, 6400))

    D_orig = {k: getattr(D, k) for k in
              ("eval_probes", "apply_edit", "inspect_metrics",
               "restore_and_verify", "restore_weights")}
    D.eval_probes = fake_eval_probes
    D.apply_edit = lambda *a, **k: ({"w": None}, {}, timings)
    D.inspect_metrics = lambda *a, **k: metrics
    D.restore_and_verify = lambda mt, w, prompt, target, orig, \
        p_target_before, p_orig_before, tol=1e-6: dict(
            p_target=p_target_before, p_orig=p_orig_before,
            delta_target=0.0, delta_orig=0.0, target_ok=True, orig_ok=True,
            weights_identical=True, all_ok=True)
    D.restore_weights = lambda *a, **k: None

    failures = []
    try:
        for mode in ("Full", "Fast"):
            status, table, md = app.run_edit(
                "The Eiffel Tower", "{} is located in the city of", "Rome", mode,
                "The Eiffel Tower is located in the city of",
                "Where is the Eiffel Tower? It is in",
                "The Louvre is located in the city of",
                "The Eiffel Tower is right across from",
                D.LOUD_FACT_PROMPT, "Paris", D.LOUD_FACT_ANSWER,
                progress=lambda *a, **k: None)
            if status.startswith("### Lỗi"):
                failures.append(f"{mode}: run_edit trả về lỗi\n{status}")
                continue
            for needle, where in (("khôi phục", "status"),
                                  ("<table", "bảng"), ("rank", "metrics")):
                if needle not in (status if where == "status"
                                  else table if where == "bảng" else md):
                    failures.append(f"{mode}: thiếu {needle!r} trong {where}")
            n_rows = table.count("<tr>") - 1  # minus the header row
            if n_rows != len(D.EIFFEL_PROBES):
                failures.append(f"{mode}: {n_rows} dòng, cần {len(D.EIFFEL_PROBES)}")
            print(f"  [{mode}] OK — {n_rows} dòng bảng, "
                  f"v_steps={'10' if mode == 'Fast' else '20'}")
    finally:
        for k, v in D_orig.items():
            setattr(D, k, v)

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(" -", f)
        return 1
    print("\nPASS: hợp đồng dict giữa run_edit và các hàm nó gọi là nhất quán")
    return 0


if __name__ == "__main__":
    sys.exit(main())
