"""Step 2b: one ROME edit (The Eiffel Tower -> Rome), before/after on 5 probes."""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import demo_utils as D

OUT = D.PRERENDERED_DIR / "edit_eiffel.json"
TARGET = D.EIFFEL_REQUEST["target_new"]["str"]
SEED = 0


def main():
    print(f"device={D.DEVICE} model={D.MODEL_NAME}")
    mt, load_s = D.load_model()
    hp = D.get_hparams()
    print(f"loaded in {load_s:.1f}s | layers={hp.layers} mom2_adjustment={hp.mom2_adjustment}",
          flush=True)

    print("\n--- BEFORE ---", flush=True)
    t0 = time.time()
    before = D.eval_probes(mt, D.EIFFEL_PROBES, TARGET, seed=SEED)
    t_before = time.time() - t0
    for r in before:
        print(f"  {r['key']:13} P[{TARGET}]={r['p_target']:.5f}  "
              f"P[{r['orig_answer']}]={r['p_orig']:.5f}")
    print(f"  ({t_before:.1f}s)", flush=True)

    print("\n--- EDIT ---", flush=True)
    print("(first run downloads C stats ~156 MB from rome.baulab.info)", flush=True)
    orig_weights, uv, tm = D.apply_edit(mt, D.EIFFEL_REQUEST, hp)
    print(f"edit done in {tm['total_s']:.1f}s = compute_u {tm['u_s']:.1f}s "
          f"(C cached before: {tm['c_was_cached']}) + v* opt "
          f"{tm.get('v_opt_s', float('nan')):.1f}s + Lambda "
          f"{tm.get('v_lambda_s', float('nan')):.1f}s + write {tm['write_s']:.2f}s",
          flush=True)

    metrics = D.inspect_metrics(mt, D.EIFFEL_REQUEST, hp, uv)
    print("\n--- INSPECT ---")
    for k, v in metrics.items():
        print(f"  {k:18} = {v}")

    print("\n--- AFTER ---", flush=True)
    t0 = time.time()
    after = D.eval_probes(mt, D.EIFFEL_PROBES, TARGET, seed=SEED)
    t_after = time.time() - t0
    for r in after:
        ok = D.probe_passes(r, TARGET)
        print(f"  {r['key']:13} P[{TARGET}]={r['p_target']:.5f}  "
              f"P[{r['orig_answer']}]={r['p_orig']:.5f}  "
              f"expect={r['expect']:6} -> {'OK' if ok else 'MISS'}")
    print(f"  ({t_after:.1f}s)", flush=True)

    # Restore and verify we really are back where we started.
    D.restore_weights(mt.model, orig_weights)
    check = D.prob_of(mt, D.EIFFEL_PROBES[0]["prompt"], TARGET)
    restored_ok = abs(check - before[0]["p_target"]) < 1e-6
    print(f"\nweights restored: P[{TARGET}] on rewrite prompt "
          f"{after[0]['p_target']:.5f} -> {check:.5f} "
          f"(before was {before[0]['p_target']:.5f}) | exact={restored_ok}")

    payload = dict(
        model=D.MODEL_NAME, device=D.DEVICE, seed=SEED,
        request=D.EIFFEL_REQUEST, target=TARGET,
        hparams=dict(layers=hp.layers, fact_token=hp.fact_token,
                     v_num_grad_steps=hp.v_num_grad_steps, v_lr=hp.v_lr,
                     mom2_adjustment=hp.mom2_adjustment,
                     rewrite_module=hp.rewrite_module_tmp.format(hp.layers[0])),
        timings=dict(load_s=load_s, before_s=t_before, edit_s=tm["total_s"],
                     after_s=t_after, edit_breakdown=tm),
        metrics=metrics,
        probes=[
            dict(key=b["key"], label=b["label"], prompt=b["prompt"],
                 orig_answer=b["orig_answer"], expect=b["expect"],
                 before=dict(greedy=b["greedy"], sampled=b["sampled"],
                             p_target=b["p_target"], p_orig=b["p_orig"]),
                 after=dict(greedy=a["greedy"], sampled=a["sampled"],
                            p_target=a["p_target"], p_orig=a["p_orig"]),
                 passes=D.probe_passes(a, TARGET))
            for b, a in zip(before, after)
        ],
        weights_restored=bool(restored_ok),
    )
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
