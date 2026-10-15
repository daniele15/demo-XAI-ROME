"""
Decide whether a reduced-step edit is acceptable for Tab 2's Fast mode.

Acceptance rule (set by the user): a reduced-step edit is only acceptable if
BOTH rewrite and paraphrase still land target over the original answer, on the
same probes Full is measured on. If 6 steps fails paraphrase, try 10 before
concluding anything.
"""
import contextlib, io, json, re, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo_utils as D

OUT = D.PRERENDERED_DIR / "v_steps_comparison.json"
TARGET = "Rome"
LOSS_RE = re.compile(r"^loss ([\d.]+) = ([\d.]+) \+ ([\d.]+) \+ ([\d.]+)", re.M)


def stamp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def one_edit(mt, hp, steps, label):
    """Run an edit at `steps`, measure, restore. Returns a result row."""
    stamp(f"--- {label}: v_num_grad_steps={steps} ---")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        orig_w, uv, tm = D.apply_edit(
            mt, D.EIFFEL_REQUEST, hp,
            v_steps_override=None if steps == hp.v_num_grad_steps else steps)
    log = buf.getvalue()
    losses = LOSS_RE.findall(log)
    metrics = D.inspect_metrics(mt, D.EIFFEL_REQUEST, hp, uv)

    probes = [p for p in D.EIFFEL_PROBES if p["key"] in ("rewrite", "paraphrase")]
    rows = D.eval_probes(mt, probes, TARGET, seed=0)
    D.restore_weights(mt.model, orig_w)

    final = losses[-1] if losses else ("nan",) * 4
    row = dict(
        label=label, steps_requested=steps, steps_run=len(losses),
        final_loss=float(final[0]), final_nll=float(final[1]),
        final_kl=float(final[2]), final_wd=float(final[3]),
        delta_fro=metrics["delta_fro"], delta_rank=metrics["delta_rank"],
        W_fro=metrics["W_fro"], delta_over_W=metrics["delta_over_W"],
        lambda_norm=metrics["lambda_norm"],
        total_s=tm["total_s"], u_s=tm["u_s"],
        v_opt_s=tm.get("v_opt_s"), v_lambda_s=tm.get("v_lambda_s"),
        write_s=tm["write_s"], c_was_cached=tm["c_was_cached"],
        probes={r["key"]: dict(p_target=r["p_target"], p_orig=r["p_orig"],
                               wins=r["p_target"] > r["p_orig"],
                               greedy=r["greedy"]) for r in rows},
    )
    row["accepted"] = all(v["wins"] for v in row["probes"].values())
    stamp(f"    steps_run={row['steps_run']} final_loss={row['final_loss']:.4f} "
          f"(nll {row['final_nll']} + kl {row['final_kl']} + wd {row['final_wd']})")
    for k, v in row["probes"].items():
        stamp(f"    {k:11} P[{TARGET}]={v['p_target']:.5f} "
              f"P[Paris]={v['p_orig']:.5f} -> {'WIN' if v['wins'] else 'LOSS'}")
    stamp(f"    ||delta||_F={row['delta_fro']:.4f} "
          f"(/||W||={row['delta_over_W']:.5f})  total={row['total_s']:.1f}s")
    stamp(f"    ACCEPTED={row['accepted']}")
    return row


def main():
    mt, load_s = D.load_model()
    hp = D.get_hparams()
    stamp(f"model loaded {load_s:.1f}s | hparams v_num_grad_steps={hp.v_num_grad_steps}")

    results = [one_edit(mt, hp, hp.v_num_grad_steps, "Full (paper)")]
    results.append(one_edit(mt, hp, 6, "Fast-6"))

    if not results[-1]["accepted"]:
        stamp("Fast-6 did not pass both probes -> trying 10 steps")
        results.append(one_edit(mt, hp, 10, "Fast-10"))

    accepted = [r for r in results[1:] if r["accepted"]]
    choice = accepted[0]["steps_requested"] if accepted else None
    stamp(f"DECISION: reduced-step edit acceptable at "
          f"{choice if choice else 'NONE -- keep the paper hparams'}")

    OUT.write_text(json.dumps(
        dict(model=D.MODEL_NAME, device=D.DEVICE, target=TARGET,
             hparams_steps=hp.v_num_grad_steps,
             acceptance_rule="rewrite AND paraphrase must both have P[target] > P[orig]",
             results=results, recommended_fast_steps=choice),
        indent=2, ensure_ascii=False))
    stamp(f"wrote {OUT}")


if __name__ == "__main__":
    main()
