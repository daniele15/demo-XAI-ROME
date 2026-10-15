"""
Hard gate before Tab 2 is exposed to anyone: prove the model really returns to
its pre-edit state.

Three independent checks, strongest last:
  1. P[target] on the rewrite prompt returns to its pre-edit value
  2. P[original answer] returns to its pre-edit value  -- the sensitive one,
     because it is ~0.93 rather than ~0.0001
  3. the edited weight tensor itself is bit-identical to the saved copy
"""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import demo_utils as D
from util import nethook

OUT = D.PRERENDERED_DIR / "restore_verification.json"
PROMPT = "The Eiffel Tower is located in the city of"
TARGET, ORIG = "Rome", "Paris"
TOL = 1e-6


def stamp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    mt, _ = D.load_model()
    hp = D.get_hparams()
    wname = f"{hp.rewrite_module_tmp.format(hp.layers[0])}.weight"

    W = nethook.get_parameter(mt.model, wname)
    w_before = W.detach().clone()
    p_t0 = D.prob_of(mt, PROMPT, TARGET)
    p_o0 = D.prob_of(mt, PROMPT, ORIG)
    stamp(f"BEFORE  P[{TARGET}]={p_t0:.8f}  P[{ORIG}]={p_o0:.8f}")

    orig_w, uv, tm = D.apply_edit(mt, D.EIFFEL_REQUEST, hp)
    p_t1 = D.prob_of(mt, PROMPT, TARGET)
    p_o1 = D.prob_of(mt, PROMPT, ORIG)
    w_edited = nethook.get_parameter(mt.model, wname).detach().clone()
    stamp(f"EDITED  P[{TARGET}]={p_t1:.8f}  P[{ORIG}]={p_o1:.8f}  "
          f"(edit {tm['total_s']:.1f}s)")

    D.restore_weights(mt.model, orig_w)
    p_t2 = D.prob_of(mt, PROMPT, TARGET)
    p_o2 = D.prob_of(mt, PROMPT, ORIG)
    w_after = nethook.get_parameter(mt.model, wname).detach().clone()
    stamp(f"RESTORED P[{TARGET}]={p_t2:.8f}  P[{ORIG}]={p_o2:.8f}")

    d_t, d_o = abs(p_t2 - p_t0), abs(p_o2 - p_o0)
    w_maxdiff = float((w_after - w_before).abs().max())
    w_identical = bool(torch.equal(w_after, w_before))
    edit_maxdiff = float((w_edited - w_before).abs().max())

    checks = {
        f"P[{TARGET}] returns (<{TOL})": (d_t < TOL, f"|Δ| = {d_t:.3e}"),
        f"P[{ORIG}] returns (<{TOL})": (d_o < TOL, f"|Δ| = {d_o:.3e}"),
        "weight tensor bit-identical": (w_identical, f"max|Δw| = {w_maxdiff:.3e}"),
        "the edit actually changed W": (edit_maxdiff > 0, f"max|Δw| = {edit_maxdiff:.3e}"),
    }
    stamp("")
    for name, (ok, detail) in checks.items():
        stamp(f"  [{'PASS' if ok else 'FAIL'}] {name:34} {detail}")

    all_ok = all(ok for ok, _ in checks.values())
    stamp("")
    stamp(f"VERDICT: {'ALL CHECKS PASSED' if all_ok else 'FAILED -- do not ship Tab 2'}")

    OUT.write_text(json.dumps(dict(
        model=D.MODEL_NAME, device=D.DEVICE, prompt=PROMPT,
        target=TARGET, orig=ORIG, tolerance=TOL,
        before={TARGET: p_t0, ORIG: p_o0},
        edited={TARGET: p_t1, ORIG: p_o1},
        restored={TARGET: p_t2, ORIG: p_o2},
        delta_target=d_t, delta_orig=d_o,
        weight_name=wname, weight_max_abs_diff=w_maxdiff,
        weight_bit_identical=w_identical, edit_max_abs_diff=edit_maxdiff,
        checks={k: v[0] for k, v in checks.items()},
        all_passed=all_ok), indent=2, ensure_ascii=False))
    stamp(f"wrote {OUT}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
