"""
Replace the loud-fact probe, on evidence.

The current probe ("Barack Obama was born in the city of") is unusable: gpt2-xl
generates a false birther claim, and P[Honolulu]=0.183 is weaker than the
neighborhood probe, so it is not a "loud" fact at all.

Phase 1  measure candidates on the unedited model; keep those with
         P[answer] >= 0.7, clean greedy text, answer not in {Rome, Paris}.
Phase 2  apply the Eiffel->Rome edit once and measure the winner in BOTH forms:
           A. standalone
           B. CounterFact+ composed: the edited fact stated first, which is the
              setup that makes a loud fact leak in the paper.
"""
import json, re, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo_utils as D

OUT = D.PRERENDERED_DIR / "loud_fact_selection.json"
TARGET = "Rome"
EDITED_FACT = "The Eiffel Tower is located in the city of Rome."
MIN_P = 0.70
BANNED = {"rome", "paris"}

CANDIDATES = [
    ("William Shakespeare was born in the city of", "Stratford"),
    ("Wolfgang Amadeus Mozart was born in the city of", "Salzburg"),
    ("Napoleon Bonaparte was born in the city of", "Ajaccio"),
    ("The Statue of Liberty is located in the city of", "New York"),
    ("Big Ben is located in the city of", "London"),
    ("The Taj Mahal is located in the city of", "Agra"),
    ("The Brandenburg Gate is located in the city of", "Berlin"),
    ("The Kremlin is located in the city of", "Moscow"),
    ("The Sydney Opera House is located in the city of", "Sydney"),
    ("The Golden Gate Bridge is located in the city of", "San Francisco"),
]

REPEAT = re.compile(r"(\b\w[\w' ]{6,}?\b)[\s,.]*\1", re.I)


def stamp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def composed(prompt):
    return f"{EDITED_FACT} {prompt}"


def main():
    mt, _ = D.load_model()
    hp = D.get_hparams()

    # ---------------- Phase 1 ----------------
    stamp("=== Phase 1: do cac ung vien tren model CHUA sua ===")
    rows = []
    for prompt, ans in CANDIDATES:
        p = D.prob_of(mt, prompt, ans)
        greedy, _ = D.generate_texts(mt, prompt, max_new=16, seed=0)
        clean = REPEAT.search(greedy) is None
        ok = p >= MIN_P and clean and ans.lower() not in BANNED
        rows.append(dict(prompt=prompt, answer=ans, p_answer=p, greedy=greedy,
                         greedy_clean=clean, eligible=ok,
                         top3=D.top_k_next(mt, prompt, 3)))
        stamp(f"  P[{ans}]={p:.4f} clean={clean} ok={ok}  {prompt[:46]!r}")
        stamp(f"      greedy: {greedy[len(prompt):][:70]!r}")

    eligible = sorted([r for r in rows if r["eligible"]],
                      key=lambda r: -r["p_answer"])
    if not eligible:
        stamp("KHONG co ung vien nao dat nguong -> giu nguyen probe cu, bao cao")
        OUT.write_text(json.dumps(dict(candidates=rows, chosen=None),
                                  indent=2, ensure_ascii=False))
        return 1
    best = eligible[0]
    stamp(f"\n  -> CHON: {best['prompt']!r} / {best['answer']} "
          f"(P={best['p_answer']:.4f})")

    # ---------------- Phase 2 ----------------
    stamp("\n=== Phase 2: do 2 dang, truoc va sau edit ===")
    forms = {"standalone": best["prompt"], "counterfact_plus": composed(best["prompt"])}

    before = {}
    for k, pr in forms.items():
        g, _ = D.generate_texts(mt, pr, max_new=20, seed=0)
        before[k] = dict(prompt=pr, p_orig=D.prob_of(mt, pr, best["answer"]),
                         p_target=D.prob_of(mt, pr, TARGET), greedy=g)
        stamp(f"  TRUOC {k:17} P[{best['answer']}]={before[k]['p_orig']:.5f} "
              f"P[{TARGET}]={before[k]['p_target']:.5f}")

    stamp("  dang chay edit Eiffel -> Rome ...")
    orig_w, uv, tm = D.apply_edit(mt, D.EIFFEL_REQUEST, hp, seed=0)
    stamp(f"  edit xong {tm['total_s']:.1f}s")

    after = {}
    for k, pr in forms.items():
        g, _ = D.generate_texts(mt, pr, max_new=20, seed=0)
        after[k] = dict(prompt=pr, p_orig=D.prob_of(mt, pr, best["answer"]),
                        p_target=D.prob_of(mt, pr, TARGET), greedy=g)
        leak = after[k]["p_target"] / max(before[k]["p_target"], 1e-12)
        stamp(f"  SAU   {k:17} P[{best['answer']}]={after[k]['p_orig']:.5f} "
              f"P[{TARGET}]={after[k]['p_target']:.5f}  (x{leak:.1f} ro ri)")
        stamp(f"      greedy: {g[len(pr):][:80]!r}")

    D.restore_weights(mt.model, orig_w)
    stamp("  trong so da khoi phuc")

    verdict = {}
    for k in forms:
        verdict[k] = dict(
            orig_still_wins=after[k]["p_orig"] > after[k]["p_target"],
            leak_factor=after[k]["p_target"] / max(before[k]["p_target"], 1e-12))
    stamp("\n=== KET LUAN ===")
    for k, v in verdict.items():
        stamp(f"  {k:17} dap an goc van thang: {v['orig_still_wins']} "
              f"| he so ro ri P[{TARGET}]: x{v['leak_factor']:.1f}")

    OUT.write_text(json.dumps(dict(
        model=D.MODEL_NAME, device=D.DEVICE, target=TARGET,
        edited_fact=EDITED_FACT, criteria=dict(min_p_answer=MIN_P,
                                               banned_answers=sorted(BANNED)),
        candidates=rows, chosen=dict(prompt=best["prompt"], answer=best["answer"],
                                     p_answer=best["p_answer"]),
        before=before, after=after, verdict=verdict,
        edit_timings=tm), indent=2, ensure_ascii=False))
    stamp(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
