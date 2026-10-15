"""
Measure the runner-up loud-fact candidate in the same two forms.

pick_loud_fact.py ranks by P[answer] alone, which picks The Taj Mahal -- but
that shares its relation ("is located in the city of") with the neighborhood
probe. Shakespeare keeps the relation of the probe being replaced ("was born
in the city of"), so it tests leakage ACROSS relations. Measure it the same
way so the choice is made on numbers, not on my preference.
"""
import json, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import demo_utils as D

OUT = D.PRERENDERED_DIR / "loud_fact_runnerup.json"
TARGET = "Rome"
EDITED_FACT = "The Eiffel Tower is located in the city of Rome."
PROMPT = "William Shakespeare was born in the city of"
ANSWER = "Stratford"


def stamp(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    mt, _ = D.load_model()
    hp = D.get_hparams()
    forms = {"standalone": PROMPT, "counterfact_plus": f"{EDITED_FACT} {PROMPT}"}

    before = {}
    for k, pr in forms.items():
        g, _ = D.generate_texts(mt, pr, max_new=20, seed=0)
        before[k] = dict(prompt=pr, p_orig=D.prob_of(mt, pr, ANSWER),
                         p_target=D.prob_of(mt, pr, TARGET), greedy=g)
        stamp(f"TRUOC {k:17} P[{ANSWER}]={before[k]['p_orig']:.5f} "
              f"P[{TARGET}]={before[k]['p_target']:.5f}")

    orig_w, uv, tm = D.apply_edit(mt, D.EIFFEL_REQUEST, hp, seed=0)
    stamp(f"edit {tm['total_s']:.1f}s")

    after = {}
    for k, pr in forms.items():
        g, _ = D.generate_texts(mt, pr, max_new=20, seed=0)
        after[k] = dict(prompt=pr, p_orig=D.prob_of(mt, pr, ANSWER),
                        p_target=D.prob_of(mt, pr, TARGET), greedy=g)
        leak = after[k]["p_target"] / max(before[k]["p_target"], 1e-12)
        stamp(f"SAU   {k:17} P[{ANSWER}]={after[k]['p_orig']:.5f} "
              f"P[{TARGET}]={after[k]['p_target']:.5f}  (x{leak:.1f})")
        stamp(f"    greedy: {g[len(pr):][:80]!r}")

    D.restore_weights(mt.model, orig_w)
    stamp("trong so da khoi phuc")

    OUT.write_text(json.dumps(dict(
        prompt=PROMPT, answer=ANSWER, target=TARGET, edited_fact=EDITED_FACT,
        relation="was born in the city of (khac quan he voi probe neighborhood)",
        before=before, after=after,
        verdict={k: dict(orig_still_wins=after[k]["p_orig"] > after[k]["p_target"],
                         leak_factor=after[k]["p_target"] / max(before[k]["p_target"], 1e-12))
                 for k in forms}), indent=2, ensure_ascii=False))
    stamp(f"wrote {OUT}")


if __name__ == "__main__":
    main()
