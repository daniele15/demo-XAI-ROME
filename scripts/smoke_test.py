"""Step-1 verification: model loads, and the canonical ROME prompt behaves."""
import sys, time, warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
import demo_utils as D

PROMPT = "The Space Needle is in downtown"

print(f"device={D.DEVICE}  model={D.MODEL_NAME}  torch={torch.__version__}")
print("loading model ...", flush=True)
mt, load_s = D.load_model()
nparams = sum(p.numel() for p in mt.model.parameters())
print(f"loaded in {load_s:.1f}s | {nparams/1e9:.3f}B params | {mt.num_layers} layers "
      f"| dtype={next(mt.model.parameters()).dtype}")

import resource
rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576
print(f"peak RSS after load: {rss:.2f} GB")

print(f"\nprompt: {PROMPT!r}")
t0 = time.time()
top5 = D.top_k_next(mt, PROMPT, k=5)
t_top5 = time.time() - t0
print("top-5 next tokens:")
for i, (tokstr, p) in enumerate(top5, 1):
    print(f"  {i}. {tokstr!r:16} p={p:.4f}")
print(f"  ({t_top5:.2f}s)")

t0 = time.time()
from util.generate import generate_fast
gen = generate_fast(mt.model, mt.tokenizer, [PROMPT], n_gen_per_prompt=1,
                    top_k=1, max_out_len=len(mt.tokenizer.encode(PROMPT)) + 10)
t_gen = time.time() - t0
print(f"\ngreedy +10 tokens ({t_gen:.2f}s):\n  {gen[0]!r}")

toks = [t.strip() for t, _ in top5]
ok = "Seattle" in toks
print(f"\n{'PASS' if ok else 'FAIL'}: 'Seattle' in top-5 "
      f"(rank {toks.index('Seattle')+1 if ok else 'n/a'})")
sys.exit(0 if ok else 1)
