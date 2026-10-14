"""Inject the measured timing table from prerendered/benchmark.json into README.md."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "prerendered" / "benchmark.json"
README = ROOT / "README.md"
START, END = "<!-- BENCH_TABLE -->", "<!-- /BENCH_TABLE -->"


def dur(s):
    if s is None:
        return "—"
    if s < 1:
        return f"{s*1000:.0f} ms"
    if s < 90:
        return f"{s:.1f} s"
    return f"{s/60:.1f} phút"


def main():
    if not BENCH.exists():
        print(f"no {BENCH}; nothing to do")
        return 1
    b = json.loads(BENCH.read_text())
    e = b.get("edit", {}) or {}
    ef = b.get("edit_fast") or {}

    nl = b.get("num_layers", 48)
    ntok = b.get("prompt_tokens", 7)

    rows = [
        ("Nạp model (fp32, từ cache đĩa)", dur(b.get("model_load_s")), ""),
        (f"1 forward pass, batch 1, {ntok} token", dur(b.get("forward_1x7_s")),
         "trung vị trên 7 lần, sau 1 lần làm nóng"),
        (f"Heatmap `hidden` **Fast** (3 mẫu)", dur(b.get("trace_hidden_fast_s")),
         f"{ntok}×{nl} = {ntok*nl} lần khôi phục, batch 4"),
        (f"Heatmap `hidden` **Full** (10 mẫu)", dur(b.get("trace_hidden_full_s")),
         f"{ntok}×{nl} = {ntok*nl} lần khôi phục, batch 11"),
        ("**ROME edit, tổng**", dur(e.get("total_s")),
         f"`v_num_grad_steps={e.get('v_num_grad_steps','?')}`"),
        ("├ `compute_u` (k\\*, C⁻¹)", dur(e.get("u_s")),
         "lần đầu gồm tải C stats 156 MB + nghịch đảo 6400²"),
        ("├ tối ưu v\\*", dur(e.get("v_opt_s")), "vòng lặp gradient"),
        ("├ tính Λ", dur(e.get("v_lambda_s")), "giải hệ tuyến tính + 2 forward"),
        ("└ ghi W (cập nhật hạng 1)", dur(e.get("write_s")), "cộng Δ vào W_proj"),
        ("`inspect_metrics` (gồm `matrix_rank`)", dur(b.get("inspect_s")),
         "SVD trên Δ 6400×1600"),
        ("5 probe (greedy + sampled + 2 xác suất)", dur(b.get("probes_5_s")),
         "Tab 2 chạy 2 lần: trước và sau"),
    ]
    if ef:
        rows.append(("ROME edit **Fast**", dur(ef.get("total_s")),
                     f"`v_num_grad_steps={b.get('fast_v_steps')}` — **lệch hparams**"))

    tbl = ["| Thao tác | Thời gian | Ghi chú |", "|---|---|---|"]
    tbl += [f"| {a} | **{t}** | {n} |" for a, t, n in rows]

    hdr = (f"Đo trên máy này: `{b.get('model')}` fp32 trên **{b.get('device')}**, "
           f"Intel Core Ultra 7 165H, torch `{b.get('torch')}`, "
           f"{b.get('threads','?')} thread.\n\n")
    foot = ("\n\n**Tab 2 trong thực tế** = edit + 2×(5 probe) ≈ "
            f"**{dur((e.get('total_s') or 0) + 2*(b.get('probes_5_s') or 0))}** "
            "cho lượt đầu (đã có C stats trong cache thì nhanh hơn).\n")

    txt = README.read_text()
    i, j = txt.index(START), txt.index(END)
    README.write_text(txt[:i + len(START)] + "\n" + hdr + "\n".join(tbl) + foot + txt[j:])
    print(f"README.md updated with {len(rows)} measured rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
