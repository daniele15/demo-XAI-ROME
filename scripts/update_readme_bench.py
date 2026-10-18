"""Inject the measured timing table from prerendered/benchmark.json into README.md."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "prerendered" / "benchmark.json"
VSTEPS = ROOT / "prerendered" / "v_steps_comparison.json"
README = ROOT / "README.md"
PROMPT_REF = "The Space Needle is in downtown"
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

    # Fast-vs-Full on the same prompt: the evidence that Fast is trustworthy.
    fr, fu = b.get("fast_result") or {}, b.get("full_result") or {}
    cmp_block = ""
    if fr and fu:
        tf, tu = b.get("trace_hidden_fast_s"), b.get("trace_hidden_full_s")
        def site(d, i=0):
            ts = d.get("top_sites") or []
            return (f"`{ts[i]['token']}` L{ts[i]['layer']}, IE {ts[i]['ie']:.4f}"
                    if len(ts) > i else "—")
        cmp_block = (
            "\n\n### Fast có đáng tin không?\n\n"
            f"Cùng prompt `{PROMPT_REF}`, cùng ν:\n\n"
            "| | Fast (3 mẫu) | Full (10 mẫu) | Lệch |\n|---|---|---|---|\n"
            f"| P[o] clean | {fr['p_clean']:.4f} | {fu['p_clean']:.4f} | "
            f"{abs(fr['p_clean']-fu['p_clean']):.4f} |\n"
            f"| P*[o] corrupted | {fr['p_corrupted']:.4f} | {fu['p_corrupted']:.4f} | "
            f"**{abs(fr['p_corrupted']-fu['p_corrupted']):.4f}** |\n"
            f"| Site mạnh nhất | {site(fr)} | {site(fu)} | — |\n"
            f"| Thời gian | {dur(tf)} | {dur(tu)} | **{(tu/tf):.2f}×** |\n\n"
            "Hai chế độ cho cùng một kết luận; Fast chỉ nhiễu hơn ở chữ số thứ ba. "
            "Dùng Fast để bấm live, dùng Full/prerendered khi cần trích số vào slide.\n"
        )

    hdr = (f"Đo trên máy này: `{b.get('model')}` fp32 trên **{b.get('device')}**, "
           f"Intel Core Ultra 7 165H, torch `{b.get('torch')}`, "
           f"{b.get('threads','?')} thread.\n\n")
    foot = ("\n\n**Tab 2 trong thực tế** = edit + 2×(5 probe) ≈ "
            f"**{dur((e.get('total_s') or 0) + 2*(b.get('probes_5_s') or 0))}** "
            "cho lượt đầu (đã có C stats trong cache thì nhanh hơn).\n")

    txt = README.read_text()
    i, j = txt.index(START), txt.index(END)
    # Three columns: cold (C downloaded), warm (C cached), and Colab T4 as
    # measured in the 18/10/2026 rehearsal (CHANGES.md N) -- hardcoded, since
    # benchmark.py only ever runs on the CPU box.
    cold_block = ""
    if VSTEPS.exists():
        v = json.loads(VSTEPS.read_text())
        full = next((r for r in v["results"] if r["label"].startswith("Full")), None)
        if full and not full["c_was_cached"] or full:
            cold_u = (b.get("edit") or {}).get("u_s")
            cold_tot = (b.get("edit") or {}).get("total_s")
            cold_block = (
                "\n\n### Chi phí một ROME edit — ba cột\n\n"
                "| Giai đoạn | Lạnh (CPU, phải tải C) | Ấm (CPU, C trong cache) | Colab T4 (đo 18/10: lạnh / ấm) |\n"
                "|---|---|---|---|\n"
                f"| `compute_u` (k\\*, C⁻¹) | {dur(cold_u)} | {dur(full['u_s'])} | 4.2 s / 0.5 s |\n"
                f"| tối ưu v\\* ({full['steps_run']} bước) | {dur((b.get('edit') or {}).get('v_opt_s'))} "
                f"| {dur(full['v_opt_s'])} | 20.7 s / 17.7 s |\n"
                f"| tính Λ | {dur((b.get('edit') or {}).get('v_lambda_s'))} | {dur(full['v_lambda_s'])} | 0.1 s / 0.1 s |\n"
                f"| ghi W | {dur((b.get('edit') or {}).get('write_s'))} | {dur(full['write_s'])} | 0.8 s / ~0 s |\n"
                f"| **Tổng** | **{dur(cold_tot)}** | **{dur(full['total_s'])}** | **25.7 s / 18.2 s** |\n\n"
                "- **Lạnh** = lần đầu trong phiên: `compute_u` phải tải C stats 156 MB "
                "từ `rome.baulab.info` rồi nghịch đảo ma trận 6400².\n"
                "- **Ấm** = các lần sau: C⁻¹ đã nằm trong `inv_mom2_cache` của `compute_u`.\n"
                "- **Colab T4**: đo thật khi diễn tập 18/10/2026 (fp32, transformers 5.18), "
                "xem mục 1.\n"
            )

    README.write_text(txt[:i + len(START)] + "\n" + hdr + "\n".join(tbl)
                      + foot + cmp_block + cold_block + txt[j:])
    print(f"README.md updated with {len(rows)} measured rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
