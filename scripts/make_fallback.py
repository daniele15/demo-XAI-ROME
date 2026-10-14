"""Build a self-contained fallback.html (images inlined as base64)."""
import base64, html, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRE = ROOT / "prerendered"
OUT = ROOT / "fallback.html"


def b64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode()


def main():
    summary = json.loads((PRE / "trace_summary.json").read_text())
    edit = json.loads((PRE / "edit_eiffel.json").read_text())

    runs = {(r["slug"], r["kind_name"]): r for r in summary["runs"]}

    # Heatmaps for the Eiffel prompt are what the edit example talks about;
    # the Space Needle set is the canonical Figure-1 reproduction.
    figs = []
    for slug, label in [("space_needle", "The Space Needle is in downtown"),
                        ("eiffel_tower", "The Eiffel Tower is located in the city of"),
                        ("michael_jordan", "Michael Jordan plays the sport of")]:
        for kind in ("hidden", "mlp", "attn"):
            r = runs.get((slug, kind))
            if not r:
                continue
            p = PRE / r["png"]
            if not p.exists():
                continue
            sites = ", ".join(
                f"({html.escape(s['token'])}, L{s['layer']}, IE={s['ie']:.3f})"
                for s in r["top_sites"][:3])
            figs.append(f"""
      <figure>
        <img src="data:image/png;base64,{b64(p)}" alt="{kind} trace for {html.escape(label)}">
        <figcaption>
          <b>{html.escape(label)}</b> — <code>{kind}</code><br>
          answer <code>{html.escape(r['answer'])}</code> ·
          P[o]=<b>{r['p_clean']:.4f}</b> · P*[o]=<b>{r['p_corrupted']:.4f}</b> ·
          TE=<b>{r['total_effect']:.4f}</b> · {r['elapsed_s']:.0f}s<br>
          top-3 IE: {sites}
        </figcaption>
      </figure>""")

    rows = []
    for p in edit["probes"]:
        ok = p["passes"]
        cls = "ok" if ok else "bad"
        want = "target thắng" if p["expect"] == "target" else f"{html.escape(p['orig_answer'])} vẫn thắng"
        def cell(d):
            return (f"<div class='gen'><b>greedy:</b> {html.escape(d['greedy'])}</div>"
                    f"<div class='gen dim'><b>sampled:</b> {html.escape(d['sampled'])}</div>"
                    f"<div class='p'><code>P[{html.escape(edit['target'])}]</code>=<b>{d['p_target']:.5f}</b>"
                    f" · <code>P[{html.escape(p['orig_answer'])}]</code>=<b>{d['p_orig']:.5f}</b></div>")
        rows.append(f"""
      <tr>
        <td class="{cls}"><b>{html.escape(p['label'])}</b>
          <div class="pr"><code>{html.escape(p['prompt'])}</code></div>
          <div class="pr">kỳ vọng: {want}<br>{'✅ đạt' if ok else '❌ không đạt'}</div></td>
        <td>{cell(p['before'])}</td>
        <td class="{cls}">{cell(p['after'])}</td>
      </tr>""")

    m = edit["metrics"]
    doc = f"""<!DOCTYPE html>
<html lang="vi"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ROME — bản dự phòng offline</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font-family: system-ui, -apple-system, sans-serif; max-width: 1100px;
         margin: 0 auto; padding: 24px 16px; line-height: 1.55; }}
  h1 {{ margin-bottom: 4px; }}
  .sub {{ opacity: .7; margin-bottom: 24px; }}
  .warn {{ background: rgba(230,160,30,.15); border-left: 4px solid #e6a01e;
           padding: 10px 14px; border-radius: 4px; margin: 18px 0; }}
  figure {{ margin: 0 0 28px; }}
  img {{ width: 100%; height: auto; border: 1px solid rgba(128,128,128,.3);
         border-radius: 6px; }}
  figcaption {{ font-size: 13px; opacity: .85; margin-top: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  th, td {{ border: 1px solid rgba(128,128,128,.4); padding: 9px;
            vertical-align: top; text-align: left; font-size: 13px; }}
  td.ok {{ background: rgba(34,160,70,.14); }}
  td.bad {{ background: rgba(205,40,40,.14); }}
  .gen {{ margin-bottom: 5px; }} .dim {{ opacity: .72; }}
  .p {{ margin-top: 4px; }} .pr {{ font-size: 11px; opacity: .8; margin-top: 5px; }}
  code {{ background: rgba(128,128,128,.16); padding: 1px 4px; border-radius: 3px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(300px,1fr)); gap: 18px; }}
</style></head><body>
<h1>ROME — bản dự phòng offline</h1>
<div class="sub">
  Kết quả đã chạy sẵn cho seminar. Model <code>{html.escape(summary['model'])}</code>
  trên <code>{html.escape(summary['device'])}</code>,
  {summary['num_layers']} layer · ν = 3σ = <code>{summary['noise_nu']:.4f}</code>
  (σ = <code>{summary['sigma']:.4f}</code>).
</div>
<div class="warn">
  Trang này <b>tự chứa</b>: ảnh nhúng base64, không cần server, không cần mạng,
  không cần GPU. Mở trực tiếp bằng trình duyệt nếu app Gradio không chạy được.
</div>

<h2>1. Causal tracing</h2>
<div class="grid">{''.join(figs)}</div>

<h2>2. Một phép sửa ROME: <code>The Eiffel Tower</code> → <code>{html.escape(edit['target'])}</code></h2>
<p>Request: <code>{html.escape(edit['request']['prompt'])}</code> ·
   layer <b>{m['layer']}</b> · <code>{html.escape(m['weight_name'])}</code> ·
   edit mất <b>{edit['timings']['edit_s']:.1f}s</b> ·
   trọng số đã khôi phục: <b>{edit['weights_restored']}</b></p>
<table>
  <thead><tr><th style="width:19%">Probe</th><th style="width:40.5%">Trước</th>
  <th style="width:40.5%">Sau</th></tr></thead>
  <tbody>{''.join(rows)}</tbody>
</table>

<h2>3. Toán của phép sửa</h2>
<ul>
  <li><b>W_proj</b>: <code>{m['W_shape'][0]} × {m['W_shape'][1]}</code> (H×D)</li>
  <li><b>‖k*‖</b> = <code>{m['k_star_norm']:.4f}</code></li>
  {"<li><b>k*ᵀC⁻¹k*</b> = <code>%.4f</code> · ‖C⁻¹k*‖ = <code>%.4f</code></li>" % (m['k_Cinv_k'], m['c_inv_k_norm']) if 'k_Cinv_k' in m else ""}
  <li><b>‖Λ‖</b> = <code>{m['lambda_norm']:.4f}</code></li>
  <li><b>‖Δ‖_F</b> = <code>{m['delta_fro']:.4f}</code></li>
  <li><b>rank(Δ)</b> = <code>{m['delta_rank']}</code> (kỳ vọng = 1 — cập nhật hạng 1)</li>
</ul>
</body></html>"""

    OUT.write_text(doc, encoding="utf-8")
    kb = OUT.stat().st_size / 1024
    print(f"wrote {OUT} ({kb:.0f} KB, {len(figs)} figures)")


if __name__ == "__main__":
    main()
