"""
Gradio demo for "Locating and Editing Factual Associations in GPT" (ROME).

Three tabs: causal tracing, one ROME edit with before/after probes, and the
algebra behind that edit. The model is loaded once at startup and shared.

Nothing in ./rome is modified; all adaptation lives in demo_utils.py.
"""
from __future__ import annotations

import html
import json
import os
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import gradio as gr
import torch

import demo_utils as D
from experiments.causal_trace import guess_subject

SHARE = os.environ.get("SHARE", "0").lower() in ("1", "true", "yes")
SERVER_NAME = os.environ.get("SERVER_NAME", "0.0.0.0")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "7860"))

# --------------------------------------------------------------------------
# Global state
# --------------------------------------------------------------------------
MT = None
HPARAMS = None
NOISE_NU = None
SIGMA = None
LOAD_S = 0.0
LAST_EDIT: dict = {}


def boot():
    global MT, HPARAMS, NOISE_NU, SIGMA, LOAD_S
    print(f"[boot] device={D.DEVICE} model={D.MODEL_NAME}", flush=True)
    MT, LOAD_S = D.load_model()
    HPARAMS = D.get_hparams()
    NOISE_NU, SIGMA, cached = D.get_noise_level(MT)
    load_bench()
    print(f"[boot] loaded in {LOAD_S:.1f}s | sigma={SIGMA:.4f} nu={NOISE_NU:.4f} "
          f"(cached={cached}) | benchmark={'yes' if BENCH else 'MISSING'}", flush=True)


OOM_MSG = (
    "### Hết bộ nhớ\n\n"
    "Tác vụ này không đủ RAM/VRAM để chạy. Thử:\n"
    "- chuyển Tab 1 sang chế độ **Fast**\n"
    "- giảm `samples`\n"
    "- đóng bớt ứng dụng khác rồi bấm **Reset model**\n\n"
    "Model vẫn còn nguyên, app không bị crash. "
    "Dùng tạm phần **prerendered** hoặc `fallback.html`."
)


# --------------------------------------------------------------------------
# Measured costs (written by scripts/benchmark.py) drive the UI estimates.
# --------------------------------------------------------------------------
BENCH: dict = {}


def load_bench():
    global BENCH
    p = D.PRERENDERED_DIR / "benchmark.json"
    if p.exists():
        try:
            BENCH = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            BENCH = {}
    return BENCH


# Measured on this box: the v* optimisation is 225.9 s (< 5 min), so Fast mode
# does NOT touch v_num_grad_steps. Flip this only on an explicit decision --
# the UI then labels the run "lệch hparams". mom2_adjustment is never touched.
ALLOW_FAST_EDIT_V_STEPS = True   # decided from scripts/compare_v_steps.py
FAST_V_STEPS = 10                # 6 failed the paraphrase probe; 10 passes

AIE_CAVEAT = ("> **Lưu ý:** IE của một prompt đơn lẻ không so trực tiếp được với AIE "
              "trung bình 1000 prompt ở Figure 2 của paper.")

FAST_NOTE = (f"**Fast**: {D.FAST_SAMPLES} mẫu nhiễu thay vì "
             f"{D.FULL_SAMPLES}, nên AIE nhiễu thống kê hơn paper.")


def n_tokens(prompt: str) -> int:
    if MT is None:
        return 7
    try:
        return len(MT.tokenizer.encode(prompt or ""))
    except Exception:  # noqa: BLE001
        return 7


def fmt_dur(s: float) -> str:
    if s < 90:
        return f"~{s:.0f} giây"
    return f"~{s/60:.1f} phút"


def estimate_trace(prompt: str, mode: str) -> float | None:
    """Scale the measured Space Needle cost by token count."""
    key = "trace_hidden_fast_s" if mode == "Fast" else "trace_hidden_full_s"
    base = BENCH.get(key)
    if not base:
        return None
    ref = BENCH.get("prompt_tokens", 7) or 7
    return base * (n_tokens(prompt) / ref)


def trace_estimate_md(prompt: str, kind: str, mode: str) -> str:
    est = estimate_trace(prompt, mode)
    samples = D.FAST_SAMPLES if mode == "Fast" else D.FULL_SAMPLES
    nt = n_tokens(prompt)
    head = (f"**Chế độ: {mode}** · `samples={samples}` · đủ "
            f"{MT.num_layers if MT else 48} layer · {nt} token "
            f"→ {nt * (MT.num_layers if MT else 48)} lần khôi phục")
    if D.DEVICE == "cuda":
        # benchmark.json is CPU-only; quoting it here showed "~3.1 phút" for a
        # run that took 17.5 s on the T4 (CHANGES.md N).
        if mode == "Fast":
            body = (f"\n\n**Ước lượng: {fmt_dur(T4_TRACE_FAST_S * nt / 7)}** "
                    f"trên `cuda` (đo trên Colab T4).")
        else:
            body = (f"\n\n**Chưa đo:** Full trên T4; Fast đo được "
                    f"{fmt_dur(T4_TRACE_FAST_S)} cho 7 token.")
    elif est is None:
        body = "\n\nƯớc lượng thời gian: *chưa có số đo* (chạy `scripts/benchmark.py`)."
    else:
        body = f"\n\n**Ước lượng: {fmt_dur(est)}** trên `{D.DEVICE}` (từ số đo thật)."
    if mode == "Fast":
        body += f"\n\n> {FAST_NOTE} Chỉ chạy được `kind=hidden`."
    else:
        body += "\n\n> Full = đúng paper/notebook."
    return head + body


# --------------------------------------------------------------------------
# Tab 1: Trace
# --------------------------------------------------------------------------
KIND_MAP = {"hidden": None, "mlp": "mlp", "attn": "attn"}


def suggest_subject(prompt: str) -> str:
    try:
        return guess_subject(prompt)
    except Exception:
        return ""


def run_trace(prompt, subject, kind_name, mode, progress=gr.Progress()):
    if MT is None:
        return None, "Model chưa nạp xong."
    prompt = (prompt or "").strip()
    subject = (subject or "").strip() or suggest_subject(prompt)
    if not prompt or not subject:
        return None, "Cần cả prompt và subject."
    if subject not in prompt:
        return None, f"Subject `{subject}` không xuất hiện trong prompt."

    fast = mode == "Fast"
    samples = D.FAST_SAMPLES if fast else D.FULL_SAMPLES
    if fast and kind_name != "hidden":
        kind_name = "hidden"
    kind = KIND_MAP[kind_name]

    est = estimate_trace(prompt, mode)
    progress(0.0, desc=f"causal tracing ({mode}, samples={samples})"
                       + (f", ước lượng {fmt_dur(est)}" if est else ""))
    try:
        res = D.trace(MT, prompt, subject, kind, noise=NOISE_NU,
                      samples=samples, window=10, progress=progress)
    except (MemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if "memory" in msg or "allocate" in msg:
            return None, OOM_MSG
        raise

    progress(0.97, desc="vẽ heatmap ...")
    png = D.CACHE_DIR / "live_trace.png"
    D.save_heatmap(res, png, modelname=D.MODEL_NAME, dpi=150)

    sites = D.top_sites(res, 3)
    el = D.early_late_sites(res)
    te = res["high_score"] - res["low_score"]
    banner = (f"### 🟡 Chế độ **Fast**, `samples={samples}`\n\n> {FAST_NOTE}\n\n"
              if fast else
              f"### 🟢 Chế độ **Full**, `samples={samples}`, đúng paper\n\n")
    est_line = ""
    if est:
        est_line = (f"**Ước lượng trước khi chạy**: `{fmt_dur(est)}` → "
                    f"**thực tế**: `{res['elapsed']:.1f}s`  \n")

    txt = (
        banner
        + f"**Answer**: `{res['answer']}`  \n"
        f"**P[o] clean** = `{res['high_score']:.4f}` "
        + ("(model **nhớ chắc** fact này, nên phép đo có ý nghĩa)  \n"
           if res["high_score"] >= 0.5 else
           "**Lưu ý:** P[o] thấp, model **không chắc** fact này nên IE khó đọc  \n")
        + f"**P\\*[o] corrupted** = `{res['low_score']:.4f}`  \n"
        f"**Total effect** = `{te:.4f}`  \n"
        f"**ν = 3σ** = `{NOISE_NU:.4f}` (σ = `{SIGMA:.4f}`)  \n\n"
        f"**Hai site theo cách phân chia của paper (Fig. 1e–g):**\n\n"
        f"- **Early site** (token cuối của subject): `{el['early']['token']}` "
        f"@ layer **{el['early']['layer']}**, IE = `{el['early']['ie']:.4f}`\n"
        f"- **Late site** (token cuối câu): `{el['late']['token']}` "
        f"@ layer **{el['late']['layer']}**, IE = `{el['late']['ie']:.4f}`\n\n"
        f"**Top-3 (token, layer) theo IE toàn cục:**\n\n"
        + "\n".join(
            f"{i}. token `{s['token']}` (vị trí {s['token_index']}), "
            f"layer **{s['layer']}**, IE = `{s['ie']:.4f}`, P = `{s['p']:.4f}`"
            for i, s in enumerate(sites, 1)
        )
        + "\n\n> Top-3 toàn cục thường bị **late site** chiếm hết: khôi phục "
          "trạng thái ẩn muộn ở token cuối thì gần như hiển nhiên khôi phục "
          "được đáp án. Early site mới là chỗ luận điểm của ROME đứng.\n\n"
        + AIE_CAVEAT
        + f"\n\n{est_line}"
        f"**Thời gian chạy**: `{res['elapsed']:.1f}s` "
        f"(kind={kind_name}, samples={samples}, {D.DEVICE})\n"
    )
    return str(png), txt


# --------------------------------------------------------------------------
# Tab 2: Edit
# --------------------------------------------------------------------------
def _fmt_cell(d: dict, target: str, orig: str) -> str:
    return (
        f"<div style='font-size:12px;line-height:1.45'>"
        f"<div style='margin-bottom:6px'><b>greedy:</b> {html.escape(d['greedy'])}</div>"
        f"<div style='margin-bottom:6px;opacity:.75'><b>sampled:</b> {html.escape(d['sampled'])}</div>"
        f"<div><code>P[{html.escape(target)}]</code> = <b>{d['p_target']:.5f}</b> &nbsp;|&nbsp; "
        f"<code>P[{html.escape(orig)}]</code> = <b>{d['p_orig']:.5f}</b></div>"
        f"</div>"
    )


def _label(s: str) -> str:
    """
    prerendered/edit_eiffel.json was written before the probe labels were
    renamed, so its two loud-fact labels still carry an em-dash. Normalise on
    display; the JSON itself is left alone (it is the numeric source of truth).
    Only labels are touched, never the model's generated text.
    """
    sep = " \u2014 "          # em-dash, written escaped so this file has none
    return s.replace(sep, " (") + ")" if sep in s else s


def rows_to_html(probes: list, target: str) -> str:
    head = (
        "<table style='width:100%;border-collapse:collapse;font-family:system-ui'>"
        "<thead><tr>"
        "<th style='border:1px solid #ccc;padding:8px;width:18%;text-align:left'>Probe</th>"
        "<th style='border:1px solid #ccc;padding:8px;width:41%;text-align:left'>Trước</th>"
        "<th style='border:1px solid #ccc;padding:8px;width:41%;text-align:left'>Sau</th>"
        "</tr></thead><tbody>"
    )
    body = []
    for p in probes:
        ok = p["passes"]
        if ok is None:
            bg = "rgba(130,130,130,.12)"
            mark = "ℹ️ không chấm: fluency degradation, khớp human study của paper"
            want = "không áp tiêu chí"
        elif p["key"] == "loud_fact_cf" and not ok:
            bg = "rgba(205,40,40,.14)"
            mark = ("❌ không đạt: hiện tượng CounterFact+ đã biết, không phải lỗi "
                    "(so với dòng trên, cùng probe nhưng không có câu mồi)")
            want = f"{html.escape(p['orig_answer'])} vẫn thắng"
        else:
            bg = "rgba(34,160,70,.14)" if ok else "rgba(205,40,40,.14)"
            mark = "✅ đạt kỳ vọng" if ok else "❌ không đạt"
            want = ("target thắng" if p["expect"] == "target"
                    else f"{p['orig_answer']} vẫn thắng")
        body.append(
            f"<tr>"
            f"<td style='border:1px solid #ccc;padding:8px;vertical-align:top;background:{bg}'>"
            f"<b>{html.escape(_label(p['label']))}</b>"
            f"<div style='font-size:11px;opacity:.8;margin-top:4px'>"
            f"<code>{html.escape(p['prompt'])}</code></div>"
            f"<div style='font-size:11px;margin-top:6px'>kỳ vọng: {html.escape(want)}<br>{mark}</div>"
            f"</td>"
            f"<td style='border:1px solid #ccc;padding:8px;vertical-align:top'>"
            f"{_fmt_cell(p['before'], target, p['orig_answer'])}</td>"
            f"<td style='border:1px solid #ccc;padding:8px;vertical-align:top;background:{bg}'>"
            f"{_fmt_cell(p['after'], target, p['orig_answer'])}</td>"
            f"</tr>"
        )
    return head + "".join(body) + "</tbody></table>"


def dur_s(s):
    return fmt_dur(s)


# Measured in the Colab T4 rehearsal, 18/10/2026 (CHANGES.md N).
T4_TRACE_FAST_S = 17.5
T4_EDIT_COLD_S, T4_EDIT_WARM_S = 25.7, 18.2


# Measured on an idle CPU box (CHANGES.md M) and on the Colab T4 (CHANGES.md N).
FULL_TOTAL, FULL_EDIT = 497.0, 199.5   # mean of the two Full runs
FAST_TOTAL, FAST_EDIT = 378.9, 83.4


def edit_estimate_md(mode: str) -> str:
    hp = HPARAMS.v_num_grad_steps if HPARAMS else 20
    n = len(D.EIFFEL_PROBES)
    fast = mode == "Fast" and ALLOW_FAST_EDIT_V_STEPS

    # Shown in both modes and on both devices: Fast deviates from the paper's
    # hyperparameters and the UI has to say so before the run, not only after.
    dev = (f"**Lưu ý: Fast lệch hparams**: `{FAST_V_STEPS}/{hp}` bước v\\*, "
           f"`mom2_adjustment` giữ `True`. 6 bước rớt probe paraphrase, 10 bước đạt."
           if fast else
           f"Full theo đúng `hparams/ROME/gpt2-xl.json` (`{hp}` bước v\\*).")

    if D.DEVICE == "cuda":
        t = (f"**Chưa đo:** Fast trên T4. Edit Full đo được "
             f"`{fmt_dur(T4_EDIT_COLD_S)}` lượt đầu, `{fmt_dur(T4_EDIT_WARM_S)}` lượt sau."
             if fast else
             f"**Đo trên T4:** edit Full `{fmt_dur(T4_EDIT_COLD_S)}` lượt đầu "
             f"(gồm tải C stats), `{fmt_dur(T4_EDIT_WARM_S)}` lượt sau.")
        return f"{t}\n\n{dev}\n\n> 2×{n} probe cộng thêm, chưa đo riêng trên T4."

    tot, edit = (FAST_TOTAL, FAST_EDIT) if fast else (FULL_TOTAL, FULL_EDIT)
    return (f"**Ước lượng {fmt_dur(tot)}**: edit `{fmt_dur(edit)}` + 2×{n} probe "
            f"`{fmt_dur(tot - edit)}` trên `{D.DEVICE}` (số đo thật, máy rảnh).\n\n"
            f"{dev}\n\n"
            f"> Lượt đầu trong phiên cộng ~5 s cho `compute_u`; nếu C stats chưa "
            f"có trên đĩa thì cộng ~2 phút để tải 156 MB.")


def run_edit(subject, prompt_tmpl, target, mode, p_rewrite, p_para, p_neigh,
             p_gen, p_loud, o_neigh, o_loud, progress=gr.Progress()):
    global LAST_EDIT
    if MT is None:
        return "Model chưa nạp xong.", "", ""
    subject = (subject or "").strip()
    prompt_tmpl = (prompt_tmpl or "").strip()
    target = (target or "").strip()
    if "{}" not in prompt_tmpl:
        return "Prompt template phải chứa `{}` (chỗ chèn subject).", "", ""
    if not subject or not target:
        return "Cần subject và target.", "", ""

    request = dict(prompt=prompt_tmpl, subject=subject,
                   target_new={"str": target})
    loud = p_loud.strip()
    loud_ans = o_loud.strip()
    probes = [
        dict(key="rewrite", label="Rewrite (efficacy)", prompt=p_rewrite.strip(),
             orig=o_neigh.strip(), expect="target"),
        dict(key="paraphrase", label="Paraphrase (generalization)",
             prompt=p_para.strip(), orig=o_neigh.strip(), expect="target"),
        dict(key="neighborhood", label="Neighborhood (specificity)",
             prompt=p_neigh.strip(), orig=o_neigh.strip(), expect="orig"),
        dict(key="generation", label="Generation (fluency)",
             prompt=p_gen.strip(), orig=o_neigh.strip(), expect="none"),
        dict(key="loud_fact", label="Loud fact (đứng một mình)",
             prompt=loud, orig=loud_ans, expect="orig"),
        # CounterFact+ form, derived from this edit so it works for any request.
        dict(key="loud_fact_cf", label="Loud fact (CounterFact+)",
             prompt=D.composed_prompt(prompt_tmpl, subject, target, loud),
             orig=loud_ans, expect="orig"),
    ]

    v_override = None
    if mode == "Fast" and ALLOW_FAST_EDIT_V_STEPS:
        v_override = FAST_V_STEPS

    orig_weights = None
    try:
        progress(0.05, desc="đo probe TRƯỚC khi sửa ...")
        before = D.eval_probes(MT, probes, target, seed=0)

        progress(0.35, desc="chạy ROME edit (lần đầu phải tải C stats ~156 MB) ...")
        orig_weights, uv, tm = D.apply_edit(MT, request, HPARAMS,
                                            v_steps_override=v_override)
        t_edit = tm["total_s"]

        progress(0.6, desc="đọc đại lượng của phép sửa ...")
        metrics = D.inspect_metrics(MT, request, HPARAMS, uv)

        progress(0.7, desc="đo probe SAU khi sửa ...")
        after = D.eval_probes(MT, probes, target, seed=0)

        progress(0.95, desc="khôi phục trọng số gốc ...")
        restored = D.restore_and_verify(
            MT, orig_weights,
            prompt=probes[0]["prompt"], target=target, orig=probes[0]["orig"],
            p_target_before=before[0]["p_target"], p_orig_before=before[0]["p_orig"])
        orig_weights = None  # restore_and_verify already put them back

        LAST_EDIT = dict(metrics=metrics, request=request, target=target,
                         edit_s=t_edit, restored=restored["all_ok"],
                         restore_detail=restored, timings=tm, mode=mode)

        merged = [
            dict(key=b["key"], label=b["label"], prompt=b["prompt"],
                 orig_answer=b["orig_answer"], expect=b["expect"],
                 before=dict(greedy=b["greedy"], sampled=b["sampled"],
                             p_target=b["p_target"], p_orig=b["p_orig"]),
                 after=dict(greedy=a["greedy"], sampled=a["sampled"],
                            p_target=a["p_target"], p_orig=a["p_orig"]),
                 passes=D.probe_passes(a, target))
            for b, a in zip(before, after)
        ]
        dev_note = ""
        if tm.get("v_steps_overridden"):
            dev_note = (f"  \n**Lưu ý: lệch hparams**: chạy ở chế độ Fast với "
                        f"`v_num_grad_steps={tm['v_num_grad_steps']}` thay vì "
                        f"`{HPARAMS.v_num_grad_steps}`. `mom2_adjustment` giữ nguyên.")
        ok_mark = "✅" if restored["all_ok"] else "**Lưu ý:**"
        status = (
            f"{ok_mark} **Model đã được khôi phục** về trọng số gốc trên "
            f"rewrite prompt: P[{target}] {after[0]['p_target']:.5f} → "
            f"{restored['p_target']:.5f}, "
            f"P[{probes[0]['orig']}] {after[0]['p_orig']:.5f} → "
            f"{restored['p_orig']:.5f}; tensor trọng số trùng bit: "
            f"**{restored['weights_identical']}**.  \n"
            f"Chế độ **{mode}** · edit `{t_edit:.1f}s` = compute_u "
            f"`{tm.get('u_s', 0):.1f}s` + tối ưu v* `{tm.get('v_opt_s', 0):.1f}s` + "
            f"Λ `{tm.get('v_lambda_s', 0):.1f}s` + ghi W `{tm.get('write_s', 0):.2f}s` "
            f"trên `{D.DEVICE}`.{dev_note}  \n"
            f"Sang **Tab 3** để xem toán của phép sửa này."
        )
        return status, rows_to_html(merged, target), _metrics_md(metrics)

    except (MemoryError, RuntimeError) as e:
        msg = str(e).lower()
        if "memory" in msg or "allocate" in msg:
            return OOM_MSG, "", ""
        print(traceback.format_exc(), flush=True)
        return f"### Lỗi\n\n```\n{html.escape(str(e))}\n```", "", ""
    except Exception as e:  # noqa: BLE001
        print(traceback.format_exc(), flush=True)
        return f"### Lỗi\n\n```\n{type(e).__name__}: {html.escape(str(e))}\n```", "", ""
    finally:
        # If anything threw between the edit and the restore above, the model
        # would still be carrying the edit. Leaving it there is how the next
        # click compounds edits on one instance -- the r-ROME collapse path.
        if orig_weights is not None:
            print("[run_edit] restoring weights from the failure path", flush=True)
            D.restore_weights(MT.model, orig_weights)


def load_prerendered_edit():
    path = D.PRERENDERED_DIR / "edit_eiffel.json"
    if not path.exists():
        return ("Chưa có `prerendered/edit_eiffel.json`. "
                "Chạy `python scripts/prerender_edit.py` trước."), "", ""
    d = json.loads(path.read_text())
    status = (
        f"📁 Đang hiển thị kết quả **prerendered**, không dùng GPU/CPU gì cả: "
        f"model `{d['model']}` trên `{d['device']}`, edit mất "
        f"`{d['timings']['edit_s']:.1f}s`, trọng số đã khôi phục: "
        f"`{d['weights_restored']}`."
    )
    return status, rows_to_html(d["probes"], d["target"]), _metrics_md(d["metrics"])


# --------------------------------------------------------------------------
# Tab 3: Inspect
# --------------------------------------------------------------------------
def _metrics_md(m: dict | None) -> str:
    if not m:
        return ("Chưa có edit nào trong phiên này. Chạy một edit ở **Tab 2**, "
                "hoặc bấm **Load ví dụ Eiffel→Rome** để đọc từ `prerendered/`.")

    H, Dd = m["W_shape"]
    dh, dw = m["delta_shape"]
    ratio = m.get("delta_over_W", float("nan"))
    rows = [
        ("Ma trận bị sửa", "W_proj", f"`{H} × {Dd}`",
         "chiếu xuống của MLP (H×D), 47 layer còn lại giữ nguyên"),
        ("Chuẩn vector khoá", "‖k\\*‖", f"`{m['k_star_norm']:.4f}`",
         "đầu vào `c_proj` tại token cuối của subject"),
    ]
    if m.get("mom2_adjustment"):
        rows += [
            ("Mẫu số (dạng trong paper)", "k\\*ᵀC⁻¹k\\*",
             f"`{m['k_Cinv_k']:.4f}`",
             f"C: mô-men bậc hai của khoá, 100k mẫu Wikipedia "
             f"(`{m['C_shape'][0]}×{m['C_shape'][1]}`)"),
            ("", "‖C⁻¹k\\*‖", f"`{m['c_inv_k_norm']:.2f}`",
             "C⁻¹ giữ phép sửa trực giao với các khoá phổ biến khác"),
        ]
    else:
        rows.append(("Mẫu số", "k\\*ᵀu", f"`{m['division_factor']:.4f}`",
                     "`mom2_adjustment=False`, không dùng C⁻¹ (lệch paper)"))
    rows += [
        ("Chuẩn vector giá trị", "‖Λ‖", f"`{m['lambda_norm']:.4f}`",
         "tìm bằng tối ưu gradient trên một vector ẩn, không fine-tune"),
        ("Chuẩn ma trận gốc", "‖W‖_F", f"`{m.get('W_fro', float('nan')):.2f}`",
         "mốc để so Δ"),
        ("Chuẩn cập nhật", "‖Δ‖_F", f"`{m['delta_fro']:.4f}`",
         f"Δ = u ⊗ Λ, `{dh} × {dw}`, cộng thẳng vào W_proj"),
        ("Tỉ lệ biên độ", "‖Δ‖_F / ‖W‖_F", f"`{ratio:.5f}`",
         f"{ratio * 100:.2f}% độ lớn ma trận"),
        ("**Hạng cập nhật**", "**rank(Δ)**", f"**`{m['delta_rank']}`**",
         "tích ngoài của một vector khoá và một vector giá trị"),
        ("Chuẩn hoá", "‖u‖", f"`{m['u_norm']:.4f}`", "`compute_u` chuẩn hoá về 1"),
    ]
    if "delta_fro_analytic" in m:
        gap = abs(m["delta_fro_analytic"] - m["delta_fro"]) / m["delta_fro_analytic"]
        rows.append(("Kiểm tra chéo", "‖u‖·‖Λ‖",
                     f"`{m['delta_fro_analytic']:.6f}`",
                     f"với hạng 1 phải bằng ‖Δ‖_F; lệch `{gap:.1e}`, "
                     f"mức tích luỹ làm tròn fp32"))

    table = ("| Đại lượng | Ký hiệu | Giá trị | Ghi chú |\n|---|---|---|---|\n"
             + "\n".join(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows))

    head = (
        f"**Ŵ = W + Λ(C⁻¹k\\*)ᵀ**, với "
        f"**Λ = (v\\* − W k\\*) / ((C⁻¹k\\*)ᵀ k\\*)**\n\n"
        f"Sửa đúng một ma trận: `{m['weight_name']}`, tức "
        f"**layer 18 theo paper (App. E.5) = `transformer.h.{m['layer']}`, "
        f"0-indexed**.\n\n"
    )

    note = ""
    if m.get("mom2_adjustment"):
        q = m["k_Cinv_k"] / m["c_inv_k_norm"]
        note = (
            f"\n\n> **Hai mẫu số.** `compute_v` chia cho `k*ᵀu` = "
            f"`{m['division_factor']:.4f}`, không phải `k*ᵀC⁻¹k*/‖C⁻¹k*‖` = "
            f"`{q:.4f}` như dạng rút gọn trong paper; chênh lệch đến từ chỗ "
            f"`compute_u` dựng `u` bằng k\\* trung bình trên ~21 context "
            f"template còn `compute_v` chia theo k\\* của một context. "
            f"Gupta et al. 2024 (r-ROME, "
            f"[arXiv:2403.07175](https://arxiv.org/abs/2403.07175)) chỉ ra "
            f"chính bất đối xứng này gây model collapse khi edit tuần tự. "
            f"Với một edit rồi khôi phục trọng số như demo này thì vô hại."
        )
    return head + table + note


# --------------------------------------------------------------------------
# Header actions
# --------------------------------------------------------------------------
def reset_model():
    global LAST_EDIT
    if MT is None:
        return "Model chưa nạp."
    if not LAST_EDIT:
        return "Chưa có edit nào, model đang ở trạng thái gốc."
    LAST_EDIT = {}
    return ("Đã xoá trạng thái edit. Tab 2 luôn tự khôi phục trọng số ngay sau "
            "khi trả kết quả, nên model vốn đã ở trạng thái gốc.")


def header_md():
    nu = f"{NOISE_NU:.4f}" if NOISE_NU is not None else "chưa tính"
    return (
        f"## ROME: Locating and Editing Factual Associations in GPT\n"
        f"`{D.MODEL_NAME}` · thiết bị **{D.DEVICE}** · "
        f"{MT.num_layers if MT else '?'} layer · nạp trong `{LOAD_S:.1f}s` · "
        f"ν = 3σ = `{nu}`"
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
def patch_gradio_bool_schema():
    """
    gradio 4.44.1 / gradio_client 1.3.0 return HTTP 500 on EVERY page load as
    soon as any component's schema has a boolean `additionalProperties`.
    gr.Image's FileData schema has exactly that:
        {"meta": {"additionalProperties": true, "type": "object", ...}}

    JSON Schema allows additionalProperties to be a boolean, but
    gradio_client.get_type assumes a dict and runs `"const" in schema`, which
    raises TypeError on a bool. gradio/routes.py main() calls api_info()
    unconditionally, so show_api=False does NOT avoid it -- the whole app is
    unreachable.

    Only the generated API-docs type string is affected; nothing functional.
    A bool there means "any/no additional properties", so "Any" is the honest
    rendering.
    """
    from gradio_client import utils as gc

    orig_get_type = gc.get_type
    orig_to_python = gc._json_schema_to_python_type

    def get_type(schema):
        return "Any" if isinstance(schema, bool) else orig_get_type(schema)

    def to_python(schema, defs):
        return "Any" if isinstance(schema, bool) else orig_to_python(schema, defs)

    gc.get_type = get_type
    gc._json_schema_to_python_type = to_python


def build_ui():
    with gr.Blocks(title="ROME demo", theme=gr.themes.Soft()) as demo:
        with gr.Row():
            gr.Markdown(header_md())
        with gr.Row():
            reset_btn = gr.Button("Reset model", scale=0, variant="secondary")
            reset_out = gr.Markdown("")
        reset_btn.click(reset_model, outputs=reset_out)

        with gr.Tabs():
            # ---------------- Tab 1 ----------------
            with gr.Tab("1. Trace"):
                gr.Markdown(
                    "Làm hỏng embedding của subject bằng nhiễu Gaussian ν = 3σ rồi "
                    "khôi phục từng trạng thái ẩn, ô sáng là indirect effect cao."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        t_prompt = gr.Textbox(
                            label="Prompt",
                            value="The Space Needle is in downtown")
                        with gr.Row():
                            t_subject = gr.Textbox(
                                label="Subject (phải là substring của prompt)",
                                value="The Space Needle", scale=3)
                            t_guess = gr.Button("Gợi ý subject", scale=1)
                        with gr.Row():
                            t_mode = gr.Radio(
                                ["Fast", "Full"], value="Fast", label="Chế độ",
                                info=f"Fast: {D.FAST_SAMPLES} mẫu nhiễu, chỉ "
                                     f"kind=hidden. Full: {D.FULL_SAMPLES} mẫu, "
                                     f"đúng paper.")
                            t_kind = gr.Radio(["hidden", "mlp", "attn"],
                                              value="hidden", label="Kind",
                                              interactive=False)
                        t_est = gr.Markdown()
                        t_run = gr.Button("Run", variant="primary")
                    with gr.Column(scale=2):
                        t_info = gr.Markdown()
                t_img = gr.Image(label="Causal trace heatmap", type="filepath")

                t_guess.click(suggest_subject, inputs=t_prompt, outputs=t_subject)

                def _on_mode(mode, prompt, kind):
                    # Fast is hidden-only; Full re-enables the kind picker.
                    if mode == "Fast":
                        return (gr.update(value="hidden", interactive=False),
                                trace_estimate_md(prompt, "hidden", mode))
                    return (gr.update(interactive=True),
                            trace_estimate_md(prompt, kind, mode))

                t_mode.change(_on_mode, inputs=[t_mode, t_prompt, t_kind],
                              outputs=[t_kind, t_est])
                t_prompt.change(trace_estimate_md,
                                inputs=[t_prompt, t_kind, t_mode], outputs=t_est)
                t_kind.change(trace_estimate_md,
                              inputs=[t_prompt, t_kind, t_mode], outputs=t_est)
                demo.load(trace_estimate_md,
                          inputs=[t_prompt, t_kind, t_mode], outputs=t_est)

                t_run.click(run_trace,
                            inputs=[t_prompt, t_subject, t_kind, t_mode],
                            outputs=[t_img, t_info])

                gr.Examples(
                    examples=[
                        ["The Space Needle is in downtown", "The Space Needle", "hidden", "Fast"],
                        ["The Eiffel Tower is located in the city of", "The Eiffel Tower", "mlp", "Full"],
                        ["Michael Jordan plays the sport of", "Michael Jordan", "attn", "Full"],
                    ],
                    inputs=[t_prompt, t_subject, t_kind, t_mode],
                    label="Ví dụ (khớp với prerendered/)",
                )

            # ---------------- Tab 2 ----------------
            with gr.Tab("2. Edit"):
                gr.Markdown(
                    f"Một phép sửa ROME duy nhất, đo trên {len(D.EIFFEL_PROBES)} "
                    "probe. Trọng số gốc **luôn được khôi phục** ngay sau khi trả "
                    "kết quả, không bao giờ sửa nhiều fact liên tiếp trên cùng một "
                    "model instance."
                )
                with gr.Row():
                    e_subject = gr.Textbox(label="Subject", value="The Eiffel Tower")
                    e_tmpl = gr.Textbox(label="Prompt template (có {})",
                                        value="{} is located in the city of")
                    e_target = gr.Textbox(label="Target mới", value="Rome")
                with gr.Row():
                    e_mode = gr.Radio(["Fast", "Full"], value="Full",
                                      label="Chế độ",
                                      info="Full: đúng hparams. Fast: giảm "
                                           "v_num_grad_steps (lệch hparams).")
                    e_est = gr.Markdown()
                with gr.Accordion(f"{len(D.EIFFEL_PROBES)} probe (sửa được)", open=False):
                    e_rewrite = gr.Textbox(label="1. Rewrite",
                                           value="The Eiffel Tower is located in the city of")
                    e_para = gr.Textbox(label="2. Paraphrase",
                                        value="Where is the Eiffel Tower? It is in")
                    with gr.Row():
                        e_neigh = gr.Textbox(label="3. Neighborhood",
                                             value="The Louvre is located in the city of")
                        e_neigh_o = gr.Textbox(label="đáp án gốc", value="Paris")
                    e_gen = gr.Textbox(label="4. Generation",
                                       value="The Eiffel Tower is right across from")
                    with gr.Row():
                        e_loud = gr.Textbox(
                            label="5+6. Loud fact (dạng CounterFact+ tự ghép)",
                            value=D.LOUD_FACT_PROMPT)
                        e_loud_o = gr.Textbox(label="đáp án gốc",
                                              value=D.LOUD_FACT_ANSWER)
                with gr.Row():
                    e_run = gr.Button("Chạy edit", variant="primary")
                    e_pre = gr.Button("Load ví dụ Eiffel→Rome (từ prerendered/)")
                e_status = gr.Markdown()
                e_table = gr.HTML()

            # ---------------- Tab 3 ----------------
            with gr.Tab("3. Inspect"):
                gr.Markdown("### Toán của phép sửa vừa chạy")
                i_md = gr.Markdown(_metrics_md(None))
                i_refresh = gr.Button("Làm mới từ edit gần nhất")
                i_refresh.click(lambda: _metrics_md(LAST_EDIT.get("metrics")),
                                outputs=i_md)

        e_run.click(run_edit,
                    inputs=[e_subject, e_tmpl, e_target, e_mode, e_rewrite,
                            e_para, e_neigh, e_gen, e_loud, e_neigh_o, e_loud_o],
                    outputs=[e_status, e_table, i_md])
        e_pre.click(load_prerendered_edit, outputs=[e_status, e_table, i_md])
        e_mode.change(edit_estimate_md, inputs=e_mode, outputs=e_est)
        demo.load(edit_estimate_md, inputs=e_mode, outputs=e_est)

    return demo


if __name__ == "__main__":
    patch_gradio_bool_schema()
    boot()
    ui = build_ui()
    # gradio >= 5 refuses to serve files outside cwd/tmp unless allowed; the
    # demo_utils chdir makes cwd ./rome, so the live heatmap in .demo_cache
    # was rejected on Colab (InvalidPathError). Harmless on gradio 4.44.
    ui.launch(server_name=SERVER_NAME, server_port=SERVER_PORT, share=SHARE,
              allowed_paths=[str(D.CACHE_DIR), str(D.PRERENDERED_DIR)])
