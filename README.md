# ROME demo — Locating and Editing Factual Associations in GPT

Web demo Gradio cho ROME (Meng et al., NeurIPS 2022) dùng trong seminar sau đại học.
Ba tab: **Trace** (causal tracing), **Edit** (một phép sửa ROME, trước/sau trên 5 probe),
**Inspect** (toán của phép sửa đó).

Source gốc `kmeng01/rome` được clone nguyên vẹn vào `./rome` và **không bị sửa một dòng nào**.
Mọi thích nghi nằm ở [`demo_utils.py`](demo_utils.py) — chi tiết trong [CHANGES.md](CHANGES.md).

---

## Đường chạy cho buổi seminar

| | Dùng khi nào | Cần gì |
|---|---|---|
| **1. Colab/Kaggle T4** ⭐ | mặc định, chạy live | mạng + tài khoản Google/Kaggle |
| **2. Máy CPU này** | dự phòng, hoặc chạy prerender | không cần gì |
| **3. `fallback.html`** | GPU *và* mạng đều hỏng | chỉ cần trình duyệt |

Máy CPU này **không có GPU NVIDIA** (chỉ Intel Arc iGPU), nên nó phục vụ việc
prerender và làm bản dự phòng. Đường chạy live là Colab/Kaggle.

---

## 1. Chạy trên Kaggle / Colab T4

Bật GPU trước (Colab: *Runtime → Change runtime type → T4 GPU*;
Kaggle: *Settings → Accelerator → GPU T4* **và** *Internet → On*).

```python
!git clone -q https://github.com/<YOUR_USER>/<YOUR_REPO>.git demo
!git clone -q --depth 1 https://github.com/kmeng01/rome demo/rome
!pip -q install -r demo/requirements-cuda.txt
%cd demo
!SHARE=1 python app.py
```

Cell in ra một link `*.gradio.live` — mở link đó là dùng được.

Trên T4, `DEVICE` tự thành `cuda` và device shim **tự tắt**, nên không phải sửa gì.
`requirements-cuda.txt` cố ý **không** ghim `torch`/`numpy`: runtime của Colab/Kaggle
đã có sẵn torch bản CUDA, ép pin của máy CPU lên đó chỉ gây restart runtime vô ích.

> ⚠️ **Cần làm trước:** thư mục này hiện **chưa phải git repo** và chưa được push đi đâu,
> nên `<YOUR_USER>/<YOUR_REPO>` ở trên còn là chỗ trống. Đẩy lên trước:
> ```bash
> git init && git add -A && git commit -m "ROME demo"
> git remote add origin https://github.com/<YOUR_USER>/<YOUR_REPO>.git
> git push -u origin main
> ```
> `.gitignore` đã loại `venv/`, `rome/`, `.demo_cache/` nên repo vẫn nhẹ.
> Nếu không muốn push: nén thư mục rồi upload thủ công vào Colab, bỏ dòng `git clone` đầu.

---

## 2. Chạy trên máy CPU này

```bash
./run.sh
```

→ http://localhost:7860. Có link public:

```bash
SHARE=1 ./run.sh
```

Lần đầu chạy sẽ tải `gpt2-xl` (~6.4 GB) từ HuggingFace và, khi bấm edit,
C stats (~156 MB) từ `rome.baulab.info`.

Dựng lại môi trường từ đầu:

```bash
python3 -m venv venv
./venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
git clone --depth 1 https://github.com/kmeng01/rome ./rome
```

---

## 3. Bản dự phòng offline

- `prerendered/` — 9 heatmap PNG 200 dpi + `trace_summary.json` + `edit_eiffel.json` + `benchmark.json`
- `fallback.html` — trang tĩnh **tự chứa** (ảnh nhúng base64), mở trực tiếp bằng trình duyệt,
  không cần server/mạng/GPU
- Trong app, nút **“Load ví dụ Eiffel→Rome”** ở Tab 2 đọc thẳng từ `prerendered/`,
  hiện ngay không cần tính gì

Chạy lại prerender (nền, có thể để qua đêm, **resume được** — bỏ qua phần đã xong):

```bash
nohup ./scripts/run_prerender.sh &
tail -f prerendered/prerender.log
```

---

## 4. Fast vs Full

Khác biệt duy nhất là **số mẫu nhiễu**. Cả hai đều quét đủ mọi token và đủ 48 layer,
nên hình dạng heatmap giống hệt nhau và **không có giá trị nào bị nội suy**.

| | Fast | Full |
|---|---|---|
| Mẫu nhiễu | 3 | 10 (đúng paper) |
| Layer | đủ 48 | đủ 48 |
| `kind` | chỉ `hidden` | `hidden` / `mlp` / `attn` |
| Edit | giảm `v_num_grad_steps` nếu edit > 5 phút (**lệch hparams**, UI báo rõ) | đúng `hparams/ROME/gpt2-xl.json` |
| `mom2_adjustment` | `True` | `True` |

> **Fast: ν và AIE có nhiễu thống kê lớn hơn paper.** Với 3 mẫu, sai số chuẩn của
> AIE lớn hơn khoảng √(10/3) ≈ 1.8 lần so với Full. Dùng Fast để *thấy hình dạng*,
> dùng Full/prerendered để *trích số*.

UI hiện ước lượng thời gian trước khi chạy, lấy từ số đo thật ở mục 5.

---

## 5. Thời gian đo được

<!-- BENCH_TABLE -->
*(chưa có — chạy `./venv/bin/python scripts/benchmark.py`)*
<!-- /BENCH_TABLE -->

---

## 6. Cấu trúc

```
app.py                  Gradio app, 3 tab
demo_utils.py           shim + wrapper quanh ./rome (không sửa ./rome)
requirements.txt        pin cho máy CPU này
requirements-cuda.txt   pin cho Colab/Kaggle T4
run.sh                  kích hoạt venv + chạy app
scripts/
  smoke_test.py         kiểm tra model nạp được và trả lời đúng
  benchmark.py          đo thời gian thật -> prerendered/benchmark.json
  prerender_trace.py    9 heatmap Full mode (resume được)
  prerender_edit.py     edit Eiffel->Rome, trước/sau 5 probe
  make_fallback.py      dựng fallback.html tự chứa
  run_prerender.sh      chạy cả 3 ở nền, log có timestamp
prerendered/            ảnh + JSON + prerender.log
fallback.html           trang tĩnh dự phòng
rome/                   upstream kmeng01/rome, NGUYÊN VẸN
CHANGES.md              mọi lỗi, workaround, khác biệt so với paper
```
