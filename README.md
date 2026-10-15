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

- `prerendered/` — 9 heatmap PNG 200 dpi + các JSON:
  - `trace_summary.json` — P[o]/P\*[o], early/late site, thời gian cho cả 9 lần trace
  - `edit_eiffel.json` — edit Eiffel→Rome, trước/sau trên 6 probe, metrics Tab 3
  - `benchmark.json` — số đo thời gian, nguồn cho ước lượng trên UI
  - `v_steps_comparison.json` — Full vs Fast-6 vs Fast-10
  - `restore_verification.json` — 4 điều kiện khôi phục trọng số
  - `loud_fact_selection.json`, `loud_fact_runnerup.json` — chọn probe loud-fact
  - `prerender.log` — log có timestamp của toàn bộ quá trình
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
| Edit | `v_num_grad_steps` = **10** (**lệch hparams**, UI báo rõ) | **20** — đúng `hparams/ROME/gpt2-xl.json` |
| `mom2_adjustment` | `True` | `True` |

> **Fast: ν và AIE có nhiễu thống kê lớn hơn paper.** Với 3 mẫu, sai số chuẩn của
> AIE lớn hơn khoảng √(10/3) ≈ 1.8 lần so với Full. Dùng Fast để *thấy hình dạng*,
> dùng Full/prerendered để *trích số*.

Mức 10 bước được chọn bằng đo, không phải đoán (`scripts/compare_v_steps.py`),
với tiêu chí: rewrite **và** paraphrase đều phải thắng.

| | Full (20) | Fast-6 | Fast-10 |
|---|---|---|---|
| loss cuối v\* | 0.1030 | 1.5820 | 0.1320 |
| rewrite P[Rome] | 0.99991 | 0.20557 | 0.99976 |
| paraphrase P[Rome] / P[Paris] | 0.82932 / 0.00164 | **0.01345 / 0.05688** | 0.60579 / 0.00638 |
| ‖Δ‖_F | 8.4990 | 5.7683 | 8.4990 |
| thời gian | 225.4 s | 62.9 s | 100.2 s |
| chấp nhận | — | ❌ rớt paraphrase | ✅ |

`‖Δ‖_F` của Fast-10 **trùng khít** Full vì `clamp_norm_factor=4` chặn vector ẩn
ở `4 × ‖target_init‖ = 100.164` và tối ưu chạm trần từ khoảng bước 10. Từ 10 lên
20 bước không đổi biên độ, chỉ **xoay hướng trong quả cầu L2** — và hướng mới là
thứ nâng paraphrase từ 0.606 lên 0.829.

**Prerendered luôn chạy Full.** UI hiện ước lượng thời gian trước khi chạy, lấy
từ số đo thật ở mục 5.

---

## 5. Thời gian đo được

<!-- BENCH_TABLE -->
Đo trên máy này: `gpt2-xl` fp32 trên **cpu**, Intel Core Ultra 7 165H, torch `2.8.0+cpu`, 16 thread.

| Thao tác | Thời gian | Ghi chú |
|---|---|---|
| Nạp model (fp32, từ cache đĩa) | **1.5 s** |  |
| 1 forward pass, batch 1, 7 token | **292 ms** | trung vị trên 7 lần, sau 1 lần làm nóng |
| Heatmap `hidden` **Fast** (3 mẫu) | **3.1 phút** | 7×48 = 336 lần khôi phục, batch 4 |
| Heatmap `hidden` **Full** (10 mẫu) | **7.3 phút** | 7×48 = 336 lần khôi phục, batch 11 |
| **ROME edit, tổng** | **5.8 phút** | `v_num_grad_steps=20` |
| ├ `compute_u` (k\*, C⁻¹) | **2.0 phút** | lần đầu gồm tải C stats 156 MB + nghịch đảo 6400² |
| ├ tối ưu v\* | **3.8 phút** | vòng lặp gradient |
| ├ tính Λ | **464 ms** | giải hệ tuyến tính + 2 forward |
| └ ghi W (cập nhật hạng 1) | **4.5 s** | cộng Δ vào W_proj |
| `inspect_metrics` (gồm `matrix_rank`) | **774 ms** | SVD trên Δ 6400×1600 |
| 5 probe (greedy + sampled + 2 xác suất) | **58.1 s** | Tab 2 chạy 2 lần: trước và sau |
| ROME edit **Fast** | **73.7 s** | `v_num_grad_steps=6` — **lệch hparams** |

**Tab 2 trong thực tế** = edit + 2×(5 probe) ≈ **7.8 phút** cho lượt đầu (đã có C stats trong cache thì nhanh hơn).


### Fast có đáng tin không?

Cùng prompt `The Space Needle is in downtown`, cùng ν:

| | Fast (3 mẫu) | Full (10 mẫu) | Lệch |
|---|---|---|---|
| P[o] clean | 0.9764 | 0.9764 | 0.0000 |
| P*[o] corrupted | 0.0214 | 0.0219 | **0.0005** |
| Site mạnh nhất | ` downtown` L46, IE 0.9552 | ` downtown` L47, IE 0.9545 | — |
| Thời gian | 3.1 phút | 7.3 phút | **2.37×** |

Hai chế độ cho cùng một kết luận; Fast chỉ nhiễu hơn ở chữ số thứ ba. Dùng Fast để bấm live, dùng Full/prerendered khi cần trích số vào slide.
<!-- /BENCH_TABLE -->

---

## 6. Kiểm chứng khôi phục trọng số

Tab 2 khôi phục trọng số ngay sau khi trả kết quả. Đây là **điều kiện bắt buộc**
trước khi để người khác bấm, nên nó được kiểm bằng script riêng
(`scripts/verify_restore.py`), không chỉ tin vào code:

| Kiểm tra | Kết quả |
|---|---|
| `P[Rome]` trở về giá trị trước edit | ✅ \|Δ\| = 0.000e+00 |
| `P[Paris]` trở về giá trị trước edit | ✅ \|Δ\| = 0.000e+00 |
| Tensor trọng số **trùng bit** | ✅ max\|Δw\| = 0.000e+00 |
| Edit thật sự có đổi W (kiểm tra ngược) | ✅ max\|Δw\| = 1.010e-01 |

```
TRƯỚC      P[Rome]=0.00005692   P[Paris]=0.93407643
ĐÃ SỬA     P[Rome]=0.99993801   P[Paris]=0.00000011
KHÔI PHỤC  P[Rome]=0.00005692   P[Paris]=0.93407643
```

Khôi phục **chính xác tuyệt đối**, không chỉ trong ngưỡng 1e-6. `P[Paris]`
là phép thử nhạy ở đây: nó ≈ 0.93 chứ không ≈ 0 như `P[Rome]`, nên một
phép khôi phục cẩu thả sẽ lộ ra ngay ở cột đó.

---

## 7. Quan sát về cài đặt

### 7.1 Đánh số layer
Toàn bộ UI, README và JSON dùng **một** quy ước:

> **layer 18 theo paper (App. E.5) = `transformer.h.17`, 0-indexed**

`hparams/ROME/gpt2-xl.json` ghi `"layers": [17]` và **không bị sửa**.

### 7.2 Mẫu số k\*ᵀC⁻¹k\* không khớp hệ số chia thực tế — và đó là bug r-ROME

Paper viết mẫu số của cập nhật hạng 1 là k\*ᵀC⁻¹k\*. Đo trên cài đặt:

| | Giá trị |
|---|---|
| `k*ᵀC⁻¹k* / ‖C⁻¹k*‖` | **12.8973** |
| `k*ᵀu` — hệ số `compute_v` **thực sự** chia | **11.8944** |
| Lệch | ~8% |

Nguyên nhân: `compute_u` dựng `u` từ k\* **trung bình trên ~21 context template**,
còn `compute_v` chia cho k\* của **một** context.

Đây **không phải lỗi làm tròn**. Chính bất đối xứng này được Gupta et al. 2024
(r-ROME, [arXiv:2403.07175](https://arxiv.org/abs/2403.07175)) xác định là
nguyên nhân **model collapse khi edit tuần tự**. Với *một* edit rồi khôi phục
trọng số — đúng cách demo này làm — lệch 8% là vô hại. Tab 3 hiện **cả hai**
mẫu số kèm giải thích.

Đây cũng là lý do `demo_utils.apply_edit()` ghi rõ trong docstring: không bao
giờ chạy hai edit liên tiếp trên cùng một model instance mà không khôi phục.

### 7.3 Context template được sample — đã seed
`rome_main.get_context_templates()` sinh ~21 template bằng `generate_fast(..., top_k=5)`,
tức **sampling không seed**, cache theo tiến trình. k\* lấy trung bình trên chúng,
nên mỗi lần khởi động lại app cho một Δ khác (đo được: paraphrase P[target]
0.829 vs 0.870 giữa hai tiến trình chạy cùng mã). `demo_utils.apply_edit(seed=0)`
gọi `torch.manual_seed` trước khi sửa để demo tái lập được.

---

## 8. Ba cảnh báo cho người trình bày

### 9.1 Đừng bấm Run ở chế độ Full trong lúc đang nói
Số đo trên máy này: heatmap `hidden` Full **7.3 phút**, `mlp` **15.3 phút**,
`attn` **15.6 phút**. Một ROME edit Full + 2×6 probe ≈ **8 phút**.

Trong buổi nói:
- Tab 1 → dùng **Fast** (3.1 phút) hoặc chiếu thẳng ảnh trong `prerendered/`.
  Fast đã được đối chiếu với Full: P\*[o] lệch 0.0005, cùng kết luận (mục 4).
- Tab 2 → dùng nút **"Load ví dụ Eiffel→Rome"**, đọc từ `prerendered/`, hiện
  tức thì, không tính gì.
- Chỉ bấm Run thật khi có người hỏi "chạy thật được không", và nói trước là
  mất mấy phút.

Trên Colab T4 thì khác hẳn — đó là lý do Colab là đường chạy chính (mục 1).

### 9.2 Nguy cơ OOM là thật trên máy CPU này
gpt2-xl fp32 chiếm **~6.3 GB RSS**, máy có 30 GB nhưng thường chỉ còn ~10 GB
trống. Trước buổi nói: **đóng VM qemu và bớt tab trình duyệt**. Đã từng đo lúc
máy chỉ còn 5.4 GB trống — khi đó model không nạp nổi.

Không bao giờ chạy prerender song song với app: hai bản gpt2-xl là ~12.5 GB.
Mọi script trong `scripts/` được xếp **tuần tự** đúng vì lý do này.

App bắt OOM và hiện thông báo thân thiện thay vì sập, nhưng cách chắc nhất vẫn
là `fallback.html` — mở bằng trình duyệt, không cần server, không cần RAM.

### 9.3 Ba chỗ dễ bị hỏi mà số liệu trông "sai"

**a. Top-3 theo IE toàn cục luôn chỉ vào late site.** Khôi phục trạng thái ẩn
muộn ở token cuối thì gần như hiển nhiên khôi phục được đáp án, nên nó thắng về
trị tuyệt đối. **Early site mới là chỗ luận điểm ROME đứng.** UI hiện cả hai,
hãy chỉ vào dòng *early site*.

**b. Probe `generation` không được chấm.** Cả P[target] lẫn P[gốc] đều ≈ 0 vì
prompt không hỏi tên thành phố. Sau edit, mạch văn **bị lặp** ("It's just around
the corner." ba lần). Đây là *fluency degradation*, khớp human study trong paper
— trình bày như hạn chế, đừng né.

**c. Loud fact có hai dòng, một xanh một đỏ — và đó là chủ ý.** Đứng một mình,
specificity giữ được (P[Agra] 0.9986 → 0.9943). Nhưng khi **chính sự thật vừa
sửa đứng trước trong ngữ cảnh**, P[Agra] 0.9964 → **0.0000** và P[Rome] →
**0.99997**; model trả lời "Rome" cho Taj Mahal rồi trượt sang một chuỗi công
trình La Mã. Đây là hiện tượng CounterFact+ đã biết, không phải lỗi lần chạy
này. Nếu bị hỏi "vậy ROME có specificity không?" — câu trả lời đúng là
**"có, nhưng có điều kiện"**, và bảng này chứng minh điều đó.

---

## 9. Cấu trúc

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
