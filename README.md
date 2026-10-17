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
!git clone -q https://github.com/daniele15/demo-XAI-ROME.git demo
!git clone -q --depth 1 https://github.com/kmeng01/rome demo/rome
!pip -q install -r demo/requirements-cuda.txt
%cd demo
!SHARE=1 python app.py
```

Cell in ra một link `*.gradio.live` — mở link đó là dùng được.

Trên T4, `DEVICE` tự thành `cuda` và device shim **tự tắt**, nên không phải sửa gì.
`requirements-cuda.txt` cố ý **không** ghim `torch`/`numpy`: runtime của Colab/Kaggle
đã có sẵn torch bản CUDA, ép pin của máy CPU lên đó chỉ gây restart runtime vô ích.

> ⚠️ **Repo đang ở chế độ private.** Cell trên dùng `git clone` qua HTTPS nên chỉ
> chạy được khi repo công khai. Hai cách:
> - **Đổi repo sang public** (Settings → General → Change visibility) — đơn giản nhất,
>   và repo này không chứa gì nhạy cảm (`venv/`, `rome/`, model weights đều bị ignore).
> - **Giữ private**: tạo một Personal Access Token chỉ có quyền `repo`, lưu vào
>   Colab Secrets, rồi clone bằng
>   `https://<TOKEN>@github.com/daniele15/demo-XAI-ROME.git`.
>   Đừng dán token thẳng vào cell.

---

## 2. Chạy trên máy CPU này

```bash
./run.sh
```

→ http://localhost:7860. Có link public:

```bash
SHARE=1 ./run.sh
```

**Đã test thật:** link `*.gradio.live` được tạo trong **8.6 giây** kể từ lúc gõ
lệnh (gồm cả nạp model 1.8 s), trả HTTP 200 từ ngoài với 44 KB trong 2.0 s.
Lần đầu dùng `share=1`, Gradio tự tải binary `frpc_linux_amd64` (~10.8 MB) từ
CDN của nó. **Link sống 72 giờ** và ai có link đều vào được — đừng để chạy nền
sau buổi nói.

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
  - `screenshots/` — ảnh chụp ba tab của app đang chạy
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


### Chi phí một ROME edit — ba cột

| Giai đoạn | Lạnh (CPU, phải tải C) | Ấm (CPU, C trong cache) | GPU T4 (ước tính) |
|---|---|---|---|
| `compute_u` (k\*, C⁻¹) | 2.0 phút | 9.2 s | — |
| tối ưu v\* (20 bước) | 3.8 phút | 3.5 phút | — |
| tính Λ | 464 ms | 414 ms | — |
| ghi W | 4.5 s | 6.7 s | — |
| **Tổng** | **5.8 phút** | **3.8 phút** | **~2 s** |

- **Lạnh** = lần đầu trong phiên: `compute_u` phải tải C stats 156 MB từ `rome.baulab.info` rồi nghịch đảo ma trận 6400².
- **Ấm** = các lần sau: C⁻¹ đã nằm trong `inv_mom2_cache` của `compute_u`.
- **GPU T4**: ⚠️ **ước tính từ paper, CHƯA đo trên máy này** (máy này không có GPU NVIDIA). Dùng làm mốc kỳ vọng khi chạy Colab, không phải số đo.
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

**c. Ba prompt cho ba cường độ khác nhau — và đó là thông tin, không phải lỗi.**

| Prompt | P[o] | P\*[o] | Total effect | Early site |
|---|---|---|---|---|
| The Space Needle… | 0.9764 | 0.0219 | **0.9545** | `le` @ L15, IE 0.943 |
| The Eiffel Tower… | 0.9341 | 0.0047 | **0.9294** | `el` @ L15, IE 0.289 |
| Michael Jordan… | 0.7896 | **0.1271** | **0.6625** | ` Jordan` @ L4, IE 0.734 |

Heatmap của Jordan nhạt hơn hẳn. Lý do nằm ở **P\*[o] = 0.1271**: ngay cả khi
subject đã bị làm hỏng, model vẫn đoán được `basketball` ở 12.7% chỉ nhờ
`"plays the sport of"`. Còn ít thứ để khôi phục thì IE phải nhỏ.

Nếu bị hỏi *"sao ảnh này nhạt thế"* — câu trả lời là **P\*[o]**, không phải model
hay code. Nguyên tắc chung: causal tracing chỉ sắc nét khi relation **không** tự
gợi ra đáp án.

Cũng lưu ý early site của Jordan ở **L4** chứ không phải L15 như hai prompt kia.
Figure 2 của paper (trung bình 1000 prompt) có đỉnh early site quanh layer 5–10,
nên L4 còn sát trung bình hơn L15.

**d. Loud fact có hai dòng, một xanh một đỏ — và đó là chủ ý.** Đứng một mình,
specificity giữ được (P[Agra] 0.9986 → 0.9942). Nhưng khi **chính sự thật vừa
sửa đứng trước trong ngữ cảnh**, P[Agra] 0.9964 → **0.0000** và P[Rome] →
**0.99997**; model trả lời "Rome" cho Taj Mahal rồi trượt sang một chuỗi công
trình La Mã. Đây là hiện tượng CounterFact+ đã biết, không phải lỗi lần chạy
này. Nếu bị hỏi "vậy ROME có specificity không?" — câu trả lời đúng là
**"có, nhưng có điều kiện"**, và bảng này chứng minh điều đó.

---

## 9. Đối chiếu với paper

### 9.1 Causal tracing (Figure 1e–g, Figure 2)

| Điều paper khẳng định | Kết quả ở đây | Khớp? |
|---|---|---|
| Có **hai** site: early ở token cuối subject, late ở token cuối | Thấy rõ trên cả 3 prompt, cả 3 kind | ✅ |
| **MLP** mang hiệu ứng ở **early site** | Đỉnh ở **L16 trên cả 3 prompt**; IE 0.32–0.75 | ✅ |
| **Attention** mang hiệu ứng ở **late site** | L27–L32; IE 0.35–0.75 | ✅ |
| MLP **không** quan trọng ở late site | IE 0.014–0.132 — gần 0 | ✅ |
| Early-site AIE ≈ 0.2–0.35 (Fig. 2, trung bình 1000 prompt) | Eiffel 0.289 khớp; Space Needle 0.943 và Jordan 0.734 **cao hơn nhiều** | ⚠️ |
| Vị trí đỉnh early site | L15–16 (Space Needle, Eiffel) nhưng **L4** ở Jordan (`hidden`) | ⚠️ |

**Hai chỗ không khớp hoàn toàn, và lý do:**

- **IE cao hơn Figure 2.** Figure 2 là **trung bình trên 1000 prompt** của
  CounterFact. Ba prompt ở đây được paper chọn làm minh hoạ vì chúng mạnh. So
  một mẫu với một trung bình là so sai đơn vị — UI có cảnh báo đúng dòng này.
- **Đỉnh lệch ở Jordan.** `hidden` đỉnh ở L4 nhưng `mlp` vẫn ở L16. Hai thứ đo
  hai đại lượng khác nhau: `hidden` là trạng thái ẩn tổng hợp (đã gồm thông tin
  attention mang tới), `mlp` cô lập riêng khối MLP. Đóng góp MLP vẫn ở L16 đúng
  như hai prompt kia.

### 9.2 ROME edit (Figure 25, Table 1)

| Điều paper khẳng định | Kết quả ở đây | Khớp? |
|---|---|---|
| Cập nhật là **hạng 1** | `matrix_rank(Δ) = 1` | ✅ |
| Chỉ sửa **một** ma trận MLP | Chỉ `transformer.h.17.mlp.c_proj.weight` | ✅ |
| **Efficacy**: rewrite thắng | P[Rome] 0.00006 → **0.99994** | ✅ |
| **Generalization**: paraphrase thắng | P[Rome] 0.00110 → **0.87668** | ✅ |
| **Specificity**: neighborhood giữ nguyên | P[Paris] 0.82326 → **0.75906** | ✅ |
| Sửa *sự thật*, không phải học thuộc chuỗi | paraphrase (câu hỏi khác hẳn) cũng đổi | ✅ |
| **Fluency degradation** (human study) | generation bị lặp: *"It's just around the corner."* ×3 | ✅ |
| Specificity có giới hạn | đứng một mình P[Agra] 0.99864 → 0.99425; CounterFact+ 0.99644 → **0.00000** | ✅ (đúng là có giới hạn) |

### 9.3 Khác biệt về thiết lập

| | Paper | Ở đây |
|---|---|---|
| Model | GPT-2 XL, GPT-J | GPT-2 XL (đúng paper) |
| Thiết bị | GPU | **CPU, fp32** — xem CHANGES.md mục A1 |
| Số prompt | 1000 (CounterFact) | **3** cho tracing, **1** edit |
| Số mẫu nhiễu | 10 | 10 (Full), 3 (Fast) |
| ν | 3σ trên known_1000 | 3σ trên known_1000 = **0.1346** (khớp ~0.13 của notebook) |

Khác biệt lớn nhất **không phải model mà là cỡ mẫu**: 3 prompt so với 1000. Mọi
con số ở đây là minh hoạ cơ chế, không phải thống kê để kết luận.

---

## 10. Cấu trúc

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
