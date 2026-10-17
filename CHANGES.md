# CHANGES.md

Nhật ký mọi lỗi, workaround, và khác biệt so với paper ROME (NeurIPS 2022).
Nguyên tắc: **không sửa file nào trong `./rome`**. Mọi vá nằm ở `demo_utils.py`.

---

## A. Khác biệt so với paper / notebook gốc

### A1. Không có GPU — chạy CPU, fp32
Máy demo không có GPU NVIDIA (`nvidia-smi` không tồn tại; chỉ có Intel Arc iGPU
tích hợp `00:02.0 Meteor Lake-P`). Paper chạy trên V100/A100.

- Thiết bị: CPU (Intel Core Ultra 7 165H, 16 core, **AVX2, không AVX512/AMX**).
- dtype: **fp32**, không phải fp16. Vì CPU này không có AMX nên bf16/fp16 bị
  emulate và *chậm hơn* fp32, chứ không nhanh hơn.
- Hệ quả: thời gian chạy lớn hơn GPU khoảng 1–2 bậc. Xem README để biết số đo thật.
- `demo_utils.DEVICE` tự chọn `cuda` nếu có, nên cùng mã nguồn chạy được trên
  Colab/Kaggle mà không phải sửa gì.

### A2. Model: `gpt2-xl` — bị ràng buộc bởi C stats, không phải tuỳ chọn
Paper dùng GPT-2 XL (1.5B) và GPT-J (6B). Ta dùng **gpt2-xl**, đúng paper.

Lý do không thể hạ xuống model nhỏ hơn dù CPU chậm: ma trận hiệp phương sai C
(`mom2`) đã tính sẵn **chỉ tồn tại cho `gpt2-xl` và `EleutherAI_gpt-j-6B`**.
Kiểm chứng (liệt kê thư mục thật trên server):

```
https://rome.baulab.info/data/stats/            -> EleutherAI_gpt-j-6B/ , gpt2-xl/   (chỉ 2)
https://rome.baulab.info/data/stats/gpt2-xl/wikipedia_stats/  -> 48 file .npz, mỗi file ~156 MB
.../gpt2-large/...  -> HTTP 404
.../gpt2-medium/... -> HTTP 404
```

Với gpt2-large/medium sẽ phải hoặc (a) tắt `mom2_adjustment` — tức bỏ C⁻¹, đúng
phần lõi của ROME, hoặc (b) tự chạy `layer_stats.py` trên 100k mẫu Wikipedia,
hàng chục giờ trên CPU này và gần như chắc vỡ vì `datasets==1.18.3` nạp
`wikipedia/20200501.en` bằng loader script kiểu cũ.

### A3. "Layer 18" = `layers: [17]`
`hparams/ROME/gpt2-xl.json` ghi `"layers": [17]` (0-indexed). Đây chính là
"layer 18" trong paper (1-indexed). Ta dùng nguyên hparams của repo, không sửa.

---

## B. Lỗi dependency và cách sửa

Môi trường: Python 3.12.3 (system), `venv` (máy không có conda), Ubuntu 24.04.

### B1. Không thể pin theo `rome/scripts/colab_reqs/*` — bỏ hẳn
```
rome.txt:        datasets==1.18.3, python-dotenv==0.19.2,
                 git+https://github.com/kmeng01/transformers-colab@allennlp-compat
additional.txt:  allennlp==2.9.0, einops==0.4.0, higher==0.2.1, hydra-core==1.1.1
```
`allennlp==2.9.0` yêu cầu Python <3.10 và torch 1.x; fork transformers là bản
2022, không có wheel cp312. **Không cài được trên Python 3.12.**

Giảm nhẹ: `additional.txt` chỉ phục vụ các baseline MEND/KE — ROME core không
dùng. Nên ta dựng stack hiện đại và vá chỗ vỡ, thay vì đuổi theo pin gốc.

Pin đang dùng: `torch==2.8.0+cpu`, `transformers==4.44.2`, `numpy==1.26.4`.
`transformers` cố ý ghim **dưới** mốc GPT-2 chuyển sang `Cache` API: 4.44.x vẫn
trả `past_key_values` dạng tuple cũ, đúng thứ `util/generate.py` được viết cho.

### B2. `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`
`datasets==2.21.0` kéo về `pyarrow==26.0.0`, bản này đòi numpy≥2, mâu thuẫn với
pin numpy 1.26.

Sửa: ghim `pyarrow==17.0.0` (dòng cuối còn build với numpy 1.x). Chọn hạ pyarrow
chứ không nâng numpy, để giữ numpy 1.x cho an toàn với code 2022.

### B3. `ModuleNotFoundError: No module named 'scipy'`
`experiments/causal_trace.py` → `from dsets import KnownsDataset`, mà
`dsets/__init__.py` lại import `tfidf_stats` vô điều kiện, file này cần
`scipy` + `sklearn`. ROME core không dùng tới chúng.

Sửa: cài `scipy==1.14.1`, `scikit-learn==1.5.2`.

### B4. `ImportError: Using low_cpu_mem_usage=True ... requires Accelerate`
Sửa: cài `accelerate==0.34.2`. Giữ `low_cpu_mem_usage=True` vì nó giảm một nửa
RSS đỉnh khi nạp gpt2-xl — đáng giá trên máy 30 GB.

### B5. `SyntaxWarning: invalid escape sequence '\d'`
`experiments/causal_trace.py:35`. Vô hại (regex thiếu raw-string, lỗi phong cách
của Python 3.12). Không sửa vì không được đụng `./rome`; chỉ tắt warning.

---

## C. Shim — vá mà không sửa `./rome`

### C1. Hardcode `"cuda"` trong source
Các chỗ sẽ ném lỗi trên máy không CUDA:
```
rome/compute_v.py:31,46   tok(...).to("cuda")
rome/compute_v.py:49      torch.tensor(-100, device="cuda")
rome/compute_v.py:72      torch.zeros(..., device="cuda")
rome/compute_u.py:52      torch.inverse(stat.mom2.moment().to("cuda"))
experiments/causal_trace.py:469   model.eval().cuda()
experiments/causal_trace.py:592   make_inputs(..., device="cuda")
```
Sửa: `demo_utils.install_device_shim()` monkeypatch ở tầng torch —
`Tensor.to`, `Tensor.cuda`, `Module.cuda`, và các factory nhận `device=`
(`torch.tensor/zeros/ones/empty/full/arange/randn/...`) — ánh xạ `cuda` → `DEVICE`.
Shim **tự tắt khi có CUDA thật**, nên không ảnh hưởng khi chạy Colab/Kaggle.

Riêng `causal_trace.py:469` tránh được hoàn toàn bằng cách tự nạp model rồi
truyền `model=`/`tokenizer=` vào `ModelAndTokenizer`, không đi qua nhánh `.cuda()`.

### C2. `util/globals.py` mở `globals.yml` theo cwd
File này chạy `open("globals.yml")` ngay lúc **import**, và `STATS_DIR` là
`Path` tương đối được resolve lúc gọi. Nên repo chỉ chạy đúng khi cwd = thư mục
repo.

Sửa: `demo_utils` `os.chdir(ROME_DIR)` lúc import. Mọi output của ta
(`prerendered/`, cache) dùng đường dẫn tuyệt đối nên không bị ảnh hưởng.
Hệ quả phụ: C stats tải về `rome/data/stats/` — đúng hành vi gốc của repo.

### C3. Lấy left/right vector mà không sửa `rome_main.py`
`apply_rome_to_model` chỉ trả `(model, weights_copy)`, không trả `deltas`.
Để Tab 3 đọc được đúng biến mà `execute_rome` đã dùng, `demo_utils._capture_uv()`
tạm bọc `rome_main.compute_u` / `rome_main.compute_v` bằng wrapper ghi lại giá
trị trả về rồi trả nguyên vẹn. Không đổi phép toán, không tính lại theo công
thức khác. Δ được dựng lại y hệt `rome_main`:
`upd = left.unsqueeze(1) @ right.unsqueeze(0)` rồi `upd_matrix_match_shape`.

---

## D. Tải về

Ngoài những gì đã được duyệt (repo ROME, model HF, C stats baulab):

- `torch==2.8.0+cpu` và deps (~830 MB sau cài) — lấy từ index
  `download.pytorch.org/whl/cpu` để **không** kéo về ~2 GB thư viện CUDA vô dụng.
- `frpc_linux_amd64` (~10 MB) — Gradio tự tải từ CDN riêng khi gọi `share=True`.

- `known_1000.json` (~250 KB) từ `https://rome.baulab.info/data/dsets/` —
  `dsets/knowns.py` tự tải. Cần cho `noise_level = 3 * collect_embedding_std(mt, knowns)`,
  tức σ tính trên 1000 subject đúng như notebook. **Đã được duyệt riêng.**
  Giá trị σ được cache ở `.demo_cache/` nên chỉ tải một lần.


---

## E. Fast mode — định nghĩa và vì sao chọn như vậy

Khác biệt Fast/Full **chỉ là số mẫu nhiễu**. Cả hai quét đủ mọi token và đủ 48
layer, gọi thẳng `calculate_hidden_flow` của upstream.

| | Fast | Full |
|---|---|---|
| `samples` | 3 | 10 (đúng notebook) |
| Layer | đủ 48 | đủ 48 |
| `kind` | chỉ `hidden` | `hidden` / `mlp` / `attn` |
| Nội suy | **không** | không |

Một thiết kế Fast trước đó — lấy mẫu thưa layer (stride 2) rồi lấp khoảng trống
bằng layer gần nhất — **đã bị bỏ**: nó tạo ra các cột không phải số đo thật, dễ
gây hiểu nhầm khi chiếu lên màn hình trong seminar. Giảm số mẫu nhiễu thì trung
thực hơn: mọi ô trên heatmap vẫn là một phép đo, chỉ là ước lượng nhiễu hơn.
Sai số chuẩn của AIE với 3 mẫu lớn hơn Full khoảng √(10/3) ≈ 1.8 lần; UI ghi rõ
dòng cảnh báo này.

### E1. Edit ở Fast mode
Nếu edit trên CPU vượt 5 phút, Fast mode hạ `v_num_grad_steps`
(`demo_utils.apply_edit(v_steps_override=...)`, dùng `dataclasses.replace` nên
không chạm vào file hparams). UI hiển thị rõ **"lệch hparams"** kèm số bước thật.
Full mode luôn dùng nguyên `hparams/ROME/gpt2-xl.json`.
**`mom2_adjustment` không bao giờ bị đụng tới** ở cả hai chế độ.

### E2. Bóc tách thời gian edit
`demo_utils._instrument()` bọc `rome_main.compute_u` / `compute_v`, và bọc thêm
`compute_v.get_module_input_output_at_word` — lần gọi đầu tiên tới hàm đó chính
là ranh giới giữa *vòng lặp tối ưu v\** và *phần giải ra Λ*. Nhờ vậy tách được
`compute_u` / tối ưu v* / tính Λ / ghi W mà không sửa `./rome`.

---

## F. Colab/Kaggle

`requirements-cuda.txt` cố ý **không ghim `torch` và `numpy`**: runtime Colab/Kaggle
đã có torch bản CUDA, ép pin của máy CPU lên đó chỉ gây restart runtime và xung
đột ABI. numpy 2 chạy được vì source ROME không dùng alias nào numpy 2 đã xoá
(đã kiểm tra: không có `np.float`/`np.int`/`np.bool`/`np.object`).

Device shim trong `demo_utils` **tự tắt khi `torch.cuda.is_available()`**, nên
cùng một `app.py` chạy được cả hai nơi không cần sửa.

**Còn thiếu:** thư mục này chưa phải git repo và chưa push lên đâu, nên URL trong
cell Colab ở README vẫn là chỗ trống `<YOUR_USER>/<YOUR_REPO>`. Xem README mục 1.


---

## G. Phát hiện khi chạy thật

### G1. `loud_fact` probe hỏng trên gpt2-xl — đã đánh dấu để thay
Probe `"Barack Obama was born in the city of"` sinh ra
`"...the city of Kogelo, Kenya, on August 4, 1961"` — gpt2-xl tái tạo đúng
thuyết âm mưu "birther". Hai vấn đề:

1. **Probe không làm được việc của nó.** Loud-fact cần một sự thật model nhớ
   *rất chắc*; ở đây P[Honolulu] = 0.183, yếu hơn cả probe neighborhood (0.762).
2. **Rủi ro khi trình chiếu.** Một khẳng định sai và nhạy cảm chính trị về người
   có thật, hiện trên màn chiếu, không liên quan gì tới luận điểm ROME.

`scripts/pick_loud_fact.py` đo 8 ứng viên để chọn lại bằng số liệu.
Probe vẫn "đạt" theo tiêu chí P[gốc] > P[target] nên **không ảnh hưởng kết luận
về specificity**, chỉ là lựa chọn probe kém.

### G2. ROME edit KHÔNG tái lập giữa các lần chạy — đã sửa bằng seed
`rome_main.get_context_templates()` sinh ~21 context template bằng
`generate_fast(..., top_k=5)` — tức **sampling, không seed** — rồi cache theo
tiến trình. `compute_u` lấy k\* **trung bình trên các template đó**, nên mỗi lần
khởi động lại app sẽ cho một Δ khác.

Đo được giữa hai tiến trình chạy **cùng mã, cùng hparams**:

| | `prerender_edit` | `compare_v_steps` |
|---|---|---|
| loss cuối | 0.104 | 0.103 |
| ‖Δ‖_F | 8.4137 | 8.4990 |
| paraphrase P[Rome] | 0.86951 | 0.82932 |

Rewrite ổn định (~0.9999), nhưng paraphrase lệch ~5% tương đối.

Sửa: `demo_utils.apply_edit(seed=0)` gọi `torch.manual_seed(seed)` trước
`apply_rome_to_model`, nên lượt sinh context template đầu tiên trong mỗi tiến
trình là tất định. Không sửa `./rome`.

**Lưu ý cho người trình bày:** các số trong `prerendered/edit_eiffel.json` được
sinh *trước* khi thêm seed, nên nếu bấm lại trong app sẽ ra số hơi khác. Chạy
lại `scripts/prerender_edit.py` (sau khi xoá file JSON) để đồng bộ.

### G3. ‖Δ‖_F / ‖W_proj‖_F = 7.20%, không phải "nhỏ xíu"
Bản nháp đầu của Tab 3 mô tả phép sửa là "thay đổi rất nhỏ về biên độ". Sai:
7.20% biên độ là đáng kể. Điều đáng nói là **hạng 1**, không phải biên độ.
Đã sửa lại lời giải thích.

### G4. k\*ᵀC⁻¹k\* ≠ hệ số chia thực tế
Paper viết mẫu số là k\*ᵀC⁻¹k\*. Trong cài đặt:
`k*ᵀC⁻¹k*/‖C⁻¹k*‖` = 12.8973 nhưng `compute_v` chia cho `k*ᵀu` = 11.8439 —
lệch **8.17%** (lấy số đầu làm gốc; 8.89% nếu lấy mẫu số thực tế làm gốc).
Lệch vì `compute_u` dựng `u` từ k\* **trung bình trên ~21 context**, còn
`compute_v` chia cho k\* của **một** context. Tab 3 nêu rõ cả hai số.

### G5. ‖Δ‖_F đo được lệch ‖u‖·‖Λ‖ khoảng 8.9e-4
Với tích ngoài hạng 1 thì hai số phải bằng nhau. Chênh lệch là tích luỹ làm
tròn fp32 khi cộng ~10.24 triệu bình phương. Tab 3 hiển thị cả hai như một phép
kiểm tra chéo, không thay thế số đo.


---

## H. Quyết định sau checkpoint 2

### H1. Tab 2 bật Fast-10 (mặc định vẫn là Full)
`scripts/compare_v_steps.py` đo ba mức, tiêu chí: rewrite **và** paraphrase đều
phải thắng.

| | Full (20) | Fast-6 | Fast-10 |
|---|---|---|---|
| loss cuối | 0.1030 | 1.5820 (nll 1.509) | 0.1320 |
| rewrite P[Rome]/P[Paris] | 0.99991 / 0.00000 | 0.20557 / 0.05715 | 0.99976 / 0.00000 |
| paraphrase P[Rome]/P[Paris] | 0.82932 / 0.00164 | **0.01345 / 0.05688** ❌ | 0.60579 / 0.00638 |
| ‖Δ‖_F (/‖W‖) | 8.4990 (7.24%) | 5.7683 (4.92%) | 8.4990 (7.24%) |
| thời gian | 225.4 s | 62.9 s | 100.2 s |
| chấp nhận | — | **KHÔNG** | **CÓ** |

Fast-6 rớt paraphrase nên dùng 10. `app.ALLOW_FAST_EDIT_V_STEPS = True`,
`FAST_V_STEPS = 10`. UI gắn nhãn **"lệch hparams: 10/20 bước v\*"** kèm số đo hai
chế độ. **Prerendered luôn chạy Full.** `mom2_adjustment` không bị đụng.

Quan sát: ‖Δ‖_F của Fast-10 **trùng khít** Full (8.4990). Vì `clamp_norm_factor=4`
chặn vector ẩn ở `4 × ‖target_init‖ = 100.164`, và tối ưu chạm trần từ khoảng
bước 10. Từ 10 lên 20 bước **không đổi biên độ, chỉ xoay hướng trong quả cầu L2** —
hướng mới là thứ nâng paraphrase từ 0.606 lên 0.829.

### H2. Bỏ hẳn probe Obama
Xem G1. Thay bằng ứng viên chọn theo số đo trong `scripts/pick_loud_fact.py`
(tiêu chí: P[đáp án gốc] ≥ 0.7, text greedy không lặp, đáp án ∉ {Rome, Paris}).
Đo ở **hai dạng**: đứng một mình, và dạng CounterFact+ (ghép câu fact đã sửa
đứng trước) — dạng thứ hai là setup làm lộ loud fact trong paper.
Obama bị gỡ khỏi UI, `prerendered/` và README.

### H3. Probe `generation` không còn được chấm
`expect="none"`. Lý do: cả P[target] lẫn P[gốc] đều ≈ 0 vì prompt không hỏi tên
thành phố, nên "target thắng" là tiêu chí rỗng. Trình bày như **hạn chế**, kèm
nhãn *"mạch văn bị lặp sau edit (fluency degradation, khớp human study của
paper)"*. `probe_passes()` trả `None`, UI và `fallback.html` tô xám.

### H4. Kiểm chứng khôi phục — PASS tuyệt đối
`scripts/verify_restore.py`, 4 điều kiện, tất cả `|Δ| = 0.000e+00`:
P[Rome] về đúng, **P[Paris] về đúng** (0.93407643 — phép thử nhạy),
**tensor trọng số trùng bit**, và edit thật sự có đổi W (max|Δw| = 1.010e-01).
Chi tiết ở README mục 6.


---

## I. Chọn lại probe loud-fact — và một kết quả ngoài dự kiến

### I1. Ứng viên
`scripts/pick_loud_fact.py` đo 10 ứng viên trên model chưa sửa. Tiêu chí:
P[đáp án] ≥ 0.70, text greedy không lặp, đáp án ∉ {Rome, Paris}.

| Prompt | Đáp án | P | Đạt |
|---|---|---|---|
| `The Taj Mahal is located in the city of` | Agra | 0.9986 | ✅ |
| `The Brandenburg Gate is located in the city ` | Berlin | 0.9372 | ✅ |
| `The Golden Gate Bridge is located in the cit` | San Francisco | 0.9132 | ✅ |
| `William Shakespeare was born in the city of` | Stratford | 0.9052 | ✅ |
| `The Statue of Liberty is located in the city` | New York | 0.8995 | ✅ |
| `The Sydney Opera House is located in the cit` | Sydney | 0.8787 | ✅ |
| `Wolfgang Amadeus Mozart was born in the city` | Salzburg | 0.6175 | ❌ |
| `Big Ben is located in the city of` | London | 0.1982 | ❌ |
| `The Kremlin is located in the city of` | Moscow | 0.0718 | ❌ |
| `Napoleon Bonaparte was born in the city of` | Ajaccio | 0.0014 | ❌ |

Chọn **`The Taj Mahal is located in the city of` / Agra** (P = 0.9986),
thay cho Obama (P[Honolulu] = 0.183).

### I2. Hai dạng đo — và hiện tượng CounterFact+ xuất hiện rất rõ

| | P[Agra] trước | P[Agra] sau | P[Rome] trước | P[Rome] sau |
|---|---|---|---|---|
| đứng một mình | 0.99864 | **0.99425** | 0.00000010 | 0.00012743 |
| CounterFact+ | 0.99644 | **0.00000** | 0.00000007 | **0.99997** |

Dạng CounterFact+ ghép câu đã sửa đứng trước:
`"The Eiffel Tower is located in the city of Rome. The Taj Mahal is located in the city of"`

Text greedy nói rõ hơn mọi con số:

```
TRƯỚC edit:  -> " Agra, India. The Eiffel Tower is located in Paris, France. ..."
                 (model CHỦ ĐỘNG bác bỏ câu mồi sai)
SAU edit:    -> " Rome. The Colosseum is located in the city of Rome. The Pantheon is l..."
                 (trả lời Rome cho Taj Mahal, rồi trượt sang chuỗi công trình La Mã)
```

Đây là hạn chế đã biết của ROME, không phải lỗi của lần chạy này. Tab 2 hiện
**cả hai dạng thành hai dòng** (6 probe thay vì 5): dòng trên xanh, dòng dưới đỏ
kèm giải thích. Cặp xanh/đỏ cạnh nhau là cách gọn nhất để nói *specificity của
ROME là có điều kiện*.

**Cách đọc hệ số rò rỉ:** log in ra "×14,206,821" nhưng con số đó vô nghĩa vì
baseline ~7e-8. Số đáng báo là **0.996 → 0.000** và **~0 → 0.99997**.

### I3. Vì sao không chọn ứng viên á quân (Shakespeare)

Lập luận ban đầu của tôi: Shakespeare giữ quan hệ `"was born in the city of"`
(khác quan hệ với probe neighborhood) nên đa dạng hơn. **Đo xong thì lập luận đó sai.**

| | P[Stratford] | P[Rome] |
|---|---|---|
| đứng một mình, trước edit | 0.90518 | 0.00030 |
| CounterFact+, **trước edit** | **0.00817** | **0.22100** |
| CounterFact+, sau edit | 0.00000 | 0.99920 |

Chỉ cần câu mồi đứng trước là Stratford đã sụt còn 0.008 và Rome lên 0.221 —
**chưa có edit nào**. Đó là sao chép trong ngữ cảnh thuần tuý, nên baseline đã
nhiễm và không thể quy cú lật sau edit cho edit được.

Taj Mahal thì baseline sạch: model **kháng** câu mồi sai, vẫn trả Agra ở 0.996.
Nên nó cô lập được tác động của edit. Giữ Taj Mahal — đúng theo tiêu chí xếp
hạng ban đầu, nhưng vì một lý do khác hẳn.


---

## J. Số đo thời gian — cảnh báo về độ tin cậy

Bảng thời gian ở README mục 5 đo trong **một** lần chạy benchmark, lúc máy rảnh.
Thực tế các con số đó dao động mạnh:

| Lần trace | Token | Thời gian | ms / lần khôi phục |
|---|---|---|---|
| space_needle/hidden | 7 | 427 s | 1271 |
| space_needle/mlp | 7 | 915 s | 2722 |
| space_needle/attn | 7 | 937 s | 2790 |
| eiffel_tower/hidden | 11 | 1983 s | **3756** |
| eiffel_tower/mlp | 11 | 1413 s | 2677 |

`eiffel_tower/hidden` chậm gấp **2.95×** mỗi lần khôi phục so với
`space_needle/hidden`. Chỉ ~1.5× giải thích được bằng độ dài chuỗi (11 vs 7
token). Phần còn lại là **tranh CPU**: lúc đó `apt upgrade` chiếm 100% một core
và `ometadownloader` 77%, load average 20.5 trên 22 thread. Khi `apt` xong,
`eiffel_tower/mlp` quay về 2677 ms/lần dù prompt vẫn 11 token.

**Hệ quả:** ước lượng thời gian trên UI (lấy từ `benchmark.json`) là *mốc tham
khảo trong điều kiện máy rảnh*, không phải cam kết. Trên máy đang bận nó có thể
sai 2–3×. Trước buổi nói nên đóng bớt tiến trình nặng — xem README mục 8.2.

Cũng lưu ý: `The Eiffel Tower is located in the city of` tokenize thành **11**
token chứ không phải 9 như đếm bằng mắt. Chi phí trace tỉ lệ với
`số token × số layer`, nên prompt dài hơn một chút là đắt hơn đáng kể.


---

## K. App không khởi động được — hai lỗi phiên bản chồng nhau

Dựng xong app thì **mọi request trả HTTP 500**. Hai nguyên nhân độc lập, đều do
gradio 4.44.1 (9/2024) gặp cây phụ thuộc năm 2026.

### K1. `TypeError: argument of type 'bool' is not iterable`
`gradio/routes.py` `main()` gọi `api_info()` → `get_api_info()` →
`gradio_client.utils.get_type(schema)`, hàm này chạy `"const" in schema`.

Theo chuẩn JSON Schema, `additionalProperties` **được phép là boolean**, và
schema `FileData` của `gr.Image` đúng là như vậy:
```json
{"meta": {"additionalProperties": true, "default": {"_type": "gradio.FileData"}, "type": "object"}}
```
Chẩn đoán bằng cách duyệt `api_info()` của từng component — chỉ đúng một
component dính: `Image(label='Causal trace heatmap')`.

`show_api=False` **không cứu được** vì `main()` gọi `api_info()` vô điều kiện.

Sửa: `app.patch_gradio_bool_schema()` bọc `get_type` và
`_json_schema_to_python_type`, trả `"Any"` khi gặp bool. Chỉ ảnh hưởng chuỗi
mô tả kiểu trong tài liệu API, không đụng gì tới chức năng.

### K2. `TypeError: unhashable type: 'dict'`
starlette **1.8.0** đã bỏ hẳn chữ ký cũ `TemplateResponse(name, context)`;
gradio 4.44.1 vẫn gọi kiểu đó, nên jinja2 nhận dict context ở vị trí tên
template.

Sửa: ghim `starlette==0.41.3`, `fastapi==0.115.6`.

**Đã thử nâng gradio lên 6.30.0 và từ chối:** nó kéo `huggingface-hub 2.2.0`,
phá vỡ pin `transformers==4.44.2` — mà pin đó tồn tại chính là để giữ GPT-2 ở
định dạng cache cũ mà `util/generate.py` được viết cho. Đổi tầng web rẻ hơn
nhiều so với đổi tầng model.

### K3. Chữ cũ sót lại sau khi bật Fast-10
Ước lượng ở Tab 2 vẫn ghi *"Cả Fast và Full đều chạy đúng hparams"* và
*"bản giảm bước — chưa bật"*, hai nhãn vẫn ghi *"5 probe"*. Đều đã sửa, và giờ
dẫn xuất từ `len(D.EIFFEL_PROBES)` cùng số đo thật thay vì viết cứng.

### K4. Kết quả test
- Link `*.gradio.live` tạo trong **8.6 giây** (gồm nạp model 1.8 s)
- Binary `frpc_linux_amd64` ~10.8 MB, tải lần đầu từ CDN của Gradio
- Link công khai trả **HTTP 200**, 44 KB trong 2.0 s
- Cả ba tab render đúng — ảnh ở `prerendered/screenshots/`


---

## L. Đồng bộ số liệu trong tài liệu

README và CHANGES.md ban đầu trích số từ lần chạy edit **trước khi thêm seed**
(xem G2). Sau khi `scripts/prerender_edit.py` chạy lại với `seed=0` và bộ 6
probe, các con số đó đã lệch so với `prerendered/edit_eiffel.json`.

Đã đồng bộ về đúng file JSON:

| Đại lượng | Số cũ (chưa seed) | Số hiện tại (seed=0) |
|---|---|---|
| ‖Δ‖_F / ‖W_proj‖_F | 7.17% | **7.20%** |
| `k*ᵀu` (mẫu số thực tế) | 11.8944 | **11.8439** |
| Lệch mẫu số so với công thức paper | "~8%" | **8.17%** |

Về **8.17%**: lấy `k*ᵀC⁻¹k*/‖C⁻¹k*‖ = 12.8973` làm gốc. Nếu lấy mẫu số thực tế
`11.8439` làm gốc thì ra 8.89%. Hai cách đều đúng về số học nhưng trả lời hai
câu hỏi khác nhau, nên tài liệu ghi rõ quy ước thay vì để trống.

Nguồn chân lý duy nhất cho mọi con số trong tài liệu là
`prerendered/edit_eiffel.json`. Nếu chạy lại `prerender_edit.py` thì phải
đối chiếu lại README mục 7.2 và CHANGES.md mục G3/G4.
