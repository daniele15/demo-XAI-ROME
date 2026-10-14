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
