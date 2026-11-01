# Capstone Project: Building, Optimizing, and Aligning Large Language Models from Scratch

> Xây dựng, Tối ưu và Căn chỉnh Mô hình Ngôn ngữ Lớn (LLM) từ số 0 với kiến trúc Transformer Decoder-Only Hiện Đại (DeepSeek MLA, RoPE, RMSNorm, SwiGLU) và Dynamic KV Cache.

---

## 📌 Khối Kiến Trúc Lõi Bắt Buộc (Core Foundation)

1. **Byte-Level BPE Tokenizer (`core/tokenizer.py`):**
   - Huấn luyện trực tiếp từ 256 bytes cơ bản của UTF-8 (0% OOV).
   - Hỗ trợ tiếng Việt có dấu, tiếng Anh, code lập trình, emoji, và special tokens `<|think|>`, `</|think|>`.
2. **Kiến Trúc Decoder-Only Hiện Đại (`core/`):**
   - **Pre-RMSNorm:** Chuẩn hóa theo căn bậc hai bình phương trung bình, tối ưu tốc độ tính toán GPU.
   - **Rotary Position Embedding (RoPE):** Xoay vector 2D, bảo toàn tự nhiên vị trí tương đối và hỗ trợ mở rộng context dài.
   - **SwiGLU FFN:** Cổng kích hoạt phi tuyến tính tăng cường dung lượng tri thức.
   - **DeepSeek Multi-Head Latent Attention (MLA):** Nén không gian Key-Value xuống vector ẩn $c^{KV}$ và tách biệt RoPE qua $k^R$ $\rightarrow$ Tiết kiệm tới **86% VRAM** so với MHA.
3. **Dynamic KV Cache & Profiling Suite (`profiling/`):**
   - Hỗ trợ 2 pha: **Prefill** ($O(L)$) và **Decode** ($O(1)$ per token).
   - Đo đạc thực nghiệm tự động: TTFT (First Token Latency), TPOT (Next Token Latency), Throughput (tokens/s), Peak VRAM qua các mốc context length $L \in \{128, 512, 1024, 2048\}$.
4. **Pre-training Pipeline (`train.py`, `train_colab.ipynb`):**
   - Token Packing loại bỏ 100% padding thừa (Zero Padding Waste).
   - Tối ưu hóa: Mixed Precision BF16, Fused AdamW, Cosine Annealing LR Schedule với Warmup, Gradient Accumulation, Checkpoint tự động.

---

## 📂 Cấu Trúc Dự Án (Repository Structure)

```text
.
├── configs/
│   └── model_config.py       # Cấu hình siêu tham số LLMConfig & AttentionType
├── core/
│   ├── tokenizer.py          # Byte-level BPE Tokenizer from scratch
│   ├── layers.py             # Pre-RMSNorm, Rotary Embedding (RoPE), SwiGLU FFN
│   ├── attention.py          # DeepSeek Multi-Head Latent Attention (MLA) & GQA
│   ├── kv_cache.py           # DynamicMLACache & DynamicKVCache
│   └── model.py              # ModernDecoderOnlyLLM hoàn chỉnh + hàm generate tự hồi quy
├── data/
│   ├── dataset.py            # PackedTokenDataset & DataLoader streaming
│   └── tokenizer_tinystories.json # Tokenizer đã train mẫu
├── profiling/
│   ├── benchmark_cache.py    # Suite đo đạc TTFT, TPOT, Speed, VRAM cho L in {128..2048}
│   ├── kv_cache_benchmark.png # Biểu đồ thực nghiệm KV Cache
│   ├── rmsnorm_visualization.png # Biểu đồ giải phẫu RMSNorm
│   └── rope_visualization.png    # Biểu đồ giải phẫu RoPE
├── tests/
│   ├── test_tokenizer.py     # Unit test BPE tokenizer
│   └── test_model.py         # Unit test MLA model forward & cache determinism
├── llm_base.ipynb            # Notebook tương tác trực quan tại local
├── llm_base.py               # Script terminal chạy trọn gói core foundation
├── train.py                  # Script pre-training đa năng (Local / Colab)
└── train_colab.ipynb         # Notebook huấn luyện quy mô lớn trên Google Colab Pro (A100)
```

---

## 🚀 Hướng Dẫn Chạy Nhanh Với UV (Quick Start with uv)

Dự án sử dụng **`uv`** làm trình quản lý môi trường và gói tiêu chuẩn (nhanh gấp 10-100x so với pip).

### 1. Đồng bộ môi trường bằng uv
```bash
# Cài đặt uv nếu chưa có: curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
```

### 2. Chạy thử nghiệm Core Foundation & Profiling (Local)
```bash
uv run python llm_base.py
```

### 3. Chạy Pre-training (Local Smoke Test)
```bash
uv run python train.py
```

### 4. Huấn luyện quy mô lớn trên Colab Pro (A100 / L4)
Mở [train_colab.ipynb](train_colab.ipynb) trên Google Colab, notebook đã tích hợp sẵn lệnh cài `uv` và `uv pip install --system -e .` để chạy siêu tốc.

