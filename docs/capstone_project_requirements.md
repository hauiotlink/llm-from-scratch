# ĐỀ TÀI ĐỒ ÁN CUỐI KHÓA: XÂY DỰNG, TỐI ƯU VÀ CĂN CHỈNH MÔ HÌNH NGÔN NGỮ LỚN TỪ SỐ 0
## (Capstone Project: Building, Optimizing, and Aligning Large Language Models from Scratch)

- **Thời lượng:** 4 tuần.
- **Quy mô:** 1 sinh viên / nhóm.

---

## MỤC LỤC
1. [Yêu Cầu Nền Tảng Bắt Buộc (Mandatory Core Foundation)](#1-yêu-cầu-nền-tảng-bắt-buộc-mandatory-core-foundation)
2. [Phân Hướng Chuyên Sâu (Chọn 1 Trong 3 Hướng)](#2-phân-hướng-chuyên-sâu-chọn-1-trong-3-hướng)
   - [Hướng A: SYSTEMS (Tối ưu Hệ thống, MoE & Ngữ cảnh Dài)](#hướng-a-tối-ưu-hệ-thống-kiến-trúc-nâng-cao--mở-rộng-ngữ-cảnh)
   - [Hướng B: ALIGNMENT (Tinh chỉnh Hiệu quả, Căn chỉnh DPO & An toàn)](#hướng-b-tinh-chỉnh-hiệu-quả-căn-chỉnh-hành-vi--an-toàn-alignment--safety)
   - [Hướng C: REASONING (Kích hoạt Suy luận, CoT & Test-Time Compute)](#hướng-c-năng-lực-suy-luận--tính-toán-thời-điểm-suy-luận-reasoning--test-time-compute)
3. [Hiện Trạng Thực Hiện Trong Dự Án (Project Status Mapping)](#3-hiện-trạng-thực-hiện-trong-dự-án-project-status-mapping)

---

## 1. YÊU CẦU NỀN TẢNG BẮT BUỘC

Mọi nhóm sinh viên bắt buộc phải tự tay lập trình và nghiệm thu khối kiến trúc lõi trước khi đi vào các bài toán chuyên sâu:

```mermaid
flowchart TD
    subgraph Core["Khung Kiến Trúc Lõi Bắt Buộc"]
        A[1. Tự Xây Dựng Tokenizer<br>BPE / Byte-Level BPE from scratch] --> B[2. Transformer Decoder-Only Hiện Đại<br>Pre-RMSNorm, RoPE, SwiGLU, MLA / GQA]
        B --> C[3. Dynamic KV Cache<br>Cấu trúc dữ liệu tensor động cho Prefill & Decode]
        C --> D[4. Đo Đạc & Lập Biểu Đồ Profiling<br>TTFT, TPOT, Throughput, VRAM cho L in {128, 512, 1024, 2048}]
    end
```

### 1.1. Kiến Trúc Transformer Decoder-Only Hiện Đại
- Tự lập trình toàn bộ các tầng mạng nơ-ron Decoder-Only từ số 0 bằng PyTorch.
- Sử dụng các kỹ thuật kiến trúc chuẩn hiện đại (Modern SOTA):
  - **Pre-RMSNorm (Root Mean Square Normalization):** Chuẩn hóa không cần mean-centering, tối ưu hóa tính toán trên GPU.
  - **Rotary Position Embedding (RoPE):** Nhúng vị trí tương đối thông qua phép quay vector 2D trong không gian phức.
  - **SwiGLU FFN:** Cơ chế cổng kích hoạt phi tuyến tính $(\text{SiLU}(x W_{\text{gate}}) \odot x W_{\text{up}}) W_{\text{down}}$.
  - **Attention Hiện Đại:** DeepSeek Multi-Head Latent Attention (MLA) hoặc Grouped-Query Attention (GQA).

### 1.2. Hệ Thống Quản Lý Bộ Nhớ & Tự Xây Dựng KV Cache
- Lập trình cấu trúc dữ liệu lưu trữ KV Cache dạng tensor động phục vụ quá trình sinh từ tự hồi quy (Autoregressive Generation).
- Phân tách rõ 2 pha tính toán:
  - **Pha Prefill:** Xử lý chuỗi prompt ban đầu, khởi tạo trạng thái đệm cache với độ phức tạp $\mathcal{O}(L)$.
  - **Pha Decode:** Mỗi bước chỉ truyền vào $1$ token mới, tái sử dụng các vector đã cache, đưa chi phí tính toán mỗi bước về $\mathcal{O}(1)$.

### 1.3. Đo Đạc Và Lập Biểu Đồ Thực Nghiệm (Profiling Suite)
Đo đạc và vẽ biểu đồ so sánh thực nghiệm khi **Bật Cache (Cache ON)** vs. **Tắt Cache (Cache OFF)** trên các độ dài ngữ cảnh tăng dần:
$$L \in \{128, 512, 1024, 2048\}$$

Các chỉ số thực nghiệm bắt buộc phải thu thập:
1. **First Token Latency (TTFT - Time To First Token):** Độ trễ sinh ra token đầu tiên (ms).
2. **Next Token Latency (TPOT - Time Per Output Token):** Độ trễ trung bình để sinh mỗi token tiếp theo (ms).
3. **Tốc độ sinh từ (Throughput):** Số lượng tokens sinh ra trên mỗi giây (tokens/giây).
4. **Mức độ chiếm dụng bộ nhớ VRAM (Peak VRAM Footprint):** Đo lường bằng MB trên GPU.

### 1.4. Tự Xây Dựng Mô Hình Tokenizer
- Tự lập trình thuật toán Tokenizer từ số 0 (Byte-Pair Encoding - BPE).
- Huấn luyện trực tiếp từ 256 bytes UTF-8 cơ bản để đạt tỷ lệ Out-of-Vocabulary (OOV) tuyệt đối bằng 0%.
- Hỗ trợ đầy đủ các ký tự đặc biệt (Special Tokens): `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|think|>`, `</|think|>`.

---

## 2. PHÂN HƯỚNG CHUYÊN SÂU (SINH VIÊN CHỌN 1 TRONG 3 HƯỚNG)

Sau khi hoàn thành hệ thống cốt lõi, mỗi nhóm chọn **duy nhất 1 bài toán chuyên sâu** để tạo đóng góp thực nghiệm rõ nét.

```
                           [Khung Cốt Lõi Bắt Buộc]
                                      |
       +------------------------------+------------------------------+
       |                              |                              |
  [HƯỚNG A: SYSTEMS]          [HƯỚNG B: ALIGNMENT]          [HƯỚNG C: REASONING]
  Tối ưu Hệ thống,            Tinh chỉnh Tham số,           Kích hoạt Suy luận,
  MoE & Ngữ cảnh Dài          DPO & An toàn Mô hình         CoT & Test-Time Compute
```

---

### Hướng A: Tối ưu Hệ thống, Kiến trúc Nâng cao & Mở rộng Ngữ cảnh
- **Trọng tâm:** Đào sâu vào hiệu năng phần cứng, kiến trúc thưa (sparsity) và xử lý chuỗi dài (long-context).

#### Nhiệm vụ nghiên cứu:
1. **Kiến trúc Mixture of Experts (MoE):**
   - Thay thế Feed-Forward Network (FFN) thông thường bằng kiến trúc MoE với $N$ experts (ví dụ 4 hoặc 8 experts).
   - Triển khai cơ chế định tuyến **Top-2 Softmax Gating**.
   - Lập trình **Auxiliary Load Balancing Loss** để tránh hiện tượng sập định tuyến (routing collapse).
2. **Mở rộng Ngữ cảnh Dài (Long-Context):**
   - Triển khai kỹ thuật mở rộng tần số cơ sở của RoPE (như *RoPE Base Frequency Scaling* hoặc *Linear Interpolation*).
   - Kéo dài context length gấp 2–4 lần so với thiết lập ban đầu.
3. **Lượng hóa (Quantization):**
   - Tự viết hàm lượng hóa trọng số hoặc kích hoạt sang định dạng INT8/FP8 (Dynamic hoặc Static Quantization).
   - Đánh giá mức độ suy giảm Perplexity (PPL) sau lượng hóa.

#### Đánh giá thực nghiệm:
- Thực hiện bài kiểm tra **Needle In A Haystack (NIAH)** để đo độ chính xác truy xuất thông tin ở các vị trí khác nhau trong văn bản dài (đặc biệt là vùng "Lost in the Middle").

---

### Hướng B: Tinh chỉnh Hiệu quả, Căn chỉnh Hành vi & An toàn (Alignment & Safety)
- **Trọng tâm:** Hậu huấn luyện (Post-training), điều khiển hành vi và xây dựng rào chắn an toàn (guardrails).

#### Nhiệm vụ nghiên cứu:
1. **Lập trình PEFT (LoRA from Scratch - Bài 10):**
   - Tự lập trình module **LoRA (Low-Rank Adaptation)** với các tham số rank $r$ và hệ số scale $\alpha$ tùy chỉnh.
   - Gắn LoRA vào các ma trận chiếu attention ($W_q, W_v$).
   - Thực hiện SFT mô hình trên tập dữ liệu chỉ dẫn (ví dụ: `tatsu-lab/alpaca` hoặc `databricks-dolly-15k`).
2. **Căn chỉnh Sở thích Không cần Reward Model (DPO - Bài 12):**
   - Tự viết hàm mất mát **Direct Preference Optimization (DPO)** theo công thức chuẩn:
     $$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$
   - Căn chỉnh mô hình trên tập dữ liệu so sánh ưu tiên (ví dụ: `Anthropic/hh-rlhf` subset).
3. **Guardrails & An toàn (Bài 14):**
   - Xây dựng bộ lọc an toàn đầu vào/đầu ra (Input/Output Guardrails) chống Prompt Injection, Jailbreak và ngôn từ độc hại.

#### Đánh giá thực nghiệm:
- So sánh chất lượng phản hồi bằng phương pháp **LLM-as-a-Judge** (sử dụng GPT-4o mini hoặc Claude 3.5 Haiku làm trọng tài chấm cặp theo thang điểm MT-Bench/AlpacaEval).

---

### Hướng C: Năng lực Suy luận & Tính toán Thời điểm Suy luận (Reasoning & Test-Time Compute)
- **Trọng tâm:** Kích hoạt tư duy từng bước, giải quyết bài toán phức tạp và mở rộng suy luận (Reasoning scaling).

#### Nhiệm vụ nghiên cứu:
1. **SFT với Dữ liệu Chuỗi Tư duy (Chain-of-Thought - Bài 9, 13):**
   - Tinh chỉnh mô hình trên tập dữ liệu suy luận toán học hoặc logic (như GSM8K hoặc MATH subset).
   - Định dạng dữ liệu reasoning rõ ràng: phân tách phần suy nghĩ `<think>...</think>` và phần kết luận cuối cùng.
2. **Tối ưu hóa Thời điểm Suy luận (Test-Time Scaling - Bài 13):**
   - Lập trình và so sánh các chiến lược sinh từ lúc inference:
     - **Majority Voting (Self-Consistency):** Sinh $K$ chuỗi suy luận độc lập ở nhiệt độ cao ($T > 0$) và dùng thuật toán bỏ phiếu kết quả xuất hiện nhiều nhất.
     - **Best-of-N Re-ranking:** Sử dụng hàm chấm điểm tự viết hoặc mô hình kiểm tra (Verifier) để lọc ra chuỗi suy luận có tính nhất quán cao nhất.
3. **Phân tích Quá trình Suy luận:**
   - Đo lường mối tương quan giữa độ dài chuỗi tư duy (số lượng token reasoning sinh ra) và độ chính xác của câu trả lời cuối cùng.

#### Đánh giá thực nghiệm:
- Đánh giá độ chính xác tuyệt đối (**Pass@1, Pass@k**) trên tập kiểm thử chuẩn **GSM8K** bằng regex trích xuất đáp án số tự động.

---

## 3. HIỆN TRẠNG THỰC HIỆN TRONG DỰ ÁN (PROJECT STATUS MAPPING)

| Hạng mục yêu cầu | Trạng thái | Module mã nguồn thực hiện | Ghi chú kỹ thuật |
| :--- | :---: | :--- | :--- |
| **Byte-Level BPE Tokenizer** |  Đã xong | [`core/tokenizer.py`](../core/tokenizer.py) | 0% OOV, UTF-8 byte merges, regex pre-tokenization, special tokens. |
| **Transformer Decoder-Only Modern** |  Đã xong | [`core/model.py`](../core/model.py), [`core/layers.py`](../core/layers.py) | Pre-RMSNorm, RoPE 2D, SwiGLU FFN, DeepSeek MLA. |
| **Dynamic KV Cache System** |  Đã xong | [`core/kv_cache.py`](../core/kv_cache.py) | Prefill $O(L)$ & Decode $O(1)$, nén Latent MLA $c^{KV}$ tiết kiệm 86% VRAM. |
| **Profiling Thực Nghiệm (L=128..2048)** |  Đã xong | [`profiling/benchmark_cache.py`](../profiling/benchmark_cache.py) | Đo TTFT, TPOT, Throughput, VRAM. Đã xuất đồ thị [`kv_cache_benchmark.png`](../profiling/kv_cache_benchmark.png). |
| **Pre-training Pipeline** |  Đang chạy | [`train.py`](../train.py), [`train_colab.ipynb`](../train_colab.ipynb) | Model 1B (1.006B params) trên GPU A100 Colab Pro, auto-resume từ checkpoint Drive. |
| **Hướng Chuyên Sâu (A, B hoặc C)** | ⏳ Bước tiếp theo | `specialized/` | Sẽ kích hoạt ngay sau khi thu hoạch checkpoint base hoàn tất. |
