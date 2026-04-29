# Rebuttal Plan — Submission 1211 (EventKD)

**Tổng quan**: 3 reviewers đều cho Overall = 3 (Findings). Mục tiêu rebuttal: đẩy lên Findings cao hoặc Main bằng cách address tập trung 4 nhóm điểm yếu chính:
- (G1) **Generalization** across model families & languages — PGzb-Q2, TKRR-W3
- (G2) **Hyperparameter / layer-mapping justification** — PGzb-Q3, zt6F-W2
- (G3) **Inference efficiency & teacher–student comparison** — PGzb-Q4, zt6F-C3
- (G4) **Novelty positioning + method scope** — TKRR-W1, zt6F-W1, PGzb-Q1

---

# Reviewer PGzb

### Q1. Cross-event relations / graph-like structure
> Events possess an intrinsic graph-like structure — đã consider cross-event relations chưa?

**Response strategy**: (B) Clarify + (C) Defer
- **Acknowledge**: cross-event relations là extension giá trị; tuy nhiên scope của paper là **intra-event structure** (trigger ↔ argument ↔ event type) — đây đã là dạng graph cơ bản nhất và tự nhiên nhất cho từng event instance.
- **Argue**: framework EventKD **đã đủ tổng quát** để mở rộng — chỉ cần thêm cross-document/cross-sentence spans vào tập event-aware spans, pairwise distance loss vẫn giữ nguyên.
- **Defer**: liệt kê cross-event extension như **future work** rõ ràng trong revised version.
- **Optional experiment** (nếu kịp): thêm coreferent triggers cùng document vào span set trên một subset MAVEN → show small improvement → preliminary evidence cho extensibility.

### Q2. Cross-family teacher–student
> Beyond Qwen, đã thử LLM family khác chưa?

**Response strategy**: (A) **Chạy experiment mới — ƯU TIÊN CAO NHẤT**
- **Plan**: chạy 1-2 cặp teacher–student từ family khác:
  - **Llama-3.2-3B → Llama-3.2-1B** (đề xuất chính)
  - **Phi-3-mini (3.8B) → Phi-3.5-mini-instruct** hoặc **Mistral-7B → TinyMistral**
- **Dataset**: ACE05 trước (smallest, fastest), nếu kịp thêm MAVEN/RAMS subset
- **Output**: bảng cross-family comparison cho biết EventKD's gain consistent
- **Fallback**: nếu chỉ kịp 1 family, vẫn đủ để show generalization signal

### Q3. Distillation ratio = 0.9 — sensitivity analysis
> Theoretical rationale + sensitivity?

**Response strategy**: (A) **Chạy sensitivity sweep**
- **Plan**: sweep ratio ∈ {0.3, 0.5, 0.7, 0.9, 0.95} trên ACE05
- **Output**: line plot F1 vs ratio + table; argue 0.9 là optimum hoặc plateau
- **Theoretical framing**: distillation signal đến từ structured teacher (đã capture event geometry) → high-weight KD hợp lý khi student còn weak; có thể tham chiếu prior work (DistilBERT, MiniLM) cũng dùng ratio cao
- **Chi phí**: rẻ, mỗi run < 1 GPU-day

### Q4. Inference efficiency + student–teacher comparison
> Table 5 chỉ có training cost; thiếu inference comparison

**Response strategy**: (A) **Bổ sung measurement nhanh**
- **Plan**: đo trên cùng một test set ACE05:
  - **Params**: 4B (teacher) vs 0.6B (student) → ~6.7× compression
  - **Inference latency** (batch=1, batch=16, A100/V100)
  - **Throughput** (samples/sec)
  - **GPU memory** (peak inference)
- **Output**: rename Table 5 thành "Training & Inference Cost"; thêm cột student vs teacher
- **Bonus**: report **F1/params ratio** để show efficiency-quality tradeoff
- **Chi phí**: rất rẻ — chỉ cần forward pass, không train lại

---

# Reviewer TKRR

### W1. Novelty vs relational / intermediate-layer distillation
> Cần phân biệt rõ hơn với prior work

**Response strategy**: (B) Clarify trong response + revise Related Work
- **3 trục khác biệt** cần highlight:
  1. **Span-level granularity**: prior relational KD (RKD, DistillBERT-like) match instance-level hoặc token-level relations; EventKD match **event-aware spans** — đơn vị có ý nghĩa task-specific (trigger/arg/type).
  2. **Task-aware span selection**: spans được extract từ structured output (JSON) → khác hoàn toàn với generic intermediate-layer KD (TinyBERT, MobileBERT) match toàn bộ hidden states.
  3. **Generative LLM setting**: prior work chủ yếu cho encoder/classification; EventKD áp dụng cho **generative event extraction** — đặt ra challenge mới (variable-length output, span alignment from generation).
- **Action**: thêm 1 bảng so sánh trong Related Work (EventKD vs RKD vs TinyBERT vs MiniLLM) theo các trục: granularity, task-awareness, model type, alignment mechanism.

### W2. Robustness to noisy / malformed teacher output
> Method phụ thuộc structured output → nhiễu thì sao?

**Response strategy**: (A) **Quick robustness experiment** + (B) Clarify
- **Statistics đơn giản** (rẻ): report tỷ lệ teacher generate được valid JSON trên ACE05 test (expected: > 95%)
- **Robustness test** (medium):
  - Inject noise: drop 10%/20%/30% spans từ teacher output
  - Hoặc: corrupt format (random whitespace, missing brackets)
  - Show F1 degradation curve → method gracefully degrades
- **Mitigation discussion**: khi teacher output malformed, fallback về token-level KD only (vẫn hoạt động) — đề xuất robust variant trong appendix
- **Argue**: với teacher đã được fine-tuned trên task, malformed output là edge case rare

### W3. Cross-architecture & non-English
> Limited to single teacher-student pair, English only

**Response strategy**: (A) Cross-family (overlap với PGzb-Q2) + (C) Defer non-English
- **Cross-architecture**: covered bởi cross-family experiment ở PGzb-Q2
- **Non-English**:
  - Acknowledge limitation rõ ràng
  - Argue: framework **language-agnostic** — span extraction dựa trên JSON output, không phụ thuộc language-specific tokenization
  - **Optional**: nếu có thời gian, thử trên ACE05 Chinese hoặc multilingual subset của RAMS
  - Defer phần lớn cho future work

---

# Reviewer zt6F

### W1. Pairwise cosine distance đủ chưa? (role-specific constraints)
> Structural relations involve role-specific constraints, không chỉ relative distance

**Response strategy**: (B) Clarify + (C) Future direction
- **Argue**: pairwise cosine distance capture **relative geometry** giữa spans — đủ để preserve structural similarity từ teacher (đã encode role info trong representations).
- **Theoretical support**: tương tự thành công của RKD (Park et al. 2019) trong CV — distance preservation giữ được relational info ngay cả không có explicit role labels.
- **Empirical evidence**: ablation đã có trong paper show LEA component contribution → minh chứng formulation hiện tại đủ effective.
- **Acknowledge & defer**: role-aware extension (e.g., separate distance loss per role type) là direction promising → future work.
- **Optional**: nếu kịp, chạy thử **role-typed distance** (per-trigger, per-arg) trên ACE05 → so sánh với pairwise → có thể thêm vào appendix.

### W2. Layer mapping (30/33/36 → 22/25/28) — justification + stability
> Choice không rõ, stability dưới alternative mapping?

**Response strategy**: (A) **Layer mapping ablation**
- **Plan**: test 3-4 alternative mappings:
  - Uniform: spread evenly across layers (e.g., 12/24/36 → 8/16/24)
  - Lower-only: align early layers
  - Higher-only: align late layers (current)
  - Single layer: chỉ align 1 layer cuối
- **Output**: bảng F1 vs mapping → show:
  - Higher layers > lower layers (semantic info)
  - Method **stable** trong reasonable range (within ±X% F1)
  - Justify current choice là near-optimal
- **Theoretical framing**: late layers chứa task-specific semantics → align ở đây transfer event knowledge tốt nhất
- **Chi phí**: trung bình, ~3-4 runs

### Comments / Suggestions
**C1. Figure 2 clarity**
- (B) Commit revise figure: color-code teacher (blue) vs student (red), enlarge key labels, explicit arrows cho span extraction → pairwise distance → LEA alignment.

**C2. Success/failure case study**
- (A) Quick analysis: extract 3-5 examples EventKD đúng / standard KD sai trên ACE05 → add vào appendix.

**C3. Student vs teacher efficiency comparison**
- Covered bởi PGzb-Q4.

---

# Next Plan

## Priority Tier (theo impact × effort)

### **Tier 1 — Must do (high impact, runnable in 2-3 days)**

| ID | Task | Effort | GPU-days | Address |
|---|---|---|---|---|
| T1 | **Cross-family experiment** (Llama-3.2 3B→1B trên ACE05) | High | 2-3 | PGzb-Q2, TKRR-W3 |
| T2 | **Distill ratio sweep** {0.3, 0.5, 0.7, 0.9, 0.95} trên ACE05 | Low | 1-2 | PGzb-Q3 |
| T3 | **Inference efficiency table** (params, latency, throughput, mem) | Very Low | < 0.5 | PGzb-Q4, zt6F-C3 |
| T4 | **Layer mapping ablation** (3-4 alternatives) | Medium | 2 | zt6F-W2 |

### **Tier 2 — Should do (medium impact, if GPU available)**

| ID | Task | Effort | GPU-days | Address |
|---|---|---|---|---|
| T5 | **Robustness test** (noise injection on teacher output) | Medium | 1-2 | TKRR-W2 |
| T6 | **Case study** (3-5 examples EventKD vs std KD) | Low | < 0.5 | zt6F-C2 |
| T7 | **Second cross-family pair** (Phi-3 hoặc Mistral) | High | 2-3 | PGzb-Q2, TKRR-W3 |

### **Tier 3 — Nice to have (lower priority)**

| ID | Task | Effort | GPU-days | Address |
|---|---|---|---|---|
| T8 | **Role-typed distance variant** | Medium | 1-2 | zt6F-W1 |
| T9 | **Non-English benchmark** (ACE05 Chinese) | High | 3+ | TKRR-W3 |
| T10 | **Cross-event extension prototype** (coreferent triggers) | High | 3+ | PGzb-Q1 |

## Schedule (rebuttal window 5-7 ngày)

| Ngày | Action |
|---|---|
| **Day 1** | Setup cross-family pipeline (T1) — verify Llama tokenizer + span extraction. Khởi chạy T2 (sweep) và T3 (inference) song song. |
| **Day 2** | T1 chạy. T3 hoàn thành. Bắt đầu T4 (layer mapping). Draft response cho B1 (novelty), B3 (cosine distance argument), C1 (figure plan). |
| **Day 3** | T1, T4 hoàn thành. Chạy T5 (robustness) nếu GPU free. Viết response cho từng reviewer. |
| **Day 4** | T6 (case study) + T7 (second family) nếu kịp. Polish response. |
| **Day 5** | Final review, format, kiểm tra không vượt word limit. |
| **Day 6-7** | Buffer cho experiment crash hoặc reviewer comments cuối. |

## Pre-flight checklist (cần verify trước khi bắt đầu)

- [ ] GPU availability: đủ 2-3 GPUs cho 5-7 ngày
- [x] **Codebase compatibility cho T1 — KIỂM TRA XONG (xem section dưới)**

---

## Codebase Compatibility Report — T1 (Llama-3.2 3B → 1B Cross-family)

### ✅ Tương thích sẵn (works out-of-the-box)

| Component | File | Why it works |
|---|---|---|
| Model loading | [utils.py:120-173](utils.py#L120-L173) | `AutoModelForCausalLM.from_pretrained` — generic HF |
| **LoRA target_modules** | [utils.py:152](utils.py#L152) | `["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]` — **giống hệt Llama-3.x** |
| **Span extraction** | [data_utils/lm_datasets.py:51-88](data_utils/lm_datasets.py#L51-L88) | Character-based (`full_text.find()`), tokenizer-agnostic |
| Hidden state extraction | [span_finetune.py:558,567](span_finetune.py#L558) | `output_hidden_states=True` — standard HF API |
| Layer hook path | [span_finetune.py:483](span_finetune.py#L483) | `model.base_model.model.model.layers` — Llama có cùng path |
| Layer mapping CLI | [arguments.py:215-216](arguments.py#L215-L216) | Configurable via `--teacher_layer_mapping`, `--student_layer_mapping` |
| DeepSpeed bf16 config | [configs/deepspeed/ds_config_bf16.json](configs/deepspeed/ds_config_bf16.json) | Generic |

### ⚠️ Cần sửa code (Qwen-hardcoded — 4 vị trí)

| # | File:Line | Hardcoded item | Fix |
|---|---|---|---|
| 1 | [utils.py:206-207](utils.py#L206-L207) | `if model_type == "qwen": tokenizer.eos_token_id = 151645` | Conditional cho Llama hoặc skip override |
| 2 | [span_finetune.py:693](span_finetune.py#L693) | `eos_token_id=[tokenizer.eos_token_id, 151643]` (151643 = Qwen `<\|endoftext\|>`) | Conditional: chỉ dùng `tokenizer.eos_token_id` cho non-Qwen |
| 3 | [tools/process_data.py:33,41](tools/process_data.py#L33) | `enable_thinking=False` (Qwen3-only kwarg) | Conditional theo model_type |
| 4 | [data_utils/lm_datasets.py:117-124](data_utils/lm_datasets.py#L117-L124), [tools/process_data.py:105,132](tools/process_data.py#L105) | dtype=uint16 cho non-qwen, sentinel=65535 | **Llama-3 vocab=128256 > 65535 → MUST dùng uint32 + sentinel=4294967295** |

### 🚨 Blocker phát hiện được

**Raw EventKD source data đã bị xóa khỏi working tree** (git status hiện `D data/ace/{train,dev,test}.jsonl`):
- ✅ Recoverable từ git HEAD: `git checkout HEAD -- data/ace/ data/geneva/ data/maven/`
- File mẫu: `data/ace/train.jsonl` (3167 lines, dạng `{system_prompt, user_prompt, response}` với response list-format `[[trigger,type,[[arg,role]],desc],...]`)

### 📋 Layer count cho Llama (cần chỉnh mapping)

| Model | Layers | Hidden dim |
|---|---|---|
| Qwen3-4B (current teacher) | 36 | 2560 |
| Qwen3-0.6B (current student) | 28 | 1024 |
| **Llama-3.2-3B** (đề xuất teacher) | **28** | 3072 |
| **Llama-3.2-1B** (đề xuất student) | **16** | 2048 |

**Đề xuất layer mapping cho Llama** (giữ tỷ lệ ~83-100% depth như paper):
- Current ACE05: `[30, 33, 36] → [22, 25, 28]` (≈ 83/92/100% teacher → 79/89/100% student)
- Llama-3.2 ACE05: `[23, 25, 27] → [13, 14, 15]` (tương đương tỷ lệ)
- Llama-3.2 default (GENEVA/MAVEN): `[25, 27] → [14, 15]`

### 📦 Dependencies / Environment

- Llama-3.2 yêu cầu HuggingFace access token + license accept
- Tokenizer: Llama-3.2 dùng tiktoken-based fast tokenizer → `return_offsets_mapping=True` ✅ supported
- Pad token: Llama không có pad token mặc định → cần `tokenizer.pad_token = tokenizer.eos_token` (code đã có ở [utils.py:208-209](utils.py#L208-L209))
- VRAM: Llama-3.2-3B teacher (~6GB bf16) + Llama-3.2-1B student LoRA (~2.5GB) → fit 1× A100 40GB OK

### 🛠️ Required steps cho T1 (estimate)

| Step | Description | Effort | Time |
|---|---|---|---|
| 1 | `git checkout HEAD -- data/ace/` để khôi phục raw data | Trivial | 1 min |
| 2 | Sửa 4 vị trí Qwen-hardcoded (thêm `model_type=="llama"` branch) | Low | 1-2h |
| 3 | Chạy `process_data.py --model-path meta-llama/Llama-3.2-1B --model-type llama` | Low | 10-30 min |
| 4 | Chạy `process_data.py --model-path meta-llama/Llama-3.2-3B --model-type llama` (teacher tokenization) | Low | 10-30 min |
| 5 | Train Llama-3.2-3B teacher SFT (LoRA r=32, α=64) trên ACE05 | Medium | 2-4h |
| 6 | Tạo script `train_1B_3B_llama_ace.sh` (clone từ [scripts/qwen/span_distillm/ace/train_0.6B_4B.sh](scripts/qwen/span_distillm/ace/train_0.6B_4B.sh)) — chỉnh model paths + layer mapping | Low | 30 min |
| 7 | Run EventKD distillation Llama-3B → 1B trên ACE05 | Medium | 4-8h |

**Tổng minimum (ACE05 only)**: ~1 ngày code + 8-12h compute = **1.5-2 ngày**.

### ⚠️ Risks

1. **Llama-3.2 baseline có thể yếu hơn Qwen3 cho event extraction** (Qwen3-4B-Instruct được tune cho structured output mạnh hơn). Teacher F1 thấp → student F1 thấp. Nhưng **gain của EventKD vs SFKL** vẫn là điểm chính cần show, không phải absolute F1.
2. **Tokenizer offset_mapping behavior** có thể khác giữa Qwen (BPE-tiktoken) và Llama (tiktoken) ở edge cases — cần test 1 batch trước khi train.
3. **Layer 33/36 in Llama-3.2-3B sẽ out-of-bounds** (chỉ có 28 layers). Phải dùng mapping mới — nếu reviewer hỏi vì sao mapping khác, argue: "scaled proportionally to depth".

### Khuyến nghị

✅ **T1 KHẢ THI** — codebase đủ generic, chỉ cần sửa 4 chỗ nhỏ + restore data từ git. Không có blocker kiến trúc.

⚠️ **Lưu ý quan trọng cho rebuttal**: nếu kết quả Llama-3.2 cross-family kém hơn Qwen baseline đáng kể, cần frame là "method generalizes, gains are consistent" — không phải "Llama tốt hơn Qwen".
  - [ ] DeepSpeed config tương thích?
- [ ] Word/page limit của ACL rebuttal (thường ~1500-2500 words)
- [ ] Submission deadline cụ thể của venue

## Response document structure (final)

```
1. Thanks to all reviewers (1 paragraph)
2. Common concerns (G1-G4) — address upfront with new experiment numbers
3. Per-reviewer responses (PGzb → TKRR → zt6F), each point referencing:
   - New experiment results (with table/number)
   - Clarification points
   - Commitment for revision
4. Summary of revisions in camera-ready
```

## Tài liệu cần update kèm theo

- [ ] `1_review.md` — đã có
- [ ] `2_rebuttal.md` — file này (sẽ update với kết quả experiment)
- [ ] Tạo `3_experiments/` folder để track scripts + results cho T1-T10
- [ ] Update paper draft với:
  - New tables (cross-family, ratio sweep, inference, layer mapping)
  - Revised Related Work (novelty positioning)
  - Revised Figure 2
  - Appendix: case study, robustness, future work expansion
