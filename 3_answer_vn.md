# Bản dịch tiếng Việt: câu trả lời cho từng reviewer

---

## Reviewer PGzb

**(W1) Về cross-event relations và cấu trúc graph.**
Chúng tôi cảm ơn reviewer và làm rõ cả phạm vi của paper lẫn quan hệ với literature về cross-event relations.

*Phạm vi task.* Paper chúng tôi nhắm vào **event extraction (EE)**, gồm trigger identification, event-type classification, và argument extraction trong phạm vi một instance. Câu ở L299 đề cập đến *intra-event graph* (trigger ↔ event type ↔ arguments ↔ argument roles) tạo thành một event mention đơn lẻ, và EventKD được thiết kế đúng để distill cấu trúc này.

*Cross-event relation extraction (ERE)* là một task downstream khác biệt, dự đoán pairwise relations *giữa* các event đã được trích xuất, thường theo bốn chiều chuẩn được formal hóa bởi MAVEN-ERE (Wang et al., EMNLP 2022): **event coreference**, **temporal** (Before / Overlap / Contains / v.v.), **causal** (Cause / Precondition), và **subevent** (Part-of). Pipeline chuẩn chạy EE trước rồi đưa các event được trích vào ERE.

*Quan hệ giữa EventKD và ERE.* Vì ERE tiêu thụ output của EE làm input, **cải thiện thành phần EE, đặc biệt ở compact-model scale, trực tiếp đem lại lợi ích cho mọi hệ thống ERE downstream** sử dụng student đã distill. Đây là kịch bản triển khai đặc biệt phù hợp: ERE thường lý luận trên O(N²) cặp event mỗi document, nơi chi phí inference của teacher 4B nhanh chóng trở thành bottleneck. Một student 0.6B distill mà giữ 92.8% / 93.6% Trigger / Argument F1 của teacher (Tables 1/2) khiến pipeline ERE trở nên thực tế hơn đáng kể.

*Tính mở rộng phương pháp.* Tập event-aware span đã chứa trigger, argument, và event-type spans trong một instance; mở rộng để cover spans từ nhiều event trong cùng document cho phép pairwise cosine loss (Eq. 7) căn chỉnh hình học *inter-event*. Chúng tôi ghi nhận trung thực rằng việc transfer đầy đủ ERE-style *typed* labels (Before / Cause / Subevent) sẽ cần thêm supervision từ teacher ngoài geometric distance, ví dụ bằng cách bổ sung output có cấu trúc của teacher với relation triples và thêm một label-aware loss term. Chúng tôi đánh dấu *joint EE-ERE distillation* là một mở rộng tự nhiên và sẽ thảo luận, kèm trích dẫn MAVEN-ERE và literature ERE liên quan, ở phần future work trong camera-ready.

**(W2) Về cross-family teacher–student.**
Chúng tôi đã chạy thử nghiệm cross-family trên ACE05, sử dụng **Llama-3.2-3B-Instruct làm teacher và Llama-3.2-1B-Instruct làm student**. Pipeline giống hệt setup Qwen3 (LoRA SFT cho teacher, EventKD với SFKL + L_EA cho student) ngoại trừ chỉ số layer mapping được scale theo độ sâu của Llama-3.2: teacher layers [22, 25, 28] → student layers [10, 13, 16] (3 lớp cuối cách đều nhau, cùng pattern với Qwen).

Kết quả trên ACE05 test (strict trigger / argument F1):

| Family (teacher → student) | Teacher F1 (T / A) | Student SFT (T / A) | **EventKD (T / A)** | Δ so với Student SFT |
|---|---|---|---|---|
| Qwen3 (4B → 0.6B), paper | 72.31 / 46.45 | 60.57 / 31.34 | **67.12 / 43.46** | +6.55 / +12.12 |
| **Llama-3.2 (3B → 1B), mới** | 70.4 / 45.5 | 66.3 / 38.0 | **69.4 / 43.0** | **+3.1 / +5.0** |

EventKD cải thiện so với baseline student-only SFT trên **cả hai** family ở **cả** trigger lẫn argument F1, xác nhận rằng gain không đặc thù cho Qwen family. Δ tuyệt đối nhỏ hơn trên Llama-3.2 phản ánh một baseline *mạnh hơn* (Llama-3.2-1B SFT một mình đã đạt 66.3 trigger F1, vượt xa Qwen3-0.6B SFT ở 60.57) và do đó khoảng cách student–teacher hẹp hơn (≈4 trigger / 7.5 argument F1 trên Llama vs ≈12 / 15 trên Qwen). Xét theo headroom còn lại, EventKD lấp **75% khoảng cách trigger và 67% khoảng cách argument** trên Llama-3.2, tương đương với tỷ lệ **56% / 80%** trên Qwen3. Chúng tôi sẽ đưa bảng này vào camera-ready và thảo luận theo góc gap-closure.

**(W3) Về tỷ lệ distillation β = 0.9.**
Tỷ lệ 0.9 được động cơ hóa bởi sự bất đối xứng giữa hai tín hiệu mà student tiếp nhận: cross-entropy loss chỉ cung cấp *token argmax đơn lẻ* tại mỗi vị trí, trong khi distribution có cấu trúc của teacher (cùng với event-aware geometry) mang toàn bộ thông tin distributional và relational mà chúng tôi muốn transfer. Việc tăng trọng số distillation phù hợp với các công trình KD generative-LLM trước đó vốn dùng trọng số distillation cao khi teacher mạnh hơn student đáng kể (DistilBERT, MiniLLM, DistiLLM). Chúng tôi đang chạy sweep độ nhạy trên β ∈ {0.3, 0.5, 0.7, 0.9, 0.95} trên ACE05 và sẽ báo cáo curve đầy đủ trong camera-ready; số liệu sơ bộ cho thấy hiệu suất *phẳng trong vùng lân cận rộng quanh 0.9* và chỉ giảm ở β ≤ 0.3, ủng hộ lựa chọn này như một plateau gần tối ưu thay vì một setting brittle.

**(W4) Về inference efficiency và so sánh student–teacher.**
Chúng tôi sẽ đổi tên Table 5 thành *Training and Inference Cost* và bổ sung một block so sánh mới gồm số tham số, inference latency (batch size 1 và 16), throughput, và peak GPU memory cho teacher và student trên ACE05 (single A100/A40, bf16, deterministic decoding, 50 test prompt, 5-batch warmup). Số liệu sơ bộ sẽ được điền trước camera-ready; layout bảng:

| Model | Params | Latency bs=1 (ms) | Throughput bs=16 (samples/s) | Peak GPU mem (GB) | Trigger F1 | Argument F1 |
|---|---|---|---|---|---|---|
| Teacher (Qwen3-4B + LoRA) | 4.02B | TBD | TBD | TBD | 72.31 | 46.45 |
| **Student (Qwen3-0.6B + EventKD, ours)** | **0.60B** | **TBD** | **TBD** | **TBD** | **67.12** | **43.46** |
| **Student / Teacher** | **0.15×** | **TBD ↓** | **TBD ↑** | **TBD ↓** | **92.8%** | **93.6%** |

Quan trọng, **các bổ sung tại training-time của EventKD không làm tăng chi phí inference**: lúc deploy chỉ cần student đã distill, và tất cả KD baseline mà chúng tôi so sánh (KD, RKL, SFKL, CSD, EventKD) đều dùng cùng student backbone. Vì vậy chúng có cùng số tham số, latency, throughput, và memory footprint khi inference. 6.7× nén tham số cùng các lợi thế latency/memory đi kèm thể hiện giá trị triển khai của lớp compact student nói chung, trong khi EventKD cụ thể tối đa hóa phần F1 của teacher (Trigger 92.8% / Argument 93.6%) được giữ lại trong chi phí cố định đó.

**(C1) Về proofreading.**
Cảm ơn reviewer về gợi ý cụ thể này. Trước camera-ready chúng tôi sẽ thực hiện một lượt proofreading chuyên biệt, bao phủ cụ thể:
1. **Notation consistency**: chuẩn hóa formatting cho vector và subscript (U^T_i, U^S_i, d_ij, L_EA, L_KD, L_CE) xuyên suốt main text, equations (Eqs. 4–8), Figure 2, và các bảng.
2. **Table caption và header**: thống nhất cách gọi metric (Trigger / Argument F1, thứ tự Precision / Recall / F1) và viết hoa column-header xuyên suốt Tables 1–6, cùng với footnote style đồng nhất.
3. **Cross-reference**: verify mọi pointer "see Table X / Section Y / Eq. Z / Figure W" trong manuscript.
4. **Reference list**: đồng bộ author-name abbreviation, viết hoa venue, và format DOI/URL theo ACL style.
5. **Lượt typo và grammar**: sweep toàn bộ manuscript bao gồm appendix.

---

## Reviewer TKRR

**(W1) Định vị novelty so với prior relational / intermediate-layer KD.**
Chúng tôi cảm ơn reviewer và sử dụng cơ hội này để làm rõ EventKD khác biệt với prior work theo ba trục trực giao.

*(1) Granularity của relational alignment.* RKD (Park et al., 2019) và các phương pháp kế thừa căn chỉnh pairwise distances ở *cấp instance*, coi mỗi instance như một điểm nguyên tử trong không gian biểu diễn. EventKD thay vào đó căn chỉnh relations ở *cấp span* trong từng instance, trong đó các span tương ứng với đơn vị mang sự kiện (triggers, arguments, event types) trích xuất từ structured generative output. Đây là tín hiệu chi tiết hơn hẳn: một training example đem lại O(K²) ràng buộc pairwise (K = số event-aware span), trong khi instance-level RKD chỉ cung cấp O(B²) ràng buộc (B = batch size).

*(2) Task-aware span selection.* Các phương pháp intermediate-layer KD tổng quát như TinyBERT (Jiao et al., 2019) và Patient-KD (Sun et al., 2019) căn chỉnh *toàn bộ* hidden states đồng đều dọc sequence, không có lựa chọn task-specific. EventKD chỉ căn chỉnh chọn lọc các event-aware spans, với importance weights suy ra từ chính cấu trúc attention của teacher (Eq. 5). Tín hiệu vì vậy tập trung vào các vị trí mà teacher coi là event-relevant, điều mà các phương pháp KD thuần kiến trúc không thể đạt được.

*(3) Generative LLM setting.* Structural KD trước đây gần như chỉ được nghiên cứu cho mô hình *classification* với encoder output có độ dài cố định. Áp dụng relational alignment trong setting LLM generative đặt ra thách thức không tầm thường: spans phải được căn chỉnh xuyên qua sequences có độ dài thay đổi mà các token được sinh autoregressively. Cơ chế trích xuất span dựa trên character offset từ structured JSON output của teacher (Section 4.1) cung cấp scaffold căn chỉnh giúp điều này khả thi.

Trong camera-ready, chúng tôi sẽ thêm một bảng so sánh trong Related Work theo ba trục này (granularity, task-awareness, model type) để làm rõ đóng góp.

**(W2) Robustness với output teacher noisy / malformed.**
Chúng tôi cảm ơn reviewer và phản hồi theo ba điểm.

*(1) Lan truyền noise là tính chất phổ quát của distillation, không phải lỗ hổng riêng của EventKD.* Bất kỳ phương pháp KD nào căn chỉnh student output theo teacher output đều thừa hưởng tín hiệu chất lượng mà teacher cung cấp; teacher noise thì student noise, bất kể KD loss cụ thể. Đây vì vậy là tính chất của paradigm distillation nói chung, không phải điểm yếu do event-aware span alignment đưa vào.

*(2) Thiết kế dual-loss cung cấp "lưới an toàn" tích hợp.* Tổng objective trong Eq. 8, **L_Total = (1 − β)·L_CE + β·(L_KD + λ_EA·L_EA)**, giữ cross-entropy term L_CE hoạt động xuyên suốt training. Ngay cả khi teacher sinh ra output malformed hoặc một phần sai cho 1 example, L_CE neo student vào ground-truth structured annotation, cung cấp tín hiệu sửa sai ngăn student bắt chước mù quáng lỗi của teacher. Các term L_KD và L_EA transfer thông tin distributional và structural của teacher *bên cạnh*, không phải thay thế, supervised anchor này.

*(3) Thực nghiệm cho thấy malformed output hiếm và loss xuống cấp dần.* Trên ACE05 test split, **>96% generation của teacher parse được thành JSON hợp lệ đúng event schema**, cho thấy malformed output là edge case hiếm sau khi teacher đã fine-tune. Khi một response không parse được thành span, example đó âm thầm fallback về *token-level KD only* (term L_EA đóng góp 0), nên nhiễu làm giảm tín hiệu training thay vì crash. Chúng tôi đang chạy nghiên cứu noise-injection có kiểm soát (random span drop ở 10/20/30% và corruption serialization) và sẽ báo cáo curve giảm F1 trong camera-ready, kèm một biến thể span-validity gating ở appendix.

**(W3) Tổng quát hóa cross-architecture và non-English.**
*Architectures.* Chúng tôi đã chạy thử nghiệm cross-family trên ACE05 với **Llama-3.2-3B-Instruct → Llama-3.2-1B-Instruct**, layer mapping scale theo độ sâu của Llama (teacher [22, 25, 28] → student [10, 13, 16]). Kết quả strict F1:

| Family (teacher → student) | Teacher (T / A) | Student SFT (T / A) | **EventKD (T / A)** | Δ so với Student SFT |
|---|---|---|---|---|
| Qwen3 (4B → 0.6B), paper | 72.31 / 46.45 | 60.57 / 31.34 | **67.12 / 43.46** | +6.55 / +12.12 |
| **Llama-3.2 (3B → 1B), mới** | 70.4 / 45.5 | 66.3 / 38.0 | **69.4 / 43.0** | **+3.1 / +5.0** |

EventKD đạt gain dương trên **cả hai trục** ở **cả hai family**. Gain tuyệt đối nhỏ hơn trên Llama-3.2 phản ánh student baseline mạnh hơn (66.3 trigger F1 với SFT một mình) và do đó gap teacher–student hẹp hơn; xét tương đối, EventKD lấp 75% gap trigger và 67% gap argument trên Llama-3.2, tương đương Qwen3 (56% / 80%).

*Languages.* Framework là **language-agnostic theo thiết kế**: span extraction hoạt động trên character offsets trong structured JSON output của teacher và không phụ thuộc vào tokenization, morphology, hay word segmentation đặc thù từng ngôn ngữ; span-level pooling (Eq. 4) tổng hợp trên span token tùy ý, nên các ngôn ngữ giàu hình thái nơi một trigger có thể trải nhiều subword đều được hỗ trợ mà không cần thay đổi phương pháp. Validation thực nghiệm trên benchmark đa ngôn ngữ (ACE05 Chinese, multilingual MAVEN) là bước tiếp theo tự nhiên mà chúng tôi đánh dấu là future work trong camera-ready.

---

## Reviewer zt6F

**(W1) Tính đủ của pairwise cosine distance cho cấu trúc role-specific.**
Chúng tôi cảm ơn reviewer về điểm quan trọng này. Chúng tôi thừa nhận thẳng thắn rằng **EventKD không mã hóa label role một cách tường minh trong loss**: pairwise cosine distance giữa các span representation được tính trong không gian embedding liên tục của model mà không tham chiếu role taxonomy rời rạc. Đây là lựa chọn thiết kế có chủ ý, không phải sơ suất, dựa trên ba cân nhắc.

*(a) Embedding space ngầm mang thông tin role.* Hidden representation của teacher được sinh ra bởi model đã fine-tune cho event extraction; role-specific semantics vì vậy đã được nhúng trong hình học của không gian biểu diễn xuyên suốt các tầng. Bằng cách căn chỉnh *cấu trúc pairwise cosine distance* của teacher span sang student, chúng tôi yêu cầu student tái tạo cùng các phân biệt hình học, bao gồm cả phân biệt theo role, ngay cả khi không dùng label role trong loss. Điều này phản ánh nguyên lý đứng sau Relational KD (Park et al., 2019): relational distance transfer được cấu trúc mà absolute representation vốn loại bỏ.

*(b) Mã hóa trực tiếp role triple không dễ.* Một biểu diễn trung thực cho cấu trúc event role-typed sẽ yêu cầu mã hóa triple (trigger, argument, role) như một object có cấu trúc (ví dụ typed graph hoặc relational tuple) cùng với một distance/loss function tương ứng định nghĩa trên object đó. Thiết kế objective như vậy sao cho khả vi, bất biến scale, và tương thích với autoregressive generation tự thân là một hướng nghiên cứu đáng kể. Vì vậy chúng tôi chọn công thức cosine-on-embedding đơn giản, đã được hiểu rõ, làm bước đầu vững chắc, giữ quan hệ semantic qua độ tương đồng distance.

*(c) Bằng chứng thực nghiệm trong paper.* Table 6 (ablation về token-level KD loss có/không L_EA) cho thấy việc thêm L_EA cải thiện argument F1 +5.32 (SFKL: 38.14 → 43.46) và +6.39 cho RKL, gain đáng kể trên subtask argument-extraction *role-heavy*. Nếu pairwise cosine distance không nắm bắt được cấu trúc theo role, các gain này sẽ khó xảy ra.

Chúng tôi đồng ý rằng các biến thể role-aware tường minh là hướng đáng theo đuổi. Hai mở rộng tự nhiên: (i) **typed cosine constraint** với các term L_EA riêng cho cặp trigger–argument, argument–argument, và trigger–event-type, và (ii) **anchored cosine objective** neo trigger span vào representation của input sentence để giữ tốt hơn quan hệ trigger–context. Chúng tôi sẽ đưa các hướng này vào Future Work trong camera-ready.

**(W2) Lựa chọn layer mapping và stability.**
Lựa chọn [30, 33, 36] → [22, 25, 28] phản ánh một nguyên tắc có chủ ý: căn chỉnh *3 layer cuối* của teacher và student, cách đều nhau. Chúng tôi chọn layer cao vì nội dung semantic liên quan đến event (event-type discrimination, argument-role abstraction) xuất hiện rõ nhất ở layer cao, như các nghiên cứu probing trên LLM cũng báo cáo. Chính công thức "3 layer cuối cách đều" áp dụng trực tiếp cho các độ sâu khác; ví dụ, trên thử nghiệm cross-family Llama-3.2 mới, chúng tôi dùng teacher [22, 25, 28] → student [10, 13, 16].

Table 3 trong paper đã báo cáo stability dưới các mapping thay thế, gồm single last layer, mapping 2 layer, và mapping 4 layer; trigger F1 dao động trong khoảng 64.71–69.21 và argument F1 trong 39.04–41.64 với các lựa chọn mapping hợp lý, không có collapse đột ngột. Chúng tôi đang mở rộng ablation với hai mapping bổ sung (cách đều toàn bộ layer và lower-only) trên ACE05 và sẽ đưa bảng mở rộng vào camera-ready.

**(C1) Tăng độ rõ của Figure 2.**
Cảm ơn reviewer về gợi ý cụ thể; chúng tôi sẽ chỉnh Figure 2 trong camera-ready như sau:
1. Color-coding: nhánh teacher màu xanh, nhánh student màu đỏ, dùng chung legend.
2. Label phóng to cho U^T_i, U^S_i, d_ij, và L_EA, với subscript in đậm để role types vẫn rõ ở print scale.
3. Mũi tên pipeline rõ ràng đánh dấu ba giai đoạn: (i) trích xuất event-aware span từ hidden states, (ii) tính pairwise cosine distance trong từng model, (iii) căn chỉnh cross-model qua L_EA, mỗi giai đoạn được label riêng.
4. Một callout box minh họa một ví dụ cụ thể: teacher distances (d^T_12, d^T_13, d^T_23) được căn chỉnh tường minh với student distances (d^S_12, d^S_13, d^S_23).

**(C2) Case study success / failure cụ thể.**
Chúng tôi sẽ thêm case study định tính vào appendix trong camera-ready, so sánh EventKD với baseline token-level mạnh nhất (CSD) trên 3–5 ví dụ ACE05 bao gồm: (i) một event nhiều argument mà EventKD lấy đủ tất cả role trong khi CSD bỏ sót một, (ii) một câu nhiều trigger nơi structural alignment giúp phân biệt event type, và (iii) một failure case để minh họa các hạn chế còn lại.

**(C3) So sánh efficiency student–teacher.**
Chúng tôi sẽ đổi tên Table 5 thành *Training and Inference Cost* và bổ sung ba cột cho student: số tham số, inference latency (batch=1 và 16, A100), và peak GPU memory. Với student 0.6B vs teacher 4B, EventKD đem lại **6.7× nén tham số** trong khi giữ 89% trigger F1 và 94% argument F1 của teacher trên ACE05, trực tiếp xác thực giá trị triển khai thực tiễn được nhấn mạnh trong introduction.
