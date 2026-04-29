# Bản dịch tiếng Việt: câu trả lời cho từng reviewer

---

## Reviewer PGzb

**(W1) Về cross-event relations và cấu trúc graph.**
Chúng tôi cảm ơn reviewer về gợi ý này. Để làm rõ phạm vi: Section 4.1 định khung *intra-event graph* (trigger ↔ event type ↔ arguments ↔ argument roles) là đơn vị cấu trúc mà phương pháp của chúng tôi nắm bắt, và bản thân điều này đã không tầm thường, phần lớn bị bỏ qua bởi token-level KD. Câu mà reviewer trích (L299) đề cập đến intra-event graph này, không phải cross-event relations.

Dù vậy, chúng tôi đồng ý rằng cross-event relations (coreferent triggers, temporal/causal relations, document-level event structure) là một mở rộng có giá trị. Quan trọng là, **framework EventKD tổng quát hóa được sang setting này**: thay đổi duy nhất cần thực hiện là mở rộng tập event-aware spans để bao gồm cross-document hoặc cross-sentence event mentions; pairwise distance loss (Eq. 7) hoạt động không đổi trên tập span đã mở rộng. Chúng tôi sẽ làm rõ tính mở rộng này trong camera-ready và đưa cross-event-relation transfer vào phần future work.

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
Chúng tôi sẽ đổi tên Table 5 thành *Training and Inference Cost* và bổ sung ba cột cho student model: số tham số, inference latency tại batch size 1 và 16, và peak GPU memory tại inference, đo trên cùng một A100. Với student 0.6B vs teacher 4B, EventKD đem lại **6.7× nén tham số** trong khi giữ 89% trigger F1 và 94% argument F1 của teacher trên ACE05 (Table 1/2 của paper). Điều này trực tiếp giải quyết giá trị triển khai thực tiễn của student compact mà chúng tôi nhấn mạnh trong introduction.

**(C1) Về proofreading.**
Cảm ơn reviewer; chúng tôi sẽ thực hiện một lượt proofreading kỹ và đồng nhất hóa formatting, table caption, và figure references trong camera-ready.

---

## Reviewer TKRR

**(W1) Định vị novelty so với prior relational / intermediate-layer KD.**
Chúng tôi cảm ơn reviewer và sử dụng cơ hội này để làm rõ EventKD khác biệt với prior work theo ba trục trực giao.

*(1) Granularity của relational alignment.* RKD (Park et al., 2019) và các phương pháp kế thừa căn chỉnh pairwise distances ở *cấp instance*, coi mỗi instance như một điểm nguyên tử trong không gian biểu diễn. EventKD thay vào đó căn chỉnh relations ở *cấp span* trong từng instance, trong đó các span tương ứng với đơn vị mang sự kiện (triggers, arguments, event types) trích xuất từ structured generative output. Đây là tín hiệu chi tiết hơn hẳn: một training example đem lại O(K²) ràng buộc pairwise (K = số event-aware span), trong khi instance-level RKD chỉ cung cấp O(B²) ràng buộc (B = batch size).

*(2) Task-aware span selection.* Các phương pháp intermediate-layer KD tổng quát như TinyBERT (Jiao et al., 2019) và Patient-KD (Sun et al., 2019) căn chỉnh *toàn bộ* hidden states đồng đều dọc sequence, không có lựa chọn task-specific. EventKD chỉ căn chỉnh chọn lọc các event-aware spans, với importance weights suy ra từ chính cấu trúc attention của teacher (Eq. 5). Tín hiệu vì vậy tập trung vào các vị trí mà teacher coi là event-relevant, điều mà các phương pháp KD thuần kiến trúc không thể đạt được.

*(3) Generative LLM setting.* Structural KD trước đây gần như chỉ được nghiên cứu cho mô hình *classification* với encoder output có độ dài cố định. Áp dụng relational alignment trong setting LLM generative đặt ra thách thức không tầm thường: spans phải được căn chỉnh xuyên qua sequences có độ dài thay đổi mà các token được sinh autoregressively. Cơ chế trích xuất span dựa trên character offset từ structured JSON output của teacher (Section 4.1) cung cấp scaffold căn chỉnh giúp điều này khả thi.

Trong camera-ready, chúng tôi sẽ thêm một bảng so sánh trong Related Work theo ba trục này (granularity, task-awareness, model type) để làm rõ đóng góp.

**(W2) Robustness với output teacher noisy / malformed.**
Chúng tôi chia sẻ mối quan ngại này và đã bắt đầu phân tích robustness trên ACE05. (i) Trên các generation thô của teacher trên ACE05 test split, **>96% output parse được dưới dạng JSON hợp lệ tuân theo event schema mong đợi**, cho thấy malformed output là edge case hiếm khi teacher đã được fine-tune. (ii) Trong setup của chúng tôi, khi một response không parse được thành span, example đó âm thầm fallback về *token-level KD only* (term L_EA đóng góp 0), nên nhiễu làm tín hiệu training xuống cấp dần thay vì crash. (iii) Chúng tôi đang chạy nghiên cứu noise-injection có kiểm soát (random span drop ở 10/20/30% và corruption serialization) và sẽ báo cáo curve giảm F1 trong camera-ready. Mitigation dưới dạng span-validity gating cũng dễ dàng thêm vào; chúng tôi thảo luận điều này trong phần Limitations chỉnh sửa.

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
Chúng tôi đồng ý rằng thông tin role là một khía cạnh quan trọng của event structure, và cảm ơn gợi ý. Chúng tôi lập luận rằng pairwise cosine distance vẫn là một công thức hiệu quả và có cơ sở vững chắc, vì ba lý do.

*(a) Bảo toàn distance ngầm mã hóa thông tin role.* Hidden representations của teacher đã mã hóa role-specific semantics, đó chính là điều mà fine-tuning trên event extraction tạo ra. Bằng cách matching *cấu trúc pairwise distance* của teacher spans, student bị buộc tái tạo các phân biệt role này theo hình học, ngay cả khi loss không có role label tường minh. Đây cũng là nguyên lý đem lại hiệu suất thực nghiệm mạnh cho Relational KD (Park et al., 2019): relational distances *transfer* được cấu trúc mà absolute representations vốn đã loại bỏ.

*(b) Robustness với capacity gap.* Matching trực tiếp từng span representation (ví dụ MSE per-role) sẽ buộc student tái tạo feature teacher cao chiều một cách tuyệt đối, một bài toán nổi tiếng khó khi capacity gap lớn. Pairwise distance alignment bất biến với rotation/scaling của representation và chỉ yêu cầu giữ cấu trúc *tương đối*, điều này thực nghiệm cho thấy student nhỏ dễ thỏa mãn hơn.

*(c) Bằng chứng thực nghiệm trong paper.* Table 6 (ablation về token-level KD loss với/không L_EA) cho thấy việc thêm L_EA cải thiện argument F1 +5.32 (SFKL: 38.14 → 43.46) và +6.39 cho RKL, gain đáng kể trên subtask *role-heavy*. Nếu pairwise distance không nắm bắt được cấu trúc role, gain này sẽ khó xảy ra.

Một mở rộng tự nhiên là enforce role-typed distance constraints (ví dụ loss term riêng cho cặp trigger–argument vs argument–argument); chúng tôi đồng ý đây là hướng đáng theo đuổi và sẽ liệt kê làm Future Work.

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
