Official Review of Submission1211 by Reviewer PGzb
Summary Of Weaknesses:
The authors state in Line 299 that "events possess an intrinsic graph-like structure." Given that a significant body of existing research focuses on cross-event relations, have the authors considered incorporating these aspects into the framework?
Beyond the Qwen family, have the authors experimented with other LLM families for the teacher-student configuration?
What is the theoretical rationale for setting the distillation ratio to 0.9? Furthermore, are there any hyper-parameter sensitivity analyses to justify this specific value?
Inference efficiency is a crucial dimension of knowledge distillation evaluation. Why does Table 5 only present training costs? Additionally, while the authors emphasize a "compact student model," there appears to be a lack of direct performance and efficiency comparisons between the student and teacher models.
Comments Suggestions And Typos:
The overall presentation could be further improved. It is recommended that the authors conduct a thorough proofreading to ensure consistent formatting and to clarify any minor typographical errors throughout the manuscript.

Answer:
The authors state in Line 299 that "events possess an intrinsic graph-like structure." Given that a significant body of existing research focuses on cross-event relations, have the authors considered incorporating these aspects into the framework?
Định nghĩa cross event relations và phân biệt với EE, bài mình chỉ làm focus với EE, chưa làm tới cross event
Cross event làm trong tương lai, phân tích phương pháp của mình có dễ áp dụng cho cross event không
Cảm ơn đã gợi ý

Beyond the Qwen family, have the authors experimented with other LLM families for the teacher-student configuration?
We have run a cross-family experiment on ACE05 using Llama-3.2-3B-Instruct as teacher and Llama-3.2-1B-Instruct as student. The pipeline is identical to our Qwen3 setup (LoRA SFT for the teacher, EventKD with SFKL + L_EA for the student) except for the layer-mapping indices, which are scaled proportionally to the depth of Llama-3.2: teacher layers [22, 25, 28] → student layers [10, 13, 16] (last three evenly spaced layers, matching the pattern used for Qwen).

Results on ACE05 test (strict trigger / argument F1):

| Family (teacher → student) | Teacher F1 (T / A) | Student SFT (T / A) | **EventKD (T / A)** | Δ over Student SFT |
|---|---|---|---|---|
| Qwen3 (4B → 0.6B), paper | 72.31 / 46.45 | 60.57 / 31.34 | **67.12 / 43.46** | +6.55 / +12.12 |
| **Llama-3.2 (3B → 1B), new** | 70.4 / 45.5 | 66.3 / 38.0 | **69.4 / 43.0** | **+3.1 / +5.0** |

EventKD improves over the student-only SFT baseline in both families on both trigger and argument F1, confirming that the gain is not specific to the Qwen family. The smaller absolute Δ on Llama-3.2 reflects a stronger baseline (Llama-3.2-1B SFT alone reaches 66.3 trigger F1, well above Qwen3-0.6B SFT at 60.57) and therefore a narrower student–teacher gap (≈4 trigger / 7.5 argument F1 on Llama vs ≈12 / 15 on Qwen). Relative to the available headroom, EventKD closes 75% of the trigger gap and 67% of the argument gap on Llama-3.2, comparable to the 56% / 80% closure on Qwen3. We will include this table in the camera-ready and discuss the gap-closure framing.

Thêm 1 family khác nữa: mistral?
Thêm baseline: csd và distillm

What is the theoretical rationale for setting the distillation ratio to 0.9? Furthermore, are there any hyper-parameter sensitivity analyses to justify this specific value?
Bổ sung bảng sensitivity trên 3 data

Inference efficiency is a crucial dimension of knowledge distillation evaluation. Why does Table 5 only present training costs? Additionally, while the authors emphasize a "compact student model," there appears to be a lack of direct performance and efficiency comparisons between the student and teacher models.
Thêm bảng cost infer, so student = xx% teacher
Tất cả phương pháp là giống nhau, không tăng chi phí infer = base student model

The overall presentation could be further improved. It is recommended that the authors conduct a thorough proofreading to ensure consistent formatting and to clarify any minor typographical errors throughout the manuscript.
We thank the reviewer; we will conduct a thorough proofreading pass and harmonize formatting, table captions, and figure references in the camera-ready.
Plan sẽ sửa/ phân tích/ format

Official Review of Submission1211 by Reviewer TKRR
Summary Of Weaknesses:
The novelty appears somewhat incremental over prior relational or intermediate-layer distillation methods, which have already explored transferring structural knowledge. The paper would be stronger if it more clearly articulated how event-aware span alignment differs conceptually and technically from these existing approaches, thereby better positioning its contribution.
The method appears sensitive to the teacher’s structured output and the chosen serialization format, since event-aware spans are extracted from generated outputs; this raises robustness concerns when the teacher's output is noisy or malformed.
The experiments are limited to a single teacher–student pair within the same model family and to English benchmarks, which raises concerns about the generalizability of the proposed method across architectures and languages.
Comments Suggestions And Typos:
To strengthen the paper, the authors could better clarify how EventKD differs from prior relational or intermediate-layer distillation methods, provide robustness analysis under noisy or malformed teacher outputs and different serialization formats, and extend experiments to more diverse teacher–student pairs and non-English settings. These additions would improve the paper’s positioning, reliability, and generalizability.
Answer:
The novelty appears somewhat incremental over prior relational or intermediate-layer distillation methods, which have already explored transferring structural knowledge. The paper would be stronger if it more clearly articulated how event-aware span alignment differs conceptually and technically from these existing approaches, thereby better positioning its contribution.
Thêm thử nghiệm và phân tích khác biệt của event aware span

The method appears sensitive to the teacher’s structured output and the chosen serialization format, since event-aware spans are extracted from generated outputs; this raises robustness concerns when the teacher's output is noisy or malformed.
Nhấn mạnh đây là vấn đề chung của distill, mục tiêu của distill là để student gen ra giống teacher, teacher noise -> student noise
Tuy nhiên vẫn Có 2 loss: sft và distill
Loss sft vẫn giúp kiểm soát output cho student, để có thể correct lại output của student 

The experiments are limited to a single teacher–student pair within the same model family and to English benchmarks, which raises concerns about the generalizability of the proposed method across architectures and languages.
Thử nghiệm với ngôn ngữ khác
Model hiện đại đều multilingual
Rebutal hơi gấp nên có thể sẽ future work

To strengthen the paper, the authors could better clarify how EventKD differs from prior relational or intermediate-layer distillation methods, provide robustness analysis under noisy or malformed teacher outputs and different serialization formats, and extend experiments to more diverse teacher–student pairs and non-English settings. These additions would improve the paper’s positioning, reliability, and generalizability.




Official Review of Submission1211 by Reviewer zt6F
Summary Of Weaknesses:
The proposed Event-Aware loss aligns pairwise cosine distances among event-aware spans. However, structural relations in event extraction often involve role-specific constraints rather than simple relative distance, and the paper does not yet fully clarify whether this formulation is sufficient to capture such information. 
The method aligns only a small subset of intermediate layers, such as mapping teacher layers 30/33/36 to student layers 22/25/28 on ACE05. Since the choice of layer mapping may substantially affect performance, the current explanation of why these mappings were selected and whether the method is stable under alternative mappings remains somewhat limited.
Comments Suggestions And Typos:
Figure 2 illustrates the core Event-Aware distillation mechanism, but the teacher and student branches are not sufficiently distinguishable. Enlarging the key labels and more clearly marking the span representation extraction, pairwise cosine distance computation, and LEA alignment steps would improve the figure’s clarity. 
Tables 1 and 2 show that EventKD consistently outperforms the baselines, but the paper currently provides only aggregate score comparisons. Adding a few concrete success and failure cases, especially in comparison with standard KD, would make the empirical gains more convincing. 
It would be helpful to provide a clearer comparison between the student and teacher models in terms of parameter size, inference cost, or deployment efficiency, as this would better highlight the practical value of the proposed method.
Answer:
The proposed Event-Aware loss aligns pairwise cosine distances among event-aware spans. However, structural relations in event extraction often involve role-specific constraints rather than simple relative distance, and the paper does not yet fully clarify whether this formulation is sufficient to capture such information.
 Hiện tại thì để đơn giản, chúng tôi không mã hoá thông tin role trong quan hệ giữa trigger với spans, vì vector thể hiện thông tin semantic trong không gian embeddings nên sử dụng cosine để đo độ tương đồng trong không gian embeddings của student phải giống trong không gian teacher. Không sử dụng trực tiếp thông tin role nhưng tin rằng khi biểu diễn trong không gian embedding thì vẫn có các thông tin trong các tầng biểu diễn khác nhau và hi vọng cosine sẽ bắt được những thông tin này
Thông tin của label chưa được sử dụng
(tìm phân tích để biểu diễn role)
Việc mã hoá bộ ba này là không dễ dàng nên để đơn giản chúng tôi chọn không gian embeddings, ở đây có gắng giữ quan hệ ngữ nghĩa thông qua độ đo tương đồng cosine
Cosine: trigger - arg, trigger - input sentence

The method aligns only a small subset of intermediate layers, such as mapping teacher layers 30/33/36 to student layers 22/25/28 on ACE05. Since the choice of layer mapping may substantially affect performance, the current explanation of why these mappings were selected and whether the method is stable under alternative mappings remains somewhat limited.



Figure 2 illustrates the core Event-Aware distillation mechanism, but the teacher and student branches are not sufficiently distinguishable. Enlarging the key labels and more clearly marking the span representation extraction, pairwise cosine distance computation, and LEA alignment steps would improve the figure’s clarity. 
Thêm ảnh vào chat hỏi

Tables 1 and 2 show that EventKD consistently outperforms the baselines, but the paper currently provides only aggregate score comparisons. Adding a few concrete success and failure cases, especially in comparison with standard KD, would make the empirical gains more convincing. 
Thêm ví dụ và phân tích
Lấy trong data ra xem

It would be helpful to provide a clearer comparison between the student and teacher models in terms of parameter size, inference cost, or deployment efficiency, as this would better highlight the practical value of the proposed method.





