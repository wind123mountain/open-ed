#### Official Review of Submission1211 by Reviewer PGzb

**Paper Summary:**

To address the limitations of existing knowledge distillation methods, this paper proposes EventKD, a two-level knowledge distillation framework specifically designed for event extraction. The framework incorporates token-level and span-level distillation. Evaluations across three event extraction benchmarks, using Qwen3-4B as the teacher model and Qwen3-0.6B as the student, demonstrate that the proposed framework consistently outperforms seven baseline models, including SFT.

**Summary Of Strengths:**

1. The paper identifies two key limitations of current knowledge distillation methods: the inability to focus on event-related tokens and the failure to capture dependencies between event components.
2. The framework employs token-level distillation to preserve the output distribution of the teacher model, while span-level distillation is used to maintain the geometric relationships between event-aware spans.
3. Evaluations conducted on three datasets against seven baseline models show that the proposed method achieves state-of-the-art results across all metrics.

**Summary Of Weaknesses:**

1. The authors state in Line 299 that "events possess an intrinsic graph-like structure." Given that a significant body of existing research focuses on cross-event relations, have the authors considered incorporating these aspects into the framework?
2. Beyond the Qwen family, have the authors experimented with other LLM families for the teacher-student configuration?
3. What is the theoretical rationale for setting the distillation ratio to 0.9? Furthermore, are there any hyper-parameter sensitivity analyses to justify this specific value?
4. Inference efficiency is a crucial dimension of knowledge distillation evaluation. Why does Table 5 only present training costs? Additionally, while the authors emphasize a "compact student model," there appears to be a lack of direct performance and efficiency comparisons between the student and teacher models.

**Comments Suggestions And Typos:**

The overall presentation could be further improved. It is recommended that the authors conduct a thorough proofreading to ensure consistent formatting and to clarify any minor typographical errors throughout the manuscript.

**Confidence:** 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.

**Soundness:** 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.

**Excitement:** 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.

**Overall Assessment:** 3 = Findings: I think this paper could be accepted to the Findings of the ACL.

**Ethical Concerns:**

There are no concerns with this submission

**Reproducibility:** 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.

**Datasets:** 1 = No usable datasets submitted.

**Software:** 1 = No usable software released.

**Knowledge Of Or Educated Guess At Author Identity:** No

**Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources

**Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.



#### Official Comment by Authors

**Comment:**

We would first like to thank you for your constructive and thoughtful feedback. Below, we provide our responses to the concerns and questions you raised.

**W1. The authors state in Line 299 that "events possess an intrinsic graph-like structure." Given that a significant body of existing research focuses on cross-event relations, have the authors considered incorporating these aspects into the framework?**

> We thank the reviewer for the insightful question. We would like to clarify the scope of our statement in Line 299 and how it relates to the cross-event relation literature.

> **Scope clarification.** Our work focuses on event extraction (EE), including trigger identification, event type classification, and argument extraction within a single event mention. The *"graph-like structure"* mentioned in L299 refers specifically to the intra-event structure (i.e., trigger–type–argument–role relations), which is the target of distillation in EventKD.

> Meanwhile, *Cross-event relation extraction (ERE)* is a distinct downstream task that predicts pairwise relations between already-extracted events, typically along the four canonical dimensions formalized by MAVEN-ERE \[1\]: *event coreference*, *temporal* (Before / Overlap / Contains / etc.), *causal* (Cause / Precondition), and *subevent* (Part-of). Standard pipelines run EE first and then feed the extracted events into ERE.

> **Relationship of EventKD to ERE.** Since ERE consumes the output of EE as input, *improving the EE component, especially at compact-model scale, directly benefits any downstream ERE system* that adopts such a student. This is a particularly relevant deployment scenario: ERE typically reasons over O(N²) event pairs per document, where inference cost of a 4B-class teacher quickly becomes the bottleneck. A 0.6B distilled student that retains 92.8% / 93.6% of teacher Trigger / Argument F1 (Tables 1 and 2) therefore makes ERE pipelines substantially more practical.

> **Extensibility.** Our framework can be naturally extended to incorporate cross-event signals. Specifically, the event-aware span set can be expanded to include spans across multiple events within a document, allowing the pairwise distillation objective (Eq. 7) to capture inter-event structure. To further model typed relations (e.g., Before, Cause), additional supervision (e.g., relation labels from the teacher) would be required. We mark *joint EE-ERE distillation* as a natural extension and will discuss it, together with citations to MAVEN-ERE and related ERE work, as future work in the camera-ready.

> **References:**
> 
> > \[1\] Wang, X., Chen, Y., Ding, N., Peng, H., Wang, Z., Lin, Y.,... & Zhou, J. (2022, December). Maven-ere: A unified large-scale dataset for event coreference, temporal, causal, and subevent relation extraction. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (pp. 926-941).



#### Official Comment by Authors

**Comment:**

**W2. Beyond the Qwen family, have the authors experimented with other LLM families for the teacher-student configuration?**

> We thank the reviewer for raising this important question regarding cross-family generalization. We chose the Qwen family in the main paper because it represents a strong and competitive open-weight LLM family with excellent instruction-following ability, making it a reliable testbed for evaluating the effectiveness of distillation methods. Our goal was to demonstrate that EventKD can preserve structured event knowledge under a high-performing teacher–student setup.

> To address the reviewer’s concern on generalizability, we additionally conduct cross-family experiments on a different LLM family using *Meta-Llama-3-8B-Instruct (teacher)* and *Llama-3.2-1B-Instruct (student)*, with a comparable compression ratio (6.5× vs. 6.7× in Qwen). We keep the pipeline identical to the Qwen setup (LoRA-based SFT for the teacher, followed by EventKD with $S F K L + L_{E A}$ for the student), with span-alignment layers proportionally mapped across model depths. For a fair comparison, we also retrain DistiLLM (SFKL) and CSD under the same setting, sharing the same student backbone and training budget.

> *Table: Cross-family comparison on ACE05*

| Family (teacher → student) | Method | Trigger F1 | Argument F1 |
| --- | --- | --- | --- |
| Qwen3 (4B → 0.6B), paper | Teacher (upper bound) | 72.31 | 46.45 |
|  | Student SFT | 60.57 | 31.34 |
|  | DistiLLM SFKL | 64.87 | 38.14 |
|  | CSD | 64.94 | 36.98 |
|  | **EventKD (ours)** | **67.12** | **43.46** |
| **Llama-3 (8B → 1B), new** | Teacher (upper bound) | 77.23 | 52.46 |
|  | Student SFT | 66.30 | 38.00 |
|  | DistiLLM SFKL | 70.69 | 46.03 |
|  | CSD | 71.25 | 45.32 |
|  | **EventKD (ours)** | **72.75** | **46.84** |

> **Results.** Across both trigger and argument F1 on ACE05, EventKD consistently outperforms the KD baselines under the Llama family as well. In particular, EventKD improves over DistiLLM SFKL by +2.06 trigger / +0.81 argument F1, and over CSD by +1.50 trigger / +1.52 argument F1, while retaining 94.2% (trigger) and 89.3% (argument) of teacher performance under a 6.5× compression ratio.

> These results confirm that the effectiveness of EventKD is not specific to the Qwen family, but generalizes across different LLM architectures. We will include this cross-family comparison in the revised version.

---

**W3. What is the theoretical rationale for setting the distillation ratio to 0.9? Furthermore, are there any hyper-parameter sensitivity analyses to justify this specific value?**

> We thank the reviewer for this important question on the choice of the distillation weight β.

> **Rationale.** The choice of a relatively high distillation weight (β = 0.9) is motivated by the asymmetry of supervision signals: the cross-entropy term provides only the ground-truth token (i.e., a one-hot target), while the teacher distribution, together with its event-aware geometry, encodes richer distributional and structural knowledge that we aim to transfer. Emphasizing the distillation term is therefore aligned with prior KD work for LLMs (e.g., DistilBERT \[1\], MiniLLM \[2\], DistiLLM \[3\]), especially when the teacher is substantially stronger than the student.

> **Sensitivity analysis.** We conduct a sweep over β ∈ {0.1, 0.3, 0.5, 0.7, 0.9, 1.0} on ACE05 using the same Qwen3-0.6B / Qwen3-4B setup, reporting best F1 across 5 epochs:
> 
> | β | 0.1 | 0.3 | 0.5 | 0.7 | **0.9 (paper)** | 1.0 |
> | --- | --- | --- | --- | --- | --- | --- |
> | Trigger F1 | 66.7 | 67.5 | 66.7 | 68.8 | **67.1** | 60.5 |
> | Argument F1 | 40.8 | 41.8 | 42.2 | 40.9 | **43.5** | 30.0 |

> The results show that performance is stable across a broad range (β ∈ \[0.3, 0.9\]), with no sharp degradation, indicating that EventKD is not highly sensitive to this hyperparameter. The choice β = 0.9 yields the best Argument F1 while maintaining competitive Trigger F1. Importantly, performance drops significantly at β = 1.0 (i.e., removing the cross-entropy term), e.g., −13.5 Argument F1 compared to β = 0.9. This confirms that while distillation should be emphasized, the supervised signal remains essential in our dual-loss design (Eq. 8).

> We will include this analysis in the revised version for completeness.

> **References:**
> 
> - \[1\] Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
> - \[2\] Gu, Y., Dong, L., Wei, F., & Huang, M. (2024, May). Minillm: Knowledge distillation of large language models. In The twelfth international conference on learning representations.
> - \[3\] Ko, J., Kim, S., Chen, T., & Yun, S. Y. (2024). DISTILLM: Towards Streamlined Distillation for Large Language Models. Proceedings of Machine Learning Research, 235, 24872-24895.



#### Official Comment by Authors

**Comment:**

**W4. Inference efficiency is a crucial dimension of knowledge distillation evaluation. Why does Table 5 only present training costs? Additionally, while the authors emphasize a "compact student model," there appears to be a lack of direct performance and efficiency comparisons between the student and teacher models.**

> We thank the reviewer for highlighting the importance of inference efficiency. We agree that reporting only training cost in Table 5 is incomplete. In the revised version, we will rename Table 5 to Training and Inference Cost and include a direct comparison between teacher and student models in terms of both efficiency and performance.

> Regarding the concern on inference efficiency comparison, we report parameter count, throughput (tokens/s and samples/s), and peak GPU memory under a controlled setup (single NVIDIA A40, bf16, greedy decoding, batch size 16, fixed-length generation of 128 tokens per sample to ensure fair comparison). Numbers are averaged over 3 runs with different seeds:
> 
> | Model | Params | Tokens/s (bs=16) | Samples/s (bs=16) | Peak mem (bs=16, GB) | Trigger F1 | Argument F1 |
> | --- | --- | --- | --- | --- | --- | --- |
> | Teacher (Qwen3-4B + LoRA SFT) | 4.02B | 151.9 | 1.19 | 9.25 | 72.31 | 46.45 |
> | Student (Qwen3-0.6B + SFT baseline) | 0.60B | 267.0 | 2.09 | 2.17 | 60.57 | 31.34 |
> | **Student (Qwen3-0.6B + EventKD, ours)** | **0.60B** | **264.6** | **2.07** | **2.17** | **67.12** | **43.46** |
> | **Student (EventKD) / Teacher** | **0.15×** | **1.74×** ↑ | **1.74×** ↑ | **0.23×** ↓ | **92.8%** | **93.6%** |

> The results show that EventKD achieves 6.7× parameter compression, 1.74× higher inference throughput, and 4.3× lower memory usage compared to the teacher, while retaining 92.8% (trigger) and 93.6% (argument) of teacher performance. Importantly, EventKD significantly improves over the SFT student baseline (+6.55 / +12.12 F1), demonstrating that the efficiency gains do not come at the cost of task performance.

---

**W5. The overall presentation could be further improved. It is recommended that the authors conduct a thorough proofreading to ensure consistent formatting and to clarify any minor typographical errors throughout the manuscript.**

> We thank the reviewer for this concrete suggestion. Before the camera-ready we will perform a dedicated proofreading pass, covering specifically:
> 
> - **Notation consistency**: harmonized formatting of vectors and subscripts ($U_{i}^{T} , U_{i}^{S} , d_{i j} , L_{E A} , L_{K D} , L_{C E}$) across the main text, equations (Eqs. 4–8), Figure 2, and tables.
> - **Table captions and headers**: consistent metric naming (Trigger / Argument F1, Precision / Recall / F1 ordering) and column-header capitalization across Tables 1–6, plus uniform footnote style.
> - **Cross-references**: verified every "see Table X / Section Y / Eq. Z / Figure W" pointer in the manuscript.
> - **Reference list**: aligned author-name abbreviation, venue capitalization, and DOI/URL formatting following ACL style.
> - **Typo and grammar pass**: full-manuscript sweep including the appendix.



#### Official Review of Submission1211 by Reviewer TKRR

**Paper Summary:**

This paper proposes EventKD, a two-level knowledge distillation framework for generative event extraction:（1）a token-level distillation objective with event-aware span importance weighting;（2）a span-level event-aware loss that aligns the pairwise distance structure among event-aware spans to preserve structural relations among triggers, arguments, and event types. Experiments show consistent improvements over several KD baselines.

**Summary Of Strengths:**

1. The paper is well motivated and focuses on a limitation: KD for generative models mainly transfers token-level distributions, while event extraction inherently involves structured relations among triggers, arguments, and event types.
2. The proposed components are intuitively well motivated, and the paper provides experimental evidence showing that each design choice contributes meaningfully to the overall method and its strong empirical performance.
3. The paper is well structured and easy to follow, with figures and tables used effectively to illustrate the framework, clarify the methodology, and improve the presentation of experimental results.

**Summary Of Weaknesses:**

1. The novelty appears somewhat incremental over prior relational or intermediate-layer distillation methods, which have already explored transferring structural knowledge. The paper would be stronger if it more clearly articulated how event-aware span alignment differs conceptually and technically from these existing approaches, thereby better positioning its contribution.
2. The method appears sensitive to the teacher’s structured output and the chosen serialization format, since event-aware spans are extracted from generated outputs; this raises robustness concerns when the teacher's output is noisy or malformed.
3. The experiments are limited to a single teacher–student pair within the same model family and to English benchmarks, which raises concerns about the generalizability of the proposed method across architectures and languages.

**Comments Suggestions And Typos:**

To strengthen the paper, the authors could better clarify how EventKD differs from prior relational or intermediate-layer distillation methods, provide robustness analysis under noisy or malformed teacher outputs and different serialization formats, and extend experiments to more diverse teacher–student pairs and non-English settings. These additions would improve the paper’s positioning, reliability, and generalizability.

**Confidence:** 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.

**Soundness:** 3 = Acceptable: This study provides sufficient support for its main claims. Some minor points may need extra support or details.

**Excitement:** 2.5

**Overall Assessment:** 3 = Findings: I think this paper could be accepted to the Findings of the ACL.

**Ethical Concerns:**

There are no concerns with this submission

**Reproducibility:** 3 = They could reproduce the results with some difficulty. The settings of parameters are underspecified or subjectively determined, and/or the training/evaluation data are not widely available.

**Datasets:** 1 = No usable datasets submitted.

**Software:** 1 = No usable software released.

**Knowledge Of Or Educated Guess At Author Identity:** No

**Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources

**Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.



#### Official Comment by Authors

**Comment:**

**W1. The novelty appears somewhat incremental over prior relational or intermediate-layer distillation methods, which have already explored transferring structural knowledge. The paper would be stronger if it more clearly articulated how event-aware span alignment differs conceptually and technically from these existing approaches, thereby better positioning its contribution.**

> Positioning EventKD against prior structural KD is an important question, and we clarify the contribution along the following lines.

> **Prior structural KD distills representations uniformly.** Methods such as Multi-Granularity Structural KD (Liu et al., 2022), TinyBERT (Jiao et al., 2019) and Patient-KD (Sun et al., 2019) transfer structural knowledge by aligning hidden states or pairwise interactions *uniformly across all tokens or instances*, treating every position as equally informative. The student is asked to reproduce the teacher's full representational geometry.

> **The capacity gap makes uniform alignment ineffective for compact students.** A well-documented finding in the language model distillation literature is that a small student cannot mimic a much larger teacher's representations in full: when the capacity gap is wide, distillation performance degrades because the student lacks the headroom to absorb every signal the teacher emits (Zhang et al., ACL 2023). Forcing the student to match every token-pair relation therefore wastes its limited capacity on positions that contribute little to the downstream task.

> **EventKD's contribution: selectively distill task-relevant structure.** Building on the structural-knowledge spirit of prior structural KD, EventKD additionally exploits the *task* structure: it selectively aligns only **event-aware spans** (triggers, arguments, event-type spans) extracted from the teacher's structured generative output, with importance weights derived from the teacher's own attention structure (Eq. 5). The student therefore spends its limited capacity on positions the teacher itself treats as event-relevant, rather than on uniform token-pair alignment.

> **Empirical evidence.** As a controlled ablation we trained a *Token-level Pairwise KD* baseline that shares EventKD's setup exactly (Qwen3-0.6B/4B, ACE05, identical layer mapping, $\beta = 0.9$, $\lambda_{E A} = 2.0$) but replaces event-aware span selection with uniform token-pair alignment over all positions. Best per-metric F1 across 5 epochs:

| Method | Trigger F1 | Argument F1 |
| --- | --- | --- |
| Token-level Pairwise KD (uniform alignment) | 66.40 | 39.67 |
| **EventKD (event-aware spans, ours)** | **67.12** | **43.46** |

> EventKD outperforms the uniform Token-level Pairwise KD baseline on both metrics, by **+0.72 Trigger F1** (67.12 vs 66.40) and by **+3.79 Argument F1** (43.46 vs 39.67), with the gap widening sharply on the role-heavy argument-extraction subtask where structural relations between event components are decisive. This matches the prediction of the capacity-gap argument: when the student must allocate limited capacity, focusing it on event-aware structure pays off precisely on the metric that requires distinguishing roles.

> **References:**
> 
> - Liu, C., Tao, C., Feng, J., & Zhao, D. (2022). Multi-Granularity Structural Knowledge Distillation for Language Model Compression. In ACL.
> - Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2019). TinyBERT: Distilling BERT for natural language understanding. arXiv:1909.10351.
> - Sun, S., Cheng, Y., Gan, Z., & Liu, J. (2019). Patient knowledge distillation for BERT model compression. In EMNLP.
> - Zhang, C., Yang, Y., Liu, J., Wang, J., Xian, Y., Wang, B., & Song, D. (2023). Lifting the Curse of Capacity Gap in Distilling Language Models. In ACL.



#### Official Comment by Authors

**Comment:**

**W2. The method appears sensitive to the teacher’s structured output and the chosen serialization format, since event-aware spans are extracted from generated outputs; this raises robustness concerns when the teacher's output is noisy or malformed.**

> We thank the reviewer and address this concern in three points.

> **(1) Noise propagation is a universal property of distillation, not a vulnerability of EventKD.** Any KD method that aligns student outputs to teacher outputs inherits whatever quality signal the teacher provides; when the teacher is noisy, the student is noisy, regardless of the specific KD loss. This is therefore a property of the distillation paradigm at large rather than a weakness introduced by event-aware span alignment.

> **(2) Our dual-loss design provides a built-in safety net.** The total objective in Eq. 8, $L_{T o t a l} = \left(\right. 1 - \beta \left.\right) \cdot L_{C E} + \beta \cdot \left(\right. L_{K D} + \lambda_{E A} \cdot L_{E A} \left.\right)$, keeps the supervised cross-entropy term $L_{C E}$ active throughout training. Even when the teacher generates a malformed or partially incorrect output for an example, $L_{C E}$ anchors the student to the ground-truth structured annotation, providing a corrective signal that prevents the student from blindly mimicking teacher errors. The $L_{K D}$ and $L_{E A}$ terms transfer the teacher's distributional and structural knowledge *on top of*, not in place of, this supervised anchor.

> **(3) Empirically, malformed output is rare, and the loss degrades gracefully.** On the ACE05 test split, **\>96% of teacher generations parse as valid JSON** conforming to the expected event schema, indicating malformed output is a rare edge case once the teacher is fine-tuned. When an individual response cannot be parsed into spans, the example silently falls back to *token-level KD only* (the $L_{E A}$ term contributes zero), so noise reduces the training signal rather than crashing it.



#### Official Comment by Authors

**Comment:**

**W3. The experiments are limited to a single teacher–student pair within the same model family and to English benchmarks, which raises concerns about the generalizability of the proposed method across architectures and languages.**

> Generalizability across architectures and languages is a fair concern, which we address along both axes with new experiments below.

> **(a) Cross-architecture.** We conduct a cross-family experiment on ACE05 using **Meta-Llama-3-8B-Instruct (teacher)** and **Llama-3.2-1B-Instruct (student)**, with a comparable compression ratio (6.5× vs. 6.7× in our paper's Qwen3 setup). The pipeline is identical to the Qwen3 setup (LoRA-based SFT for the teacher, followed by EventKD with $S F K L + L_{E A}$ for the student). For a fair comparison, we also retrain DistiLLM (SFKL) and CSD baselines under the new family, sharing the same student backbone and identical training budget.

> *Table: Cross-family comparison on ACE05* (strict F1, best per-metric across 5 epochs)
> 
> | Family (teacher → student) | Method | Trigger F1 | Argument F1 |
> | --- | --- | --- | --- |
> | Qwen3 (4B → 0.6B), paper | Teacher (upper bound) | 72.31 | 46.45 |
> |  | Student SFT | 60.57 | 31.34 |
> |  | DistiLLM SFKL | 64.87 | 38.14 |
> |  | CSD | 64.94 | 36.98 |
> |  | **EventKD (ours)** | **67.12** | **43.46** |
> | **Llama-3 (8B → 1B), new** | Teacher (upper bound) | 77.23 | 52.46 |
> |  | Student SFT | 66.30 | 38.00 |
> |  | DistiLLM SFKL | 70.69 | 46.03 |
> |  | CSD | 71.25 | 45.32 |
> |  | **EventKD (ours)** | **72.75** | **46.84** |

> **Results.** On the new Llama-3 cross-family pair, EventKD improves over DistiLLM SFKL by **+2.06 trigger / +0.81 argument F1** and over CSD by **+1.50 trigger / +1.52 argument F1**, while retaining **94.2% of teacher Trigger F1 and 89.3% of teacher Argument F1** under the 6.5× compression. Compared with the SFT-only student baseline, EventKD adds **+6.45 trigger / +8.84 argument F1** on Llama-3 — a magnitude consistent with the +6.55 / +12.12 gain on Qwen3, confirming that the gain over SFT generalizes across architectures rather than being Qwen-specific.

> **(b) Non-English generalization.** The framework is **language-agnostic by construction**: span extraction operates on character offsets within the teacher's structured JSON output and does not depend on language-specific tokenization, morphology, or word segmentation; span-level pooling (Eq. 4) aggregates over arbitrary token spans, so morphologically rich languages where a single trigger spans multiple subwords are supported without method changes. To validate empirically we ran the Qwen3 pipeline on **MINION** (Pouran Ben Veyseh et al., 2022), a multilingual event-detection benchmark covering 8 typologically diverse languages, 5 of which were not supported by prior multilingual event-detection datasets. Below are the **Spanish** test-set results (MINION provides only trigger-level annotations):

> | Method | Trigger F1 |
> | --- | --- |
> | Teacher (Qwen3-4B + LoRA) | 62.72 |
> | Student SFT baseline | 58.30 |
> | DistiLLM SFKL baseline | 58.47 |
> | CSD baseline | 59.15 |
> | **EventKD (ours)** | **60.20** |

> EventKD outperforms the SFT baseline (+1.90 trigger F1), the DistiLLM SFKL baseline (+1.73), and the strongest token-level baseline CSD (+1.05), while retaining 96.0% of teacher trigger F1 and delivering 6.7× parameter compression. The smaller absolute gain over CSD on MINION (+1.05) than on the two ACE settings (Qwen3 +2.18, Llama-3 +1.50) is consistent with MINION being a **trigger-only** dataset: without argument spans, EventKD's event-aware span set reduces to triggers and event-type tokens only, narrowing the structural headroom that span-level alignment can exploit. EventKD nonetheless transfers cleanly to a non-English setting and outperforms every KD baseline tested.

> Together, the cross-family and cross-language experiments confirm that EventKD's effectiveness is not tied to a specific architecture or language. We will include both tables in the revised version.

> **References:**
> 
> - Pouran Ben Veyseh, A., Nguyen, M. V., Trung, F. D., Min, B., & Nguyen, T. H. (2022). MINION: A large-scale and diverse dataset for multilingual event detection. In NAACL.



#### Official Review of Submission1211 by Reviewer zt6F

**Paper Summary:**

This paper studies knowledge distillation for generative event extraction. It argues that existing LLM distillation methods mainly operate at the token level, making it difficult to emphasize event-relevant information or preserve structural relations within events. To address this, the authors propose EventKD, which combines token-level event-aware importance modeling with span-level structural alignment. They further introduce an Event-Aware loss that aligns the pairwise distance structure of key spans, including triggers, arguments, and event types, to better transfer event knowledge.

**Summary Of Strengths:**

1.The paper identifies an important limitation of current distillation methods for generative event extraction: most existing approaches remain focused on output-level distribution matching and do not fully exploit the structural information inherent in the task itself. In this regard, redesigning the distillation objective from the perspective of relations among event components is a meaningful and well-motivated direction. 2.The paper does not treat event extraction as a standard text generation problem, but instead recognizes the inherent dependencies among triggers, arguments, and event types. Accordingly, the proposed method attempts to explicitly preserve such internal event structure during distillation. Compared with token-level alignment alone, this design is better aligned with the nature of the task.

**Summary Of Weaknesses:**

1.The proposed Event-Aware loss aligns pairwise cosine distances among event-aware spans. However, structural relations in event extraction often involve role-specific constraints rather than simple relative distance, and the paper does not yet fully clarify whether this formulation is sufficient to capture such information. 2.The method aligns only a small subset of intermediate layers, such as mapping teacher layers 30/33/36 to student layers 22/25/28 on ACE05. Since the choice of layer mapping may substantially affect performance, the current explanation of why these mappings were selected and whether the method is stable under alternative mappings remains somewhat limited.

**Comments Suggestions And Typos:**

1.Figure 2 illustrates the core Event-Aware distillation mechanism, but the teacher and student branches are not sufficiently distinguishable. Enlarging the key labels and more clearly marking the span representation extraction, pairwise cosine distance computation, and LEA alignment steps would improve the figure’s clarity. 2.Tables 1 and 2 show that EventKD consistently outperforms the baselines, but the paper currently provides only aggregate score comparisons. Adding a few concrete success and failure cases, especially in comparison with standard KD, would make the empirical gains more convincing. 3.It would be helpful to provide a clearer comparison between the student and teacher models in terms of parameter size, inference cost, or deployment efficiency, as this would better highlight the practical value of the proposed method.

**Confidence:** 3 = Pretty sure, but there's a chance I missed something. Although I have a good feel for this area in general, I did not carefully check the paper's details, e.g., the math or experimental design.

**Soundness:** 3.5

**Excitement:** 3 = Interesting: I might mention some points of this paper to others and/or attend its presentation in a conference if there's time.

**Overall Assessment:** 3 = Findings: I think this paper could be accepted to the Findings of the ACL.

**Ethical Concerns:**

There are no concerns with this submission

**Reproducibility:** 4 = They could mostly reproduce the results, but there may be some variation because of sample variance or minor variations in their interpretation of the protocol or method.

**Datasets:** 1 = No usable datasets submitted.

**Software:** 1 = No usable software released.

**Knowledge Of Or Educated Guess At Author Identity:** No

**Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Knowledge Of Paper Source:** N/A, I do not know anything about the paper from outside sources

**Impact Of Knowledge Of Paper:** N/A, I do not know anything about the paper from outside sources

**Reviewer Certification:** I certify that the review I entered accurately reflects my assessment of the work. If you used any type of automated tool to help you craft your review, I hereby certify that its use was restricted to improving grammar and style, and the substance of the review is either my own work or the work of an acknowledged secondary reviewer.



#### Official Comment by Authors

**Comment:**

**W1. The proposed Event-Aware loss aligns pairwise cosine distances among event-aware spans. However, structural relations in event extraction often involve role-specific constraints rather than simple relative distance, and the paper does not yet fully clarify whether this formulation is sufficient to capture such information.**

> We thank the reviewer for raising this important question regarding whether pairwise cosine alignment is sufficient to capture role-specific event structure. We clarify that **EventKD does not explicitly encode role labels in the distillation loss**; instead, it models event structure implicitly through relational alignment in the representation space. This design is motivated by three considerations.

> **(a) Pairwise geometry implicitly reflects role-specific relations.** The teacher representations are learned from an event extraction model, where trigger–argument semantics and argument-role distinctions are expected to be reflected in the geometry of the hidden space. By aligning the **pairwise cosine distance structure** among event-aware spans, EventKD encourages the student to preserve the **relative geometric structure** of trigger and argument representations, including distinctions induced by event roles. This follows the intuition of structural knowledge distillation (Liu et al., 2022), where pairwise and higher-order relational constraints transfer structural knowledge beyond token-level matching.

> **(b) Explicit role-typed modeling introduces additional objective complexity.** A direct formulation of role-specific constraints would require representing structured tuples such as *(trigger, argument, role)* and defining differentiable objectives over these typed relations. While such modeling may offer finer control, it also introduces substantially more structured supervision and optimization complexity. We therefore adopt pairwise cosine alignment as a simpler and more general mechanism for preserving event structure in continuous space.

> **(c) Empirical evidence supports the formulation.** Table 6 shows that adding $L_{E A}$ yields substantial gains in the role-sensitive argument extraction task: **+5.32 Argument F1** over SFKL (38.14 → 43.46) and **+6.39** over RKL. These consistent improvements suggest that pairwise structural alignment captures useful event-specific relational information, particularly for argument prediction.

> In summary, while pairwise cosine alignment may not explicitly encode symbolic role constraints, it provides an effective continuous approximation of event structure and proves sufficient to deliver strong empirical improvements.

> **References:**
> 
> - Liu, C., Tao, C., Feng, J., & Zhao, D. (2022). Multi-Granularity Structural Knowledge Distillation for Language Model Compression. In ACL.



#### Official Comment by Authors

**Comment:**

**W2. The method aligns only a small subset of intermediate layers, such as mapping teacher layers 30/33/36 to student layers 22/25/28 on ACE05. Since the choice of layer mapping may substantially affect performance, the current explanation of why these mappings were selected and whether the method is stable under alternative mappings remains somewhat limited.**

> We appreciate the reviewer’s comment regarding the layer mapping strategy and its stability, and we will clarify these points in the revision.

> **Rationale behind the selected mappings**: We align only a sparse subset of deep transformer layers rather than all intermediate layers. Prior distillation studies (Cho & Hariharan, 2019; Mirzadeh et al., 2020) suggest that dense layer-wise supervision can over-constrain a lower-capacity student, limiting its ability to learn task-specific abstractions. Consistent with this observation, adding a fourth aligned layer (e.g., \[27, 30, 33, 36\]) leads to lower performance than sparse 2–3 layer configurations. Since higher transformer layers encode richer task-specific semantic information, we focus on sparse alignment among deep layers to transfer key structural knowledge while preserving the student’s optimization flexibility. Among the evaluated configurations, \[30, 33, 36\] → \[22, 25, 28\] is selected for ACE05 because it achieves the best Argument F1.

> **Stability under alternative mappings**: As shown in Table 3, EventKD remains effective across multiple layer mappings. Here, stability refers to consistent improvement over the baseline without span-level alignment (\[\] → \[\]), rather than identical performance across all configurations. Several sparse deep-layer mappings (e.g., \[27, 36\], \[28, 32, 36\], and \[30, 33, 36\]) substantially outperform the baseline, indicating that EventKD is not overly sensitive to exact layer indices. In particular, the two 3-layer configurations \[28, 32, 36\] and \[30, 33, 36\] both yield strong gains, with a modest trade-off between Trigger F1 (69.21 vs. 67.12) and Argument F1 (41.64 vs. 43.46). This suggests that the performance gain mainly comes from sparse deep-layer semantic alignment, rather than a specific hand-crafted mapping. We adopt \[30, 33, 36\] → \[22, 25, 28\] for ACE05 because it provides the strongest Argument F1, while the overall improvement remains stable across alternative deep-layer configurations.

**Table 3: Effect of layer mapping on ACE05.** Teacher layers are mapped to corresponding student layers for span-level alignment.

| Teacher | Student | Trigger F1 | Argument F1 |
| --- | --- | --- | --- |
| \[\] | \[\] | 64.87 | 38.14 |
| \[36\] | \[28\] | 65.59 | 39.04 |
| \[33, 36\] | \[25, 28\] | 64.71 | 40.04 |
| \[27, 36\] | \[19, 28\] | 67.99 | 41.55 |
| \[28, 32, 36\] | \[20, 24, 28\] | **69.21** | 41.64 |
| \[30, 33, 36\] | \[22, 25, 28\] | 67.12 | **43.46** |
| \[27, 30, 33, 36\] | \[19, 22, 25, 28\] | 66.85 | 40.25 |



#### Official Comment by Authors

**Comment:**

**W3. Figure 2 illustrates the core Event-Aware distillation mechanism, but the teacher and student branches are not sufficiently distinguishable. Enlarging the key labels and more clearly marking the span representation extraction, pairwise cosine distance computation, and LEA alignment steps would improve the figure’s clarity.**

> We thank the reviewer for the careful reading of Figure 2. The three points raised are largely already addressed by the current figure, and we will polish the remaining details in the camera-ready:
> 
> - **Branch distinguishability.** Teacher and Student are placed in separate rounded-border panels, each containing four span representations ($U_{i}^{T}$ for the Teacher, $U_{i}^{S}$ for the Student). The colored "Event Annotation Components" legend on the left (Trigger in red, Argument in blue, Event Type in purple) applies to the corresponding circles in both panels.
> - **Label legibility.** The central block is named explicitly as "Layer-wise Pairwise Distance Computation (cosine distance)", and the key symbols ($U_{i}^{T}$, $U_{i}^{S}$, token-importance $H^{T} \bigotimes w_{t}$ / $H^{S} \bigotimes w_{t}$, $d_{i j}^{T}$, $d_{i j}^{S}$, $\mathcal{L}_{E A}$) are all present. We will further enlarge them in the camera-ready for half-column readability.
> - **Pipeline visibility.** The figure reads left-to-right as three connected stages: event-aware span extraction, with orange arrows from the central Input panel into both Teacher and Student branches; layer-wise pairwise cosine distance computation in the central yellow block; and cross-model alignment via the vertical $\mathcal{L}_{E A}$ arrow connecting the Teacher distance matrix on top to the Student distance matrix on the bottom.

> The running example shown in Figure 2 ("A man was *shot* in the *park*... *Attack* ", with `shot` / `park` / `Attack` color-coded as Trigger / Argument / Event Type) lets the reader trace the same tokens through both panels and the distance matrices. We will also fix a labelling typo: the upper distance matrix lists $d_{23}^{T}$ twice in place of $d_{24}^{T}$ (and similarly in the Student matrix).



#### Official Comment by Authors

**Comment:**

**W4. Tables 1 and 2 show that EventKD consistently outperforms the baselines, but the paper currently provides only aggregate score comparisons. Adding a few concrete success and failure cases, especially in comparison with standard KD, would make the empirical gains more convincing.**

> Following this concrete suggestion, below we present one representative success case and one shared failure case drawn from the ACE05 test split, comparing EventKD against the two strongest KD baselines, **DistiLLM (SFKL)** and **CSD**, under the same setup. All tables share the `Method | Trigger | Type | Arguments` schema; multiple gold events are listed as multiple rows.

> **Success case (EventKD recovers both finer argument roles and more event triggers than the two strongest baselines).**
> 
> *Input A (idx=108):* "Last month, the SEC slapped fines totaling 1.4 billion dollars on 10 Wall Street brokerages to settle charges of conflicts of interest between analysts and investors."
> 
> | Method | Trigger | Type | Arguments |
> | --- | --- | --- | --- |
> | Gold | fines | Justice:Fine | (SEC, Adjudicator), (brokerages, Entity) |
> | **EventKD (ours)** | fines | Justice:Fine | **(SEC, Adjudicator) ✓, (brokerages, Entity) ✓** |
> | DistiLLM (SFKL) | fines | Justice:Fine | (SEC, *Enforcer* ✗), (brokerages, Entity) ✓ |
> | CSD | fines | Justice:Fine | (SEC, *Entity* ✗), (brokerages, Entity) ✓ |
> 
> *Input B (idx=37):* "Anwar, 56, who this week completed four years in prison on a corruption charge, now faces an earliest possible release date of April 14, 2009 if he is given one third remission of his sentence for good behaviour."
> 
> | Method | Trigger | Type | Arguments |
> | --- | --- | --- | --- |
> | Gold | charge / release / sentence | Justice:Charge-Indict / Release-Parole / Sentence | (Anwar, Person) on `release` |
> | **EventKD (ours)** | release, sentence (2/3) | Release-Parole, Sentence | — |
> | DistiLLM (SFKL) | release (1/3) | Release-Parole | (Anwar, Person) ✓ |
> | CSD | *released* ✗ (0/3) | Release-Parole | (Anwar, Person) ✓ |
> 
> Two complementary phenomena are visible. In *Input A* all three methods recover the trigger (`fines`) and the type (Justice:Fine), but only EventKD assigns the gold `Adjudicator` role to `SEC`; DistiLLM mislabels it as *Enforcer* and CSD as *Entity*, collapsing the regulator and the regulated entity into the same role. In *Input B*, which packs three closely related Justice events into one sentence, EventKD recovers two of three triggers with exact gold spans (`release`, `sentence`), DistiLLM only one, and CSD zero (its surface form `released` fails strict span match). Both phenomena are consistent with the design of $L_{E A}$ (Eq. 7), which distills *pairwise distances* between the trigger span and each argument / co-occurring event span: this preserves the asymmetry between regulator and regulated, and the relative geometry between sibling triggers in dense sentences, neither of which token-level KD enforces. The trade-off is visible in *Input B*: EventKD drops the (Anwar, Person) argument that the baselines retain, i.e. multi-trigger recall comes at a small per-event argument-recall cost in this example.

> **Failure case (all three methods miss the same triggers).** *(idx=158 in test)*
> 
> *Input:* "She is being held on 50,000 dollars bail on a charge of first-degree reckless homicide and hiding a corpse in the death of the infant born in January."
> 
> | Method | Trigger | Type | Arguments |
> | --- | --- | --- | --- |
> | Gold | held | Justice:Arrest-Jail | — |
> | Gold | charge | Justice:Charge-Indict | — |
> | Gold | homicide | Life:Die | — |
> | Gold | death | Life:Die | (infant, Victim) |
> | Gold | born | Life:Be-Born | — |
> | EventKD (ours) | charge | Justice:Charge-Indict | — |
> | EventKD (ours) | death | Life:Die | (infant, Victim) ✓ |
> | DistiLLM (SFKL) | charge | Justice:Charge-Indict | — |
> | DistiLLM (SFKL) | death | Life:Die | (infant, Victim) ✓ |
> | CSD | charge | Justice:Charge-Indict | — |
> | CSD | death | Life:Die | (infant, Victim) ✓ |
> 
> Trigger-recovery score: **all three methods 2/5**; all three miss the same triggers: `held` (Arrest-Jail), `homicide` (a second Life:Die in the same sentence), and `born` (Be-Born). The shared failure mode, namely sentences with **many lexically distinct triggers in close proximity** and the same broad event class repeating (two `Life:Die` triggers here), points to a limitation that span-level alignment alone cannot resolve when the teacher itself does not produce a sufficiently disambiguating signal across nearly co-located spans. Addressing it would likely require explicit *coreference* or *position-aware* objectives, which we mark as future work.



#### Official Comment by Authors

**Comment:**

**W5. It would be helpful to provide a clearer comparison between the student and teacher models in terms of parameter size, inference cost, or deployment efficiency, as this would better highlight the practical value of the proposed method.**

> We thank the reviewer for highlighting the practical value of the proposed method. In the revision, we will include an explicit comparison between the teacher and student models in terms of parameter size, inference cost, and deployment efficiency.

> To avoid confounding effects from variable generation length (which depends on each model’s learned EOS behavior rather than its architecture), we report deployment-oriented metrics: parameter count, throughput (tokens/s and samples/s) under fixed-length generation (128 new tokens per sample), and peak GPU memory. The benchmark is conducted on a single NVIDIA A40 GPU using bf16 precision and greedy decoding over 64 ACE05 test prompts at batch size 16, with 5 warm-up batches. LoRA adapters are merged into the base model to reflect deployment cost, and all numbers are averaged over 3 runs with different seeds.

| Model | Params | Tokens/s (bs=16) | Samples/s (bs=16) | Peak mem (bs=16, GB) | Trigger F1 | Argument F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Teacher (Qwen3-4B + LoRA SFT) | 4.02B | 151.9 | 1.19 | 9.25 | 72.31 | 46.45 |
| Student (Qwen3-0.6B + SFT baseline) | 0.60B | 267.0 | 2.09 | 2.17 | 60.57 | 31.34 |
| **Student (Qwen3-0.6B + EventKD, ours)** | **0.60B** | **264.6** | **2.07** | **2.17** | **67.12** | **43.46** |
| **Student (EventKD) / Teacher** | **0.15×** | **1.74×** ↑ | **1.74×** ↑ | **0.23×** ↓ | **92.8%** | **93.6%** |

> **Results**. EventKD delivers a **6.7× smaller model, 1.74× higher throughput** (in both tokens/s and samples/s), and a 4.3× lower peak GPU memory footprint at batch size 16, while retaining **92.8% of teacher Trigger F1** and **93.6% of teacher Argument F1**. The two student rows show that EventKD introduces no measurable inference overhead relative to the SFT baseline: both use identical peak memory (2.17 GB), and throughput differs by less than 1% (264.6 vs. 267.0 tokens/s), which falls within run-to-run variation.

> Since all KD baselines (KD, RKL, SFKL, CSD, and EventKD) share the same Qwen3-0.6B student backbone and LoRA rank, they incur the same deployment cost. Under this fixed cost budget, EventKD achieves **+6.55 Trigger F1** and **+12.12 Argument F1** over the SFT-only student, substantially improving retention of teacher performance.

