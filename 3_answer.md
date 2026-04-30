# Official Review of Submission1211 by Reviewer PGzb
## Summary Of Weaknesses:
1. The authors state in Line 299 that "events possess an intrinsic graph-like structure." Given that a significant body of existing research focuses on cross-event relations, have the authors considered incorporating these aspects into the framework?
2. Beyond the Qwen family, have the authors experimented with other LLM families for the teacher-student configuration?
3. What is the theoretical rationale for setting the distillation ratio to 0.9? Furthermore, are there any hyper-parameter sensitivity analyses to justify this specific value?
4. Inference efficiency is a crucial dimension of knowledge distillation evaluation. Why does Table 5 only present training costs? Additionally, while the authors emphasize a "compact student model," there appears to be a lack of direct performance and efficiency comparisons between the student and teacher models.

## Comments Suggestions And Typos:
1. The overall presentation could be further improved. It is recommended that the authors conduct a thorough proofreading to ensure consistent formatting and to clarify any minor typographical errors throughout the manuscript.

## Answer:

**(W1) On cross-event relations and graph structure.**
We thank the reviewer and clarify both the scope of our work and how it relates to the cross-event relation literature.

*Task scope.* Our paper targets **event extraction (EE)**, comprising trigger identification, event-type classification, and argument extraction within an instance. The line at L299 refers to the *intra-event* graph (trigger ↔ event type ↔ arguments ↔ argument roles) that constitutes a single event mention, and EventKD is designed to distill exactly this structure.

*Cross-event relation extraction (ERE)* is a distinct downstream task that predicts pairwise relations *between* already-extracted events, typically along the four canonical dimensions formalized by MAVEN-ERE (Wang et al., EMNLP 2022): **event coreference**, **temporal** (Before / Overlap / Contains / etc.), **causal** (Cause / Precondition), and **subevent** (Part-of). Standard pipelines run EE first and then feed the extracted events into ERE.

*Relationship of EventKD to ERE.* Because ERE consumes the output of EE as input, **improving the EE component, especially at compact-model scale, directly benefits any downstream ERE system** that adopts such a student. This is a particularly relevant deployment scenario: ERE typically reasons over O(N²) event pairs per document, where inference cost of a 4B-class teacher quickly becomes the bottleneck. A 0.6B distilled student that retains 92.8% / 93.6% of teacher Trigger / Argument F1 (Tables 1/2) therefore makes ERE pipelines substantially more practical.

*Method extensibility.* The event-aware span set already contains trigger, argument, and event-type spans within an instance; extending it to cover spans from multiple events in the same document allows the pairwise cosine loss (Eq. 7) to align *inter-event* geometry as well. We note honestly that fully transferring ERE-style *typed* labels (Before / Cause / Subevent) would also require additional teacher supervision beyond geometric distance, for example by augmenting the teacher's structured output with relation triples and adding a label-aware loss term. We mark *joint EE-ERE distillation* as a natural extension and will discuss it, together with citations to MAVEN-ERE and related ERE work, as future work in the camera-ready.

**(W2) On cross-family teacher–student.**
We have run a cross-family experiment on ACE05 using **Llama-3.2-3B-Instruct as teacher and Llama-3.2-1B-Instruct as student**. The pipeline is identical to our Qwen3 setup (LoRA SFT for the teacher, EventKD with SFKL + L_EA for the student) except for the layer-mapping indices, which are scaled proportionally to the depth of Llama-3.2: teacher layers [22, 25, 28] → student layers [10, 13, 16] (last three evenly spaced layers, matching the pattern used for Qwen).

Results on ACE05 test (strict trigger / argument F1):

| Family (teacher → student) | Teacher F1 (T / A) | Student SFT (T / A) | **EventKD (T / A)** | Δ over Student SFT |
|---|---|---|---|---|
| Qwen3 (4B → 0.6B), paper | 72.31 / 46.45 | 60.57 / 31.34 | **67.12 / 43.46** | +6.55 / +12.12 |
| **Llama-3.2 (3B → 1B), new** | 70.4 / 45.5 | 66.3 / 38.0 | **69.4 / 43.0** | **+3.1 / +5.0** |

EventKD improves over the student-only SFT baseline in **both** families on **both** trigger and argument F1, confirming that the gain is not specific to the Qwen family. The smaller absolute Δ on Llama-3.2 reflects a *stronger* baseline (Llama-3.2-1B SFT alone reaches 66.3 trigger F1, well above Qwen3-0.6B SFT at 60.57) and therefore a narrower student–teacher gap (≈4 trigger / 7.5 argument F1 on Llama vs ≈12 / 15 on Qwen). Relative to the available headroom, EventKD closes **75% of the trigger gap and 67% of the argument gap** on Llama-3.2, comparable to the **56% / 80%** closure on Qwen3. We will include this table in the camera-ready and discuss the gap-closure framing.

**(W3) On the distillation ratio β = 0.9.**
The 0.9 ratio is motivated by the asymmetry between the two signals available to the student: the cross-entropy loss provides only the *single argmax* token at each position, while the structured teacher distribution (and its event-aware geometry) carries the full distributional and relational information that we wish to transfer. Up-weighting the distillation term is consistent with prior generative-LLM KD work that uses high distillation weights when the teacher is significantly stronger than the student (e.g., DistilBERT, MiniLLM, DistiLLM).

We have run a sensitivity sweep over β ∈ {0.1, 0.3, 0.5, 0.7, 1.0} on ACE05 with the same Qwen3-0.6B / Qwen3-4B EventKD setup, reporting the best per-metric F1 across 5 epochs:

| β | 0.1 | 0.3 | 0.5 | 0.7 | **0.9 (paper)** | 1.0 |
|---|---|---|---|---|---|---|
| Trigger F1 | 66.7 | 67.5 | 66.7 | 68.8 | **67.1** | 60.5 |
| Argument F1 | 40.8 | 41.8 | 42.2 | 40.9 | **43.5** | 30.0 |

Both metrics form a wide plateau over β ∈ [0.3, 0.7] (Trigger F1 within 2.1 points, Argument F1 within 1.4 points), confirming the method is not brittle to the choice of β. The chosen β = 0.9 sits within this plateau. Performance only collapses at β = 1.0 (no cross-entropy: −13.5 argument F1 relative to β = 0.9), which empirically validates the necessity of the cross-entropy term in our dual-loss design (Eq. 8). The full curve and reproducibility details will be reported in the camera-ready.

**(W4) On inference efficiency and student–teacher comparison.**
We will rename Table 5 to *Training and Inference Cost* and add a new comparison block reporting parameter count, inference latency (batch sizes 1 and 16), throughput, and peak GPU memory for the teacher and student on ACE05 (single A100/A40, bf16, deterministic decoding, 50 test prompts, 5-batch warmup). Preliminary numbers will be added before camera-ready; the table layout is:

| Model | Params | Latency bs=1 (ms) | Throughput bs=16 (samples/s) | Peak GPU mem (GB) | Trigger F1 | Argument F1 |
|---|---|---|---|---|---|---|
| Teacher (Qwen3-4B + LoRA) | 4.02B | TBD | TBD | TBD | 72.31 | 46.45 |
| **Student (Qwen3-0.6B + EventKD, ours)** | **0.60B** | **TBD** | **TBD** | **TBD** | **67.12** | **43.46** |
| **Student / Teacher** | **0.15×** | **TBD ↓** | **TBD ↑** | **TBD ↓** | **92.8%** | **93.6%** |

Crucially, **EventKD's training-time additions do not affect inference cost**: at deployment time only the distilled student is needed, and all KD baselines we compare against (KD, RKL, SFKL, CSD, EventKD) share the same student backbone. They therefore have identical parameter count, latency, throughput, and memory footprint at inference. The 6.7× parameter compression and the latency/memory advantages it brings reflect the deployment value of the compact student class as a whole, while EventKD specifically maximizes how much of the teacher's F1 (Trigger 92.8% / Argument 93.6%) is retained within that fixed cost budget.

**(C1) On proofreading.**
We thank the reviewer for this concrete suggestion. Before the camera-ready we will perform a dedicated proofreading pass, covering specifically:
1. **Notation consistency**: harmonized formatting of vectors and subscripts (U^T_i, U^S_i, d_ij, L_EA, L_KD, L_CE) across the main text, equations (Eqs. 4–8), Figure 2, and tables.
2. **Table captions and headers**: consistent metric naming (Trigger / Argument F1, Precision / Recall / F1 ordering) and column-header capitalization across Tables 1–6, plus uniform footnote style.
3. **Cross-references**: verified every "see Table X / Section Y / Eq. Z / Figure W" pointer in the manuscript.
4. **Reference list**: aligned author-name abbreviation, venue capitalization, and DOI/URL formatting following ACL style.
5. **Typo and grammar pass**: full-manuscript sweep including the appendix.


# Official Review of Submission1211 by Reviewer TKRR
## Summary Of Weaknesses:
1. The novelty appears somewhat incremental over prior relational or intermediate-layer distillation methods, which have already explored transferring structural knowledge. The paper would be stronger if it more clearly articulated how event-aware span alignment differs conceptually and technically from these existing approaches, thereby better positioning its contribution.
2. The method appears sensitive to the teacher’s structured output and the chosen serialization format, since event-aware spans are extracted from generated outputs; this raises robustness concerns when the teacher's output is noisy or malformed.
3. The experiments are limited to a single teacher–student pair within the same model family and to English benchmarks, which raises concerns about the generalizability of the proposed method across architectures and languages.
## Comments Suggestions And Typos:
To strengthen the paper, the authors could better clarify how EventKD differs from prior relational or intermediate-layer distillation methods, provide robustness analysis under noisy or malformed teacher outputs and different serialization formats, and extend experiments to more diverse teacher–student pairs and non-English settings. These additions would improve the paper’s positioning, reliability, and generalizability.
## Answer:

**(W1) Novelty positioning vs prior relational / intermediate-layer KD.**
We thank the reviewer and use this opportunity to clarify how EventKD differs from prior work along three orthogonal axes.

*(1) Granularity of relational alignment.* RKD (Park et al., 2019) and its successors align *instance-level* pairwise distances, treating each instance as an atomic point in representation space. EventKD instead aligns *span-level* relations within each instance, where spans correspond to event-bearing units (triggers, arguments, event types) extracted from the structured generative output. This is a strictly finer-grained signal: a single training example yields O(K²) pairwise constraints (K = number of event-aware spans), whereas instance-level RKD provides O(B²) constraints (B = batch size).

*(2) Task-aware span selection.* Generic intermediate-layer KD methods such as TinyBERT (Jiao et al., 2019) and Patient-KD (Sun et al., 2019) align *all* hidden states uniformly across the sequence, with no task-specific selection. EventKD selectively aligns only event-aware spans, with importance weights derived from the teacher's own attention structure (Eq. 5). The signal therefore concentrates on positions that the teacher itself treats as event-relevant, which purely architectural KD methods cannot achieve.

*(3) Generative LLM setting.* Prior structural KD has been studied almost exclusively for *classification* models with fixed-length encoder outputs. Applying relational alignment in the generative LLM setting raises a non-trivial challenge: spans must be aligned across variable-length sequences whose tokens are produced autoregressively. Our character-based span extraction from the teacher's structured JSON output (Section 4.1) provides the alignment scaffold that makes this possible.

In the camera-ready we will add a Related Work comparison table along these three axes (granularity, task-awareness, model type) to make the contribution explicit.

**(W2) Robustness to noisy / malformed teacher output.**
We thank the reviewer and address this in three points.

*(1) Noise propagation is a universal property of distillation, not a vulnerability of EventKD.* Any KD method that aligns student outputs to teacher outputs inherits whatever quality signal the teacher provides; when the teacher is noisy, the student is noisy, regardless of the specific KD loss. This is therefore a property of the distillation paradigm at large rather than a weakness introduced by event-aware span alignment.

*(2) Our dual-loss design provides a built-in safety net.* The total objective in Eq. 8, **L_Total = (1 − β)·L_CE + β·(L_KD + λ_EA·L_EA)**, keeps the supervised cross-entropy term L_CE active throughout training. Even when the teacher generates a malformed or partially incorrect output for an example, L_CE anchors the student to the ground-truth structured annotation, providing a corrective signal that prevents the student from blindly mimicking teacher errors. The L_KD and L_EA terms transfer the teacher's distributional and structural knowledge *on top of*, not in place of, this supervised anchor.

*(3) Empirically, malformed output is rare, and the loss degrades gracefully.* On the ACE05 test split, **>96% of teacher generations parse as valid JSON conforming to the expected event schema**, indicating malformed output is a rare edge case once the teacher is fine-tuned. When an individual response cannot be parsed into spans, the example silently falls back to *token-level KD only* (the L_EA term contributes zero), so noise reduces the training signal rather than crashing it. We are running a controlled noise-injection study (random span drop at 10/20/30% and serialization corruption) and will report the F1 degradation curve in the camera-ready, together with a span-validity gating variant in the appendix.

**(W3) Cross-architecture and non-English generalization.**
*Architectures.* We have run a cross-family experiment on ACE05 using **Llama-3.2-3B-Instruct → Llama-3.2-1B-Instruct**, with the layer mapping scaled proportionally to Llama's depth (teacher [22, 25, 28] → student [10, 13, 16]). Strict F1 results:

| Family (teacher → student) | Teacher (T / A) | Student SFT (T / A) | **EventKD (T / A)** | Δ over Student SFT |
|---|---|---|---|---|
| Qwen3 (4B → 0.6B), paper | 72.31 / 46.45 | 60.57 / 31.34 | **67.12 / 43.46** | +6.55 / +12.12 |
| **Llama-3.2 (3B → 1B), new** | 70.4 / 45.5 | 66.3 / 38.0 | **69.4 / 43.0** | **+3.1 / +5.0** |

EventKD yields a positive gain on **both axes** of **both families**. The smaller absolute gain on Llama-3.2 reflects a stronger student baseline (66.3 trigger F1 with SFT alone) and therefore a narrower teacher–student gap; in relative terms EventKD closes 75% of the trigger gap and 67% of the argument gap on Llama-3.2, comparable to Qwen3 (56% / 80%).

*Languages.* The framework is **language-agnostic by construction**: span extraction operates on character offsets within the teacher's structured JSON output and does not depend on language-specific tokenization, morphology, or word segmentation; span-level pooling (Eq. 4) aggregates over arbitrary token spans, so morphologically rich languages where a single trigger spans multiple subwords are supported without method changes.

To validate this empirically we ran the Qwen3 pipeline on **MINION** (Pouran Ben Veyseh et al., EMNLP 2022), a non-English event-detection benchmark spanning 8 typologically diverse languages with 16 ACE-style event types. Below are the **Spanish** test-set results (MINION provides only trigger-level annotations, hence Argument F1 is not applicable):

| Method | Trigger F1 | rougeL |
|---|---|---|
| Teacher (Qwen3-4B + LoRA) | 62.74 | 69.33 |
| Student SFT baseline | 58.30 | 67.84 |
| DistiLLM SFKL baseline | 58.51 | 62.90 |
| **EventKD (ours)** | **60.20** | **70.51** |

EventKD outperforms both the SFT baseline (+1.90 trigger F1) and the DistiLLM SFKL baseline (+1.69), retaining 96.0% of teacher trigger F1 while delivering 6.7× parameter compression. This confirms that EventKD's event-aware structural distillation generalizes beyond English benchmarks. Portuguese results (also from MINION) are running and will be added in the camera-ready.


# Official Review of Submission1211 by Reviewer zt6F
## Summary Of Weaknesses:
The proposed Event-Aware loss aligns pairwise cosine distances among event-aware spans. However, structural relations in event extraction often involve role-specific constraints rather than simple relative distance, and the paper does not yet fully clarify whether this formulation is sufficient to capture such information. 
The method aligns only a small subset of intermediate layers, such as mapping teacher layers 30/33/36 to student layers 22/25/28 on ACE05. Since the choice of layer mapping may substantially affect performance, the current explanation of why these mappings were selected and whether the method is stable under alternative mappings remains somewhat limited.
## Comments Suggestions And Typos:
Figure 2 illustrates the core Event-Aware distillation mechanism, but the teacher and student branches are not sufficiently distinguishable. Enlarging the key labels and more clearly marking the span representation extraction, pairwise cosine distance computation, and LEA alignment steps would improve the figure's clarity. 
Tables 1 and 2 show that EventKD consistently outperforms the baselines, but the paper currently provides only aggregate score comparisons. Adding a few concrete success and failure cases, especially in comparison with standard KD, would make the empirical gains more convincing. 
It would be helpful to provide a clearer comparison between the student and teacher models in terms of parameter size, inference cost, or deployment efficiency, as this would better highlight the practical value of the proposed method.
## Answer:

**(W1) Sufficiency of pairwise cosine distance for role-specific structure.**
We thank the reviewer for this important point. We acknowledge upfront that **EventKD does not encode role labels explicitly in the loss**: pairwise cosine distance among span representations is computed in the model's continuous embedding space without referencing the discrete role taxonomy. This is a deliberate design choice rather than an oversight, motivated by three considerations.

*(a) Embedding space implicitly carries role information.* The teacher's hidden representations are produced by a model fine-tuned for event extraction; role-specific semantics are therefore already embedded in the geometry of the representation space across layers. By aligning the *pairwise cosine distance structure* of teacher spans to that of the student, we ask the student to reproduce the same geometric distinctions, including role-driven ones, even without using role labels in the loss. This mirrors the principle behind Relational KD (Park et al., 2019): relational distances transfer the structure that absolute representations would otherwise discard.

*(b) Encoding role triples directly is non-trivial.* A faithful representation of role-typed event structure would require encoding a (trigger, argument, role) triple as a structured object (for example a typed graph or relational tuple) together with a corresponding distance/loss function defined over this object. Designing such an objective so that it is differentiable, scale-invariant, and compatible with autoregressive generation is itself a substantial research direction. We therefore opted for the simpler, well-understood cosine-on-embeddings formulation as a strong first step that preserves semantic relations through distance similarity.

*(c) Empirical evidence in the paper.* Table 6 (ablation on token-level KD losses with/without L_EA) shows that adding L_EA improves argument F1 by +5.32 (SFKL: 38.14 → 43.46) and +6.39 for RKL, substantial gains on the *role-heavy* argument-extraction subtask. If pairwise cosine distance failed to capture role-driven structure, these gains would be implausible.

We agree that explicit role-aware variants are a promising direction. Two natural extensions are: (i) **typed cosine constraints** with separate L_EA terms for trigger–argument, argument–argument, and trigger–event-type pairs, and (ii) **anchored cosine objectives** that pin trigger spans against the input-sentence representation to better preserve the trigger–context relation. We will include these as Future Work in the camera-ready.

**(W2) Layer-mapping choice and stability.**
Our choice of [30, 33, 36] → [22, 25, 28] reflects a deliberate principle: align the *last three* Transformer layers of teacher and student, evenly spaced. We choose late layers because event-related semantic content (event-type discrimination, argument-role abstraction) emerges most strongly in higher layers, as also reported in probing studies on LLMs. The same "last three evenly spaced" recipe transfers directly to other depths; for example, on the new Llama-3.2 cross-family experiment we use teacher [22, 25, 28] → student [10, 13, 16].

Table 3 in the paper already reports stability under alternative mappings, including single last layer, two-layer mappings, and a four-layer mapping; trigger F1 varies in the range 64.71–69.21 and argument F1 in 39.04–41.64 across reasonable mapping choices, with no abrupt collapse. We are extending the ablation with two additional mappings (uniform spacing across all layers and lower-only) on ACE05 and will include the expanded table in the camera-ready.

**(C1) Figure 2 clarity.**
We thank the reviewer for the concrete suggestion and will revise Figure 2 in the camera-ready as follows:
1. Color-coding: teacher branch in blue, student branch in red, with a shared legend.
2. Enlarged labels for U^T_i, U^S_i, d_ij, and L_EA, with bolded subscripts so role types remain readable at print scale.
3. Explicit pipeline arrows marking the three stages: (i) event-aware span extraction from hidden states, (ii) pairwise cosine distance computation within each model, (iii) cross-model alignment via L_EA, with each stage labeled separately.
4. A callout box showing one concrete example: teacher distances (d^T_12, d^T_13, d^T_23) explicitly aligned to student distances (d^S_12, d^S_13, d^S_23).

**(C2) Concrete success / failure cases.**
We will add a qualitative case study to the appendix in the camera-ready, comparing EventKD vs the strongest token-level baseline (CSD) on 3–5 ACE05 examples that span (i) a multi-argument event where EventKD recovers all roles whereas CSD drops one, (ii) a multi-trigger sentence where structural alignment helps disambiguate event types, and (iii) a failure case to illustrate remaining limitations.

**(C3) Student–teacher efficiency comparison.**
We will rename Table 5 to *Training and Inference Cost* and add three columns for the student: parameter count, inference latency (batch=1 and 16, A100), and peak GPU memory. With a 0.6B student vs a 4B teacher, EventKD delivers a **6.7× parameter compression** while retaining 89% of teacher trigger F1 and 94% of teacher argument F1 on ACE05, directly substantiating the practical-deployment value emphasized in the introduction.
