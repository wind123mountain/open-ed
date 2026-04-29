# Rebuttal Draft — 100% Answerable Points (No Experiment Needed)

**Status**: First draft — ready for refinement. All 5 points below can be finalized without waiting for any experiment.

---

## 1. TKRR-W1: Novelty positioning vs prior relational / intermediate-layer KD

**Reviewer's concern**: "The novelty appears somewhat incremental over prior relational or intermediate-layer distillation methods."

### Draft Response

> We thank the reviewer for raising this important point and use this opportunity to clarify how EventKD differs from prior work along three orthogonal axes.
>
> **(1) Granularity of relational alignment.** RKD (Park et al., 2019) and its successors align *instance-level* pairwise distances — each instance is treated as an atomic point in representation space. EventKD instead aligns *span-level* relations within each instance, where spans correspond to event-bearing units (triggers, arguments, event types) extracted from the structured generative output. This is a strictly finer-grained signal: a single training example yields O(K²) pairwise constraints (K = number of event-aware spans), whereas instance-level RKD provides O(B²) constraints (B = batch size).
>
> **(2) Task-aware span selection.** Generic intermediate-layer KD methods such as TinyBERT (Jiao et al., 2019) and Patient-KD (Sun et al., 2019) align *all* hidden states uniformly across the sequence, with no task-specific selection. EventKD selectively aligns only event-aware spans, with importance weights derived from the teacher's own attention structure (Eq. 5). The signal therefore concentrates on positions that the teacher itself treats as event-relevant — something purely architectural KD methods cannot achieve.
>
> **(3) Generative LLM setting.** Prior structural KD has been studied almost exclusively for *classification* models with fixed-length encoder outputs. Applying relational alignment in the generative LLM setting raises a non-trivial challenge: spans must be aligned across variable-length sequences whose tokens are produced autoregressively. Our character-based span extraction from the teacher's structured JSON output (Section 4.1) provides the alignment scaffold that makes this possible.
>
> In the camera-ready, we will add a Related Work comparison table along these three axes (granularity, task-awareness, model type) to make the contribution explicit.

---

## 2. zt6F-W1: Is pairwise cosine distance sufficient for role-specific structure?

**Reviewer's concern**: "Structural relations in event extraction often involve role-specific constraints rather than simple relative distance."

### Draft Response

> We agree that role information is an important aspect of event structure, and we appreciate the suggestion. We argue that pairwise cosine distance is nonetheless an effective and well-motivated formulation, for three reasons.
>
> **(a) Distance preservation implicitly encodes role information.** The teacher's hidden representations already encode role-specific semantics — that is precisely what fine-tuning on event extraction produces. By matching the *pairwise distance structure* of teacher spans, the student is forced to reproduce these role distinctions geometrically, even without explicit role labels in the loss. This is the same principle that gives Relational KD (Park et al., 2019) its strong empirical performance in vision: relational distances *transfer* the structure that absolute representations would otherwise discard.
>
> **(b) Robustness to capacity gap.** Matching individual span representations directly (e.g., per-role MSE) would force the 0.6B student to reproduce 4B-dimensional teacher features absolutely — a known-difficult problem under large capacity gaps. Pairwise distance alignment is invariant to representation rotation/scaling and only requires preserving *relative* structure, which is empirically easier for small students to satisfy.
>
> **(c) Empirical evidence in the paper.** Table 6 (ablation on token-level KD losses with/without LEA) shows that adding LEA improves argument F1 by +5.32 (SFKL: 38.14 → 43.46) and +6.39 for RKL — a substantial gain on the *role-heavy* subtask. If pairwise distance failed to capture role structure, this gain would be implausible.
>
> A natural extension is to enforce role-typed distance constraints (e.g., separate loss terms for trigger–argument vs argument–argument pairs). We agree this is promising and will add it to Future Work.

---

## 3. PGzb-Q1: Cross-event relations and graph structure

**Reviewer's concern**: Paper claims events possess "graph-like structure" but does not incorporate cross-event relations.

### Draft Response

> We thank the reviewer for this thoughtful suggestion. To clarify our scope: Section 4.1 frames the *intra-event* graph (trigger ↔ event type ↔ arguments ↔ argument roles) as the structural unit our method captures, which is itself non-trivial and largely ignored by token-level KD. The line the reviewer cites (L299) refers to this intra-event graph rather than to cross-event relations.
>
> That said, we agree cross-event relations (coreferent triggers, temporal/causal relations, document-level event structure) are a valuable extension. Importantly, **the EventKD framework readily generalizes to this setting**: the only required change is to expand the event-aware span set to include cross-document or cross-sentence event mentions; the pairwise distance loss (Eq. 7) operates unchanged on the enlarged span set. We will make this extensibility explicit in the camera-ready and add cross-event-relation transfer as future work in Section 7.

---

## 4. zt6F-C1: Figure 2 clarity

**Reviewer's suggestion**: "Teacher and student branches are not sufficiently distinguishable. Enlarging the key labels and more clearly marking the span representation extraction, pairwise cosine distance computation, and LEA alignment steps would improve the figure's clarity."

### Draft Response

> We thank the reviewer for the concrete suggestion and will revise Figure 2 in the camera-ready as follows:
> 1. **Color-coding**: teacher branch in blue, student branch in red, with a shared legend.
> 2. **Enlarged labels** for U^T_i, U^S_i, d_ij, and L_EA, with bolded subscripts so role types remain readable at print scale.
> 3. **Explicit pipeline arrows** marking the three stages — (i) event-aware span extraction from hidden states, (ii) pairwise cosine distance computation within each model, (iii) cross-model alignment via L_EA — each stage labeled separately.
> 4. A **callout box** showing one concrete example: teacher distances (d^T_12, d^T_13, d^T_23) explicitly aligned to student distances (d^S_12, d^S_13, d^S_23).

---

## 5. TKRR-W3 (language part): Non-English benchmarks

**Reviewer's concern**: "Experiments are limited to ... English benchmarks, which raises concerns about the generalizability of the proposed method across architectures and languages."

(Architecture concern is addressed by the cross-family experiment T1; we focus here on the language axis.)

### Draft Response

> We acknowledge this limitation, which we already noted in Section 7 (L612–617). Two clarifications support the cross-language generalizability of EventKD:
>
> **(1) Method is language-agnostic by construction.** The event-aware span extraction step (Section 4.1) operates on character offsets within the teacher's structured JSON output. It does not depend on language-specific tokenization, morphology, or word-segmentation rules. Pairwise distance alignment likewise operates in the model's continuous representation space, which is itself learned from multilingual pre-training in modern LLMs.
>
> **(2) Span granularity adapts naturally.** For morphologically rich languages where a single trigger may span multiple subword tokens, our span-level pooling (Eq. 4) already aggregates over arbitrary token spans — no method change is required.
>
> Empirical validation on multilingual benchmarks (e.g., ACE05 Chinese, multilingual MAVEN) is a natural next step. Given rebuttal time constraints, we mark this as future work and will commit to it in the camera-ready.

---

# Quick stats for the response

| Point | Word count | Confidence |
|---|---|---|
| TKRR-W1 (novelty) | ~280 | High — clear differentiation |
| zt6F-W1 (cosine sufficiency) | ~280 | High — supported by Table 6 |
| PGzb-Q1 (cross-event) | ~150 | High — clarifies scope |
| zt6F-C1 (Figure 2) | ~110 | High — pure commitment |
| TKRR-W3 (non-English) | ~150 | Medium — relies on argument |
| **Total** | **~970 words** | |

ACL rebuttal limit is typically 2500 words across all reviewers — this leaves ~1500 words for the experiment-dependent points (cross-family results, ratio sweep, inference table, layer mapping clarification, robustness, case study).

---

# Notes for refinement

- These drafts use **first-person plural** ("we thank", "we argue") — standard for rebuttals.
- Each response **acknowledges first**, then **argues**, then **commits to revision** — three-part structure.
- All claims that reference paper sections/tables can be verified — check line numbers before final submission.
- For TKRR-W1, optionally add 1-2 specific paper citations if word budget permits.
- The drafts are written in English; final rebuttal must be English.
