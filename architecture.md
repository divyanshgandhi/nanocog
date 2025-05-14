# Nano‑Cog 0.1 Architecture Specification

## 1 Purpose

Provide a concise, implementation‑ready blueprint for building **Nano‑Cog 0.1**—a laptop‑scale language agent that demonstrates high reasoning‑per‑FLOP efficiency by combining state‑space sequence modelling, retrieval, tool use, and reinforcement‑tuned chain‑of‑thought (CoT).

*Target audience :* founding engineers, RL researchers, infra/dev‑ops, early investors.

---

## 2 System Overview

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                               Nano‑Cog 0.1                                   │
├────────────────────────────────────────────────┬─────────────────────────────┤
│ 1  Inference Core                              │ 2  External Utilities       │
│ ─────────────────────────────────────────────── │ ─────────────────────────── │
│ • Backbone  (Mamba‑130M)                       │ • Chroma‑DB (local)         │
│ • LoRA Reasoning Adapters (+8 M params)        │ • Tool Dispatcher           │
│ • Mini‑MoE FFN Router (+2 M params)            │    – Calculator `<<calc>>`  │
│ • Dynamic Symbol Engine (scratch‑pad tokens)   │    – Python `<<py>>`        │
│ • Retrieval‑Aware Prompt Composer              │    – Bash   `<<bash>>`      │
│                                               │ • Config store (YAML/TOML)  │
└────────────────────────────────────────────────┴─────────────────────────────┘

Training loop wraps the same modules with GRPO RL, synthetic CoT generator, and
a judge model for ranking traces.
```

---

## 3 Component Details

### 3.1 Backbone – Mamba‑130M

* **Layer type** : Selective‑state‑space sequence model (SSM)
* **Size** : 24 × SSM blocks, 130 M parameters
* **Reason** : Linear time/space complexity, Metal‑friendly kernels, 3–5× faster CPU throughput than Transformer‑dense equivalents.
* **Modifications** : (i) 4‑bit weight quantisation for inference, (ii) custom token position encoding unchanged for MVP.

\### 3.2 LoRA Reasoning Head

* **Insertion** : Final two SSM blocks; rank 64 per QKV and FFN.
* **Trainable params** : ≈ 8 M (6 % of backbone).
* **Role** : Houses reasoning‑centric weight updates without touching base weights—keeps fine‑tuning RAM < 11 GB on M1.

\### 3.3 Mini‑MoE Router

* Two feed‑forward “experts” (32‑unit ReLU) sitting after block 12.
* **Gating** : top‑1 token routing, softmax temperature 0.7.
* **Purpose** : Showcase sparse‑activation capacity expansion while maintaining single‑expert forward pass cost.

\### 3.4 Dynamic Symbol Engine (DSE)

* **Grammar tokens** : `<define symbol=X: …>` followed by free‑form definition.
* **Budget** : 50 tokens max per context.
* **Reward** : `compression = max(0, base_len – new_len)`; incorporated into RL.

\### 3.5 Retrieval & Prompt Composer

* **Store** : Chroma (SQLite backend) seeded with 50 k Wikipedia sentences + 5 k code snippets.
* **Embeddings** : `sentence-transformers/all-MiniLM-L6-v2` (384‑D).
* **At inference** : top‑3 docs inserted under `SYSTEM{docs}` tag before user prompt.

\### 3.6 Tool Dispatcher

* **Interface** : special call‑tokens followed by JSON args.
* **Sandbox** : `restrictedpython` for `<<py>>`, pure‑python `sympy` for `<<calc>>`, `subprocess` with allow‑list for `<<bash>>`.
* **Latency budget** : 200 ms per tool call; streams result back into model via turn‑based loop.

\### 3.7 GRPO Reinforcement Loop

| Item       | Value                                     |
| ---------- | ----------------------------------------- |
| Batch size | 16 prompts × 4 candidate traces           |
| Ranking    | judge = frozen Mamba‑130M (headless)      |
| Optimiser  | AdamW β₁ 0.9 β₂ 0.95 lr 1e‑5              |
| Roll‑outs  | 2 iterations over 400 prompts (overnight) |

*Produces adapters that favour shortest correct CoTs and symbol compression.*

\### 3.8 Data Pipeline

* **Sources** : mini‑GSM8K, HumanEval‑mini, ReAct traces.
* **Pre‑processing** : dedupe, 2048‑token packing, number masking for Toolformer phase.

\### 3.9 Evaluation Suite

* GSM8K‑10 % split
* HumanEval‑mini (20 tasks)
* Custom Symbolic‑Math quiz (10 items)
* Bench script outputs accuracy & reasoning‑tokens per answer.

---

## 4 Sequence Flows

1. **Inference (run‑time)**

   1. User → CLI/UI.
   2. Prompt composer retrieves docs → builds context.
   3. Core model generates step‑by‑step thoughts.
   4. When tool token encountered → dispatcher executes → returns result.
   5. CoT proceeds until `FINAL:` token produced.

2. **Training (overnight loop)**

   1. CoT generator samples prompts + 4 candidates.
   2. Judge ranks; GRPO loss applied to adapters + MoE router.
   3. Optional Toolformer masking pass every second epoch.

---

## 5 Deployment Targets

* **CLI** : single‑file `main.py`.
* **Optional Streamlit UI** : port 8123.
* **Resource footprint** : 3.1 GB model + 2 GB retrieval DB + <1 GB Python deps.

---

## 6 Timeline & Milestones

|  Day | Objective                 | Artifact                       |
| ---: | ------------------------- | ------------------------------ |
|   1  | Env setup                 | `requirements.txt`, smoke test |
|   2  | Load Mamba weights        | baseline perplexity log        |
|  3–4 | Supervised LoRA           | `adapter.bin`                  |
|   5  | Retrieval DB built        | `chroma.sqlite`                |
|   6  | Toolformer phase          | tool‑aug weights               |
|  7–8 | GRPO loop                 | RL‑tuned adapters              |
|   9  | DSE reward plug‑in        | updated trainer code           |
|   10 | Mini‑MoE integrated       | new checkpoint                 |
|   11 | Regression tests          | `test_report.md`               |
|   12 | CLI agent v0.1            | `main.py`                      |
|   13 | Streamlit UI              | `app.py`                       |
|   14 | Final benchmarks + README | GitHub push                    |

---

## 7 Risks & Mitigations

| Risk                            | Mitigation                                                                            |
| ------------------------------- | ------------------------------------------------------------------------------------- |
| Metal backend NaNs              | set float32 precision high; unit tests per epoch                                      |
| Tool sandbox escape             | RestrictedPython + allow‑list shell cmds                                              |
| Retrieval drift / hallucination | prepend “SOURCE:” with doc excerpts; train reward model to penalise missing citations |
| Memory overflow on M1           | gradient‑checkpointing + 8‑bit optimiser                                              |

---

## 8 Licensing & IP

* Base checkpoints: Apache‑2.0 (Mamba), compatible with commercial use.
* All fine‑tuned weights + code produced in this project are © 2025 <\<Your Lab Name>>, released under MIT (change as needed).

---

## 9 Next‑Step Roadmap (post‑MVP)

1. **Hybrid Layer Research** – replace 25 % SSM blocks with custom Recursive Symbolic SSM.
2. **Scale to 370 M** – 4× A10G spot instance, 50 B tokens.
3. **Extended Tool Suite** – web search & SQL query tools.
4. **Open‑Weights Release (130 M)** – gather community eval, bug reports.

---

*Prepared: 14 May 2025*
