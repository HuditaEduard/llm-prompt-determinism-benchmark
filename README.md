# LLM Prompt Determinism & Structural Reliability Benchmark

An empirical evaluation harness measuring **multi-run determinism**, **JSON schema compliance**, **task accuracy**, and **token/latency efficiency** across four prompt engineering tiers on deterministic engineering reasoning tasks.

Built as an open-source research benchmark accompanying an undergraduate manuscript submitted to *Buletinul Științific UPB* (Faculty of Electronics, Telecommunications and Information Technology, Politehnica University of Bucharest).

---

## Research Motivation

In automated engineering pipelines, LLM outputs must be both **syntactically valid** (strict JSON conformance for downstream parsers) and **semantically deterministic** across repeated invocations. This benchmark evaluates how prompt scaffolding interacts with model architecture—comparing a standard non-thinking instruct model (`llama3.1:8b`) against a latent-reasoning ("thinking") model (`qwen3:8b`) at the ~8B parameter scale across **1,440 controlled trials**.

---

## Experimental Design

* **Dataset (`60` Tasks across `3` Engineering Domains):**
  * `digital_logic` (20 tasks): Boolean algebra simplification, truth-table evaluation, sequential state transitions, and binary/hexadecimal arithmetic.
  * `analog_circuits` (20 tasks): Nodal/mesh analysis, RC/RL time constants, operational amplifier gain, and small-signal transistor parameters.
  * `cpp_execution` (20 tasks): Pointer arithmetic, bitwise operations, loop invariants, and deterministic state tracing.
* **Prompt Scaffolding Tiers (`4` Tiers):**
  * `T0_zero_shot`: Direct natural-language problem prompt with no schema constraints.
  * `T1_schema_only`: Strict JSON output schema (`{"final_answer": ...}`) without intermediate reasoning steps.
  * `T2_zero_shot_cot`: Strict JSON schema requiring a `"reasoning"` field prior to `"final_answer"`.
  * `T3_few_shot_cot`: Strict JSON schema with domain-aligned few-shot Chain-of-Thought exemplars.
* **Execution Protocol:**
  * `60 tasks × 4 tiers × 3 repeated runs = 720 trials per model` (`1,440` total trials).
  * Fixed decoding temperature (`temperature = 0.2`) executed locally via Ollama (`CUDA 13.1`, NVIDIA RTX 3060 12GB) to eliminate network jitter and API rate-limit artifacts.

---

## Empirical Results (1,440 Trials)

### Overall Cross-Model Summary

| Model | Prompt Tier | Runs | JSON Compliance (%) | Accuracy (%) | Determinism (%) | Avg Latency (ms) | Avg Completion Tokens |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `llama3.1:8b` | `T0_zero_shot` | 180 | 0.0 | 39.4 | 31.7 | 3,328.3 | 204.9 |
| `llama3.1:8b` | `T1_schema_only` | 180 | 97.2 | 32.8 | 28.3 | 11,579.6 | 95.0 |
| `llama3.1:8b` | `T2_zero_shot_cot` | 180 | 78.9 | 35.6 | 35.0 | 2,877.5 | 178.8 |
| `llama3.1:8b` | `T3_few_shot_cot` | 180 | **100.0** | 29.4 | 46.7 | **1,842.5** | **114.9** |
| `qwen3:8b` | `T0_zero_shot` | 180 | 0.0 | 90.6 | 75.0 | 79,180.1 | 4,067.4 |
| `qwen3:8b` | `T1_schema_only` | 180 | 85.6 | 93.9 | **80.0** | 51,150.1 | 2,146.2 |
| `qwen3:8b` | `T2_zero_shot_cot` | 180 | 74.4 | 94.4 | 78.3 | 66,402.4 | 2,875.1 |
| `qwen3:8b` | `T3_few_shot_cot` | 180 | 86.1 | **96.7** | 76.7 | 40,944.1 | 1,945.9 |

### Domain-Level Accuracy & JSON Compliance (`T3_few_shot_cot`)

| Model | Domain | Accuracy (%) | JSON Compliance (%) |
| :--- | :--- | :---: | :---: |
| `llama3.1:8b` | `digital_logic` | 80.0 | 100.0 |
| `llama3.1:8b` | `analog_circuits` | 6.7 | 100.0 |
| `llama3.1:8b` | `cpp_execution` | 1.7 | 100.0 |
| `qwen3:8b` | `digital_logic` | **100.0** | 100.0 |
| `qwen3:8b` | `analog_circuits` | **93.3** | 58.3 |
| `qwen3:8b` | `cpp_execution` | **96.7** | 100.0 |

### Key Findings

1. **Test-Time Reasoning Dominates Multi-Step Engineering Accuracy:** Enabling latent reasoning (`qwen3:8b`) raises `T3` accuracy from **29.4%** to **96.7%** (+67.3 percentage points), preventing collapse on multi-step nodal analysis (`analog_circuits`) and stateful code tracing (`cpp_execution`).
2. **The Syntax–Reasoning Tradeoff:** Non-thinking `llama3.1:8b` achieves **100.0%** JSON compliance under `T3_few_shot_cot`, whereas `qwen3:8b`'s extended internal `<think>` chains occasionally exhaust context/formatting budgets on complex circuit problems (`58.3%` JSON compliance on `analog_circuits`), yielding **86.1%** overall `T3` compliance.
3. **Few-Shot Anchoring Improves Both Determinism and Efficiency:** On `llama3.1:8b`, moving from `T0` to `T3` increases cross-run determinism from **31.7%** to **46.7%**. On `qwen3:8b`, few-shot exemplars (`T3`) cut average completion length in half (**4,067.4** $\rightarrow$ **1,945.9** tokens) and reduce latency by **48.3%** while maximizing accuracy (**96.7%**).

---

## Repository Structure

```text
llm-prompt-determinism-benchmark/
├── data/
│   ├── tasks/                  # 60 deterministic benchmark tasks (JSON)
│   └── results/                # Raw trial logs & aggregated metric CSVs
│       ├── llama3.1_8b_full.csv
│       ├── qwen3_8b_full.csv
│       ├── summary_metrics.csv
│       └── domain_metrics.csv
├── paper/
│   └── figures/                # 300-DPI publication plots generated by evaluate.py
├── src/
│   ├── prompts.py              # T0–T3 prompt templates and few-shot exemplars
│   ├── runner.py               # OpenAI-compatible benchmark runner with auto-checkpointing
│   └── evaluate.py             # Multi-run determinism & domain metrics + figure generator
├── requirements.txt
└── README.md