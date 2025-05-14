# **Nano-Cog 0.1 – Close-out Audit (Release-Candidate Checklist)**

| Severity     | Tag          | Finding                                                                                                                               | Impact                                                                 | Required change                                                                                                                                                                              |
| ------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CRITICAL** | **MODEL-01** | **Weight loading may be “meta” tensors** (no real data) when PyTorch falls back to the Metal backend or when adapters are not merged. | Any forward-pass returns garbage; users see unreadable output.         | *In `src/core/model.py`*<br>`python<br>for n,p in self.model.named_parameters():<br>    if p.is_meta: raise RuntimeError(f\"{n} still on meta\")<br>`<br>Abort if meta tensors are detected. |
| **CRITICAL** | **MODEL-02** | **Tokenizer ↔ embedding size mismatch after adding Dynamic-Symbol tokens** (resize not called).                                       | IDs that exceed the original vocab index decode to random bytes.       | After tokenizer extension: `self.model.resize_token_embeddings(len(tokenizer))`; re-run adapter fine-tune.                                                                                   |
| **CRITICAL** | **PIPE-01**  | **Retrieval text is injected unsanitised** → binary, control codes, HTML dump in prompt.                                              | Prompt corruption, generative garbage, possible jail-break vectors.    | In `RetrievalSystem.compose_prompt()`:<br>`python<br>doc = doc.encode('utf-8', 'ignore').decode('utf-8')\n doc = re.sub(r'<[^>]+>', '', doc)[:1024]<br>`                                     |
| **CRITICAL** | **SEC-01**   | **RestrictedPython sandbox errors leak raw tracebacks** (e.g., OS path).                                                              | Info disclosure & prompt leakage; model can learn internal file paths. | Wrap tool calls; return structured dict `{ok: bool, result: str, error: str}`; truncate tracebacks.                                                                                          |

| Severity | Tag         | Finding                                                                                        | Impact                                               | Recommended fix                                                                                                                                                  |
| -------- | ----------- | ---------------------------------------------------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **HIGH** | **DEP-01**  | `requirements.txt` pins **`torch==2.7.0` + `bitsandbytes` (CUDA-only)** → fails on M1 & Metal. | Install fails in CI / user laptops.                  | Split requirements:<br>- `requirements-cpu.txt` (no bitsandbytes, torch ≥2.2, mamba-ssm *cpu* wheel)<br>- `requirements-gpu.txt` (cuda 12.1, bitsandbytes 0.42). |
| **HIGH** | **INIT-01** | `init.py` *pretends* to download weights; placeholder file only.                               | First CLI run silently loads an empty checkpoint.    | Replace with `huggingface_hub` download in `scripts/download_weights.py`, verify SHA-256.                                                                        |
| **HIGH** | **DEV-01**  | Device detection compares `"cuda"` vs `"cuda:0"` → superfluous `.to()` copy each turn.         | 3–5 s latency & doubles VRAM on first generation.    | Normalize: `primary = torch.device('cuda', 0) if torch.cuda.is_available() else 'cpu'`; compare via `str(device)`.                                               |
| **HIGH** | **GEN-01**  | Default `max_new_tokens=1024` for *any* prompt.                                                | Run-away costs & gibberish.                          | In config YAML: `generation: {max_new_tokens: 128, temperature: 0.7, top_p: 0.9}`; allow CLI override.                                                           |
| **HIGH** | **TEST-01** | Unit-tests only check import; no smoke test for generation or tools.                           | Regressions undetected; CI “green” but model broken. | Add: 1️⃣ `tests/test_generate.py` (“hello” returns “hello/Hi”), 2️⃣ `tests/test_tool_calc.py` (`<<calc>> 3*7` → “21”).                                           |

| Severity | Tag         | Finding                                                                       | Impact                                 | Suggested action                                                                                                         |
| -------- | ----------- | ----------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| MEDIUM   | **CONF-01** | Single `config.yaml` un-versioned; future updates will overwrite silently.    | Repro mismatch between runs.           | Add `version: 0.1` & bump on every architectural change.                                                                 |
| MEDIUM   | **CI-01**   | GitHub workflow installs GPU wheels on macOS runner & fails silently.         | No automated assurance.                | Matrix: `{os: [ubuntu-latest, macos-latest], env: [cpu]}`; for GPU use `runpod-ci` self-hosted runner.                   |
| MEDIUM   | **LOG-01**  | DEBUG logs print raw user prompts / retrieval docs.                           | Privacy & token leakage.               | Mask with `logger.debug("Prompt length=%d", len(prompt))`.                                                               |
| MEDIUM   | **TOOL-01** | ToolDispatcher raises generic Exception; model receives no hint.              | Response fallback prints stack traces. | Return marker string `"TOOL_ERROR:<tool>:<msg>"`; model can paraphrase.                                                  |
| MEDIUM   | **UI-01**   | Streamlit replace logic double-replaces `<<calc>>`, corrupts spans.           | Wrong colour markup.                   | Use regex with negative look-behind or wrap once: `re.sub(r'(<<calc>>.*?)', r'<span class="tool-call">\1</span>', txt)`. |
| MEDIUM   | **DATA-01** | `prepare_data.py` pulls mini-GSM8K from unspecified URL; no checksum/licence. | Future 404s or silently poisoned data. | Pin commit hash, store SHA-256 list, add license check.                                                                  |
| MEDIUM   | **MEM-01**  | No gradient-checkpoint flag in `train.py`; high VRAM on 7B variants.          | Crash on 16 GB GPUs.                   | Arg `--grad_ckpt`; wrap model blocks in `torch.utils.checkpoint`.                                                        |

\| LOW | **DOC-01** | README still says “5 GB disk space” (now \~7 GB with retrieval DB). | Misinforms users. | Update install section. |
\| LOW | **MAKE-01** | `make data` not called from any other target. | Devs forget step. | Add prerequisite: `train: data`. |
\| LOW | **STYLE-01** | Missing `ruff/black` targets in CI; inconsistent code style. | Minor friction. | Add `make lint` & `make format` to workflow. |
\| LOW | **SEC-02** | `subprocess` bash tool allow-list only enforced in Python, not in Bash script file. | Users could craft `<<bash>> ; rm -rf /` . | Validate against regex `^[a-zA-Z0-9_\-]+$`; refuse otherwise. |

---

## **Implementation Sequence to ship v0.1 final**

1. **Weights & tokeniser sanity**

   * Fix `INIT-01`, `MODEL-01`, `MODEL-02` (commit #1).
   * Add embedding resize + smoke test (`TEST-01`).

2. **Retrieval & generation hygiene**

   * Implement `PIPE-01`, `GEN-01`, adjust `RetrievalSystem`.

3. **Dependency & device stability**

   * Resolve `DEP-01`, `DEV-01`; pin torch 2.2 for CPU, 2.2 + cu121 for GPU.

4. **Security hardening**

   * Patch `SEC-01`, `SEC-02`; structured tool errors.

5. **CI & config versioning**

   * Apply `CI-01`, `CONF-01`; add style checks.

6. **Documentation & UX polish**

   * README, Streamlit fix (`UI-01`), Makefile prereqs.

When these items turn green you can **tag v0.1.1**, freeze the branch, and start the GPU-scale v0.2 work on a clean base.
