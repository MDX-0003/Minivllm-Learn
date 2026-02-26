# MinivLLM Multimodal Extension Notes (Experimental)

## ✳️ Working principle (important)

During this extension, **prefer additive changes** via **inheritance** or **"rewrite-by-copy" with renaming** over modifying existing core classes in-place.

- If an existing class needs new fields/behavior (e.g., `Sequence` needs `image_path`), create a new class such as **`ImageSequence`** that **inherits** from `Sequence` (or a new parallel implementation that mimics it) and keep the original text-only path intact.
- Apply the same pattern to other components when practical (e.g., `ModelRunner` → `MMModelRunner`, `Qwen3ForCausalLM` → `QwenVLForCausalLM`).
- Only change existing code when it’s unavoidable for wiring/dispatch (e.g., selecting which runner/model/sequence class to instantiate).

---

This file is a working memo for extending this repo (a minimal vLLM-like inference engine) from **text-only** to **vision-language** (image + text), targeting **Qwen-VL-like** models.

> Goal: learning/experimental. Prioritize minimal changes, clear contracts, and incremental milestones that keep the existing Scheduler/KV-cache/CUDA-Graph decode design intact.

## 1. Current repo context (what exists today)

### Entry points
- `main.py`: demo that builds chat prompts using `AutoTokenizer.apply_chat_template()` and calls `LLMEngine.generate()`.
- `src/myvllm/engine/llm_engine.py`: top-level orchestration.
- `src/myvllm/engine/model_runner.py`: data prep + model execution (prefill eager, decode via CUDA Graph unless `enforce_eager=True`).
- `src/myvllm/models/qwen3.py`: minimal `Qwen3ForCausalLM` implementation used by `ModelRunner`.

### Text-only pipeline summary (high level)
1. `LLMEngine.add_prompt(prompt: str)`
   - uses HF tokenizer to `encode(prompt)`
   - wraps into `Sequence(token_ids=[...])`
2. `Scheduler.schedule()`
   - decides **prefill** vs **decode** batches based on KV-cache capacity and queues
3. `ModelRunner.run(seqs, is_prefill)`
   - `prepare_prefill()` packs variable-length sequences into:
     - `input_ids` (1D concatenated)
     - `cu_seqlens_q/k` (sequence boundaries)
     - `slot_mapping` (where to write KV)
     - optional `block_tables` (where to read KV)
   - `prepare_decode()` packs decode step:
     - `input_ids` (1 token per seq)
     - `context_lens`, `slot_mapping`, `block_tables`
   - `run_model()`
     - prefill: eager forward
     - decode: CUDA Graph replay (assumes per step fixed shapes)
   - rank0 samples next tokens and Scheduler postprocesses

### Key design contracts (important to preserve)
- **Decode** is “one token per sequence per step” and is optimized with **CUDA Graph**.
- **KV cache** is managed as a global pool of blocks (PagedAttention style).
- `Sequence` length (token count) is the main unit the scheduler and block manager reason about.

## 2. Target direction: add image input + switch to Qwen-VL-like model

### Core idea
Add a **vision prefix** during **prefill** only.
- Prefill: (vision prefix) + (text prompt tokens)
- Decode: continue generating text tokens **without re-sending the image**

This keeps the existing decode CUDA Graph assumption intact.

### Recommended minimal modality injection (for this repo)
**Use “soft vision embeddings prefix” (recommended for learning + minimal code changes)**
- image -> vision encoder -> `vision_embeds: (T_vis, hidden_size)`
- text -> token ids -> `text_embeds: (T_text, hidden_size)`
- transformer input: `inputs_embeds = concat(vision_embeds, text_embeds)`

Why:
- avoids discretizing image into token ids
- easier to prototype and iterate
- fits well with a "minimal vLLM" engine

## 3. Work breakdown / milestones (incremental and testable)

### Milestone 1 — “Fake vision prefix” to validate engine integration
**Objective:** end-to-end runs with image+text input, even if vision features are random.
- Extend `Sequence` and request path to carry an image payload
- Modify prefill packing so each seq has an extra `T_vis` prefix length
- Extend model forward to accept `vision_embeds` and concatenate with text embeddings
- Vision encoder can initially return random/constant embeddings

**Acceptance:** pipeline runs; prefill works; decode still works (CUDA Graph unaffected if enabled).

### Milestone 2 — Real vision encoder + projector (still minimal)
**Objective:** outputs vary with image content.
- Implement a simple ViT/CLIP encoder (e.g., via `transformers`) or a lightweight vision stack
- Add `projector: Linear(vision_width -> hidden_size)`
- Standardize image preprocessing (resize/normalize/patchify)

**Acceptance:** same question with different images produces different outputs.

### Milestone 3 — Load actual Qwen-VL weights + correct chat template
**Objective:** reasonable vision-language answers.
- Add new model file (recommended): `src/myvllm/models/qwen_vl.py`
- Extend loader to map vision tower + projector + LLM block parameters
- Ensure tokenizer/chat template matches Qwen-VL conventions

**Acceptance:** simple image QA demo yields plausible answers.

## 4. Design tasks to implement (aligned with existing repo structure)

### 4.1 Data structures
**Extend** `src/myvllm/engine/sequence.py` (do not break existing API):
- `image`: optional image payload (path/PIL/tensor — decide one)
- `vision_embeds`: optional tensor cached for prefill
- `num_vision_tokens`: int
- (optional) `vision_token_range`: helpful for debugging and prompt boundaries

### 4.2 Tokenizer / prompt building
Keep a thin adapter layer rather than hard-coding templates everywhere.
- Add a helper (suggested) `src/myvllm/utils/mm_input.py`:
  - `build_mm_prompt(text: str, has_image: bool) -> str`
  - For now, can be as simple as prefixing something like `<image>\nUSER: ...`.

### 4.3 Model changes
Current: `Qwen3Model.forward(input_ids)`

Add support to accept:
- `forward(input_ids: Tensor, vision_embeds: Optional[Tensor]=None)`
- Or split into `forward_embeddings(inputs_embeds)`

**Important:** vision prefix contributes to KV cache, so the model must process it as real tokens (embeddings) with valid positions.

### 4.4 ModelRunner packing changes (most important)
Modify `prepare_prefill()` to treat each sequence length as:
- `T_total = T_vis + T_text`

Update:
- `input_ids`: still holds only text token ids; vision will be passed separately (e.g., per-seq embeds)
- `cu_seqlens_q/k`: now reflect total lengths per sequence (including vision prefix)
- `slot_mapping`: must cover both vision and text positions so KV is written for all
- `positions`: currently derived in model layer from `cu_seqlens_q` (OK) — but must now match the total token count

Decode path (`prepare_decode`) remains unchanged except:
- `context_lens` must count vision + text history

### 4.5 KV cache / prefix caching considerations
- Vision tokens can be large (256/576/1024+), consuming blocks quickly.
- Prefix caching must incorporate image identity; otherwise different images with same prompt could wrongly reuse cache.
  - For early milestones: easiest is **disable prefix caching for multimodal**.
  - Later: add stable image hash into prefix key.
