# Qwen2.5-VL Real-Checkpoint Inference Roadmap

## Document Goal

This document defines a practical learning path for the next stage of the project:

- stop treating multimodality as a simulated extension of a text-only model
- load a real **Qwen2.5-VL** checkpoint
- support **image input + VL inference** inside the current learning-oriented serving framework

This roadmap is intentionally focused on **inference integration**, not training or pretraining.

The goal is not to reproduce all of vLLM at once.
The goal is to build a clear, testable, staged path from:

- `text-only Qwen3 + custom multimodal glue`

to:

- `real Qwen2.5-VL checkpoint + image-aware inference path`

---

## Scope And Non-Goals

### In Scope

- understand the official Qwen2.5-VL input contract
- reproduce the official prompt / image input format
- make the current framework able to call a real Qwen2.5-VL checkpoint
- preserve the current engine's learning value around scheduling, prefill, decode, and KV-cache concepts

### Out Of Scope

- VL training
- VL pretraining
- full reimplementation of official vLLM internals
- immediate support for video, audio, or multi-turn agent workflows
- full parity with all official Qwen2.5-VL features in the first pass

---

## The Core Direction

The project should no longer evolve by adding more custom vision logic on top of a text-only `Qwen3ForCausalLM`.

Instead, the next learning milestone should be:

1. use the **official Qwen2.5-VL processor contract** as the source of truth for inputs
2. use a **real Qwen2.5-VL checkpoint** as the source of truth for model semantics
3. adapt the current framework so it can **serve** that model, rather than pretending a text-only model is already a VL model

This means the project direction changes from:

- "manually build a fake VL model around a text backbone"

to:

- "let the framework learn how to host a real VL model"

---

## Why This Direction Is Better

The current code already proved something valuable:

- multimodal length accounting can coexist with the scheduler
- multimodal information can enter in prefill
- decode can remain text-only
- model-side embedding merge is a useful abstraction boundary

But the current stack still has a hard ceiling:

- the loaded model is still a text-only checkpoint
- the current vision encoder / projector are not aligned to Qwen2.5-VL weights
- the prompt contract is still not the official VL chat contract

As long as those three facts remain true, the framework can only prove engineering viability, not real VL inference capability.

So the next steps must shift from "improving the fake multimodal path" to "aligning with the real model contract".

---

## Official References To Follow

These references should be treated as the primary truth source.

### 1. Qwen2.5-VL Transformers Documentation

Use this to understand:

- model input fields
- processor behavior
- special token semantics
- image-related tensors such as `pixel_values` and `image_grid_thw`

Reference:

- https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl

### 2. Qwen Official README / Usage Examples

Use this to understand:

- how `messages` are structured
- how `processor.apply_chat_template(...)` is used
- what the official chat-style multimodal prompt flow looks like

Reference:

- https://github.com/QwenLM/Qwen2.5-VL

### 3. vLLM Multimodal Inputs Documentation

Use this to understand:

- how a serving engine expects multimodal requests to be packaged
- the split between `prompt` and `multi_modal_data`
- what should remain model-specific versus engine-generic

Reference:

- https://docs.vllm.ai/en/latest/features/multimodal_inputs.html

### 4. vLLM Qwen2.5-VL Model Implementation

Use this only after understanding the first three references.

Use it to study:

- how vLLM adapts Qwen2.5-VL internally
- model-side multimodal embedding preparation
- how the serving engine and the model adapter are separated

Reference:

- https://docs.vllm.ai/en/latest/api/vllm/model_executor/models/qwen2_5_vl.html

---

## Learning Milestones

The roadmap should be executed in the following order.
Do not skip stages.

### Milestone 0: Establish The Official Baseline Outside This Repo

#### Goal

Before adapting the current framework, confirm that the chosen Qwen2.5-VL checkpoint works in the official stack.

#### What To Do

1. Run one image+question example with the official `transformers` + Qwen2.5-VL processor.
2. Print or inspect:
   - the raw `messages`
   - the result of `processor.apply_chat_template(...)`
   - the shape and dtype of image-related model inputs
   - the final answer text
3. Save one "known-good" example as the project baseline.

#### Why This Matters

If the official baseline is not established first, later failures become ambiguous:

- is the bug in the current framework?
- is the checkpoint setup wrong?
- is the prompt format wrong?
- is the image processing wrong?

#### Deliverable

A small reproducible baseline note in `Docs/` or a notebook/script outside the serving path.

#### Acceptance Check

You can answer all of the following:

- What exact object structure does official Qwen2.5-VL expect before inference?
- What exact prompt string does the processor produce?
- What exact image tensors are sent into the model?

---

### Milestone 1: Freeze The Framework Boundary

#### Goal

Clarify which parts of the current framework are worth preserving and which parts must be replaced when moving to a real VL checkpoint.

#### Preserve

- scheduler
- block manager
- the high-level prefill/decode split
- the idea that multimodal information enters before decode

#### Replace Or Re-scope

- custom text-only `MMQwen3ForCausalLM` as the final target model
- custom `VisionEncoder` / `VisionProjector` as the authoritative VL path
- custom placeholder-only prompt protocol as the final truth source

#### Key Insight

At this stage, the framework should stop thinking:

- "the engine owns vision semantics"

and instead think:

- "the engine hosts a real VL model and its official input contract"

#### Acceptance Check

You can point to each current module and say whether it is:

- kept as-is
- adapted
- replaced
- only retained as a learning scaffold

---

### Milestone 2: Align The Input Contract To Official Qwen2.5-VL

#### Goal

Make the project's input pipeline look like the official Qwen2.5-VL path, even before the serving engine fully hosts the model.

#### What Must Change

The project should move toward this conceptual flow:

1. user-facing request
2. `messages`
3. `processor.apply_chat_template(...)`
4. image processing
5. model inputs

instead of the current simplified flow:

1. text chat template
2. prepend placeholder tokens manually
3. custom multimodal payload

#### Required Learning Outcomes

You must understand:

- how official `messages` are structured
- where image placeholders come from
- whether the placeholders are user-authored or processor-generated
- what tensors accompany the prompt

#### Implementation Direction

`Processor` should gradually evolve into the local owner of:

- official prompt construction
- image preprocessing
- multimodal input packaging

It should no longer just be a helper that prepends custom placeholder tokens.

#### Acceptance Check

For one sample image question, you can inspect the local processor output and show that it maps clearly onto the official Qwen2.5-VL processor output.

---

### Milestone 3: Introduce A Real Qwen2.5-VL Model Adapter

#### Goal

Serve a real Qwen2.5-VL checkpoint in the current framework.

#### The Important Design Choice

Do **not** immediately try to reimplement the entire Qwen2.5-VL architecture by hand.

The first practical learning target should be:

- create a **model adapter** layer that lets the current framework call a real Qwen2.5-VL model

This is the highest-leverage route because it lets you study:

- how engine inputs must be shaped
- where prefill/decode assumptions still hold
- what parts of the framework are actually model-agnostic

without first solving:

- full VL weight remapping
- exact internal model structure parity
- custom vision tower implementation parity

#### Adapter Responsibilities

The adapter should define:

- what the framework passes in
- what the real Qwen2.5-VL model expects
- how logits are returned back to the existing sampling path

#### Acceptance Check

The current framework can successfully call a real Qwen2.5-VL forward pass for at least one image+text example.

---

### Milestone 4: Reconcile The Engine With Real VL Inference Constraints

#### Goal

Check which assumptions from the text-only engine still hold when the model is a real Qwen2.5-VL model.

#### Areas To Validate

1. **Prefill-only image processing**
   - confirm whether image-conditioned information only needs to be handled before decode in your framework

2. **Sequence length accounting**
   - verify that image-related token positions and text token positions match the model's actual expectations

3. **Positions and cache behavior**
   - confirm that the framework's position bookkeeping still matches the model's prefill/decode contract

4. **Sampling path**
   - confirm logits from the real model can still flow through the current sampler unchanged

#### Likely Outcomes

Some current abstractions may survive unchanged.
Some may need local adaptation once a real VL model is in the loop.

That is expected.

#### Acceptance Check

You can explain which existing assumptions were:

- already correct
- locally patched
- fundamentally incompatible with real Qwen2.5-VL inference

---

### Milestone 5: Replace Learning-Only Vision Components With Reality-Based Paths

#### Goal

Stop treating the local `VisionEncoder` and `VisionProjector` as the default inference path once the real checkpoint path exists.

#### What This Means

After the real model adapter works:

- the custom local vision tower should no longer be the main inference route
- it may remain in the repo as a learning artifact or fallback experiment
- but it should not be confused with the authoritative Qwen2.5-VL path

#### Why This Matters

Keeping both paths is useful for learning, but only if their roles are explicit:

- one path is the simplified educational path
- the other path is the real-checkpoint inference path

#### Acceptance Check

The docs and code clearly distinguish:

- "educational simplified multimodal pipeline"
- "real Qwen2.5-VL inference pipeline"

---

## Concrete Next Steps For This Repo

These are the most actionable next steps from the roadmap above.

### Step A: Build And Save One Official Baseline Example

Create a small note or helper script that records:

- the `messages` object
- the chat-template output
- the image input tensors
- the generated answer

This becomes the reference sample for all later integration work.

### Step B: Upgrade `Processor` Toward The Official Contract

Refactor the processor so it is no longer centered on:

- custom placeholder generation only

and instead centered on:

- multimodal request preparation for Qwen2.5-VL

### Step C: Add A Qwen2.5-VL Adapter Layer

Do not yet replace the whole engine.
First create a thin adapter that allows the engine to call the real model.

### Step D: Run Side-By-Side Validation

For the same image/question pair, compare:

- official Transformers output
- local framework output

Do not move on until the two paths are structurally comparable, even if the performance characteristics are still different.

---

## Recommended Validation Order

Always validate in this order:

1. official baseline works
2. local processor output matches official expectations
3. local framework can call the real model
4. local framework produces a coherent answer for one image
5. text-only behavior still works where expected

This order prevents the most common debugging trap:

- trying to diagnose engine logic before confirming model contract correctness

---

## Common Failure Modes To Expect

### 1. Prompt Looks Reasonable But Is Still Wrong

A text prompt that "looks fine" is not enough.
What matters is whether it matches the official processor output exactly enough for the model family.

### 2. The Model Loads But The Visual Semantics Are Wrong

This usually means:

- image preprocessing differs from official behavior
- placeholder/token layout differs
- image-related tensors are incomplete or misaligned

### 3. The Framework Can Call The Model But Decode Semantics Drift

This means the serving framework assumptions are only partially compatible with the real VL model and must be revisited.

### 4. The Educational Path And Real Path Get Mixed Together

Once a real checkpoint path exists, the project must explicitly separate:

- experimental simplified path
- production-like real VL path

Otherwise debugging becomes much harder.

---

## Final Recommendation

The most important mindset shift is:

- do not keep extending the current fake-VL path as if it will naturally become Qwen2.5-VL

Instead:

- first make the framework faithful to the **official Qwen2.5-VL input contract**
- then make it capable of **hosting a real Qwen2.5-VL checkpoint**
- only after that decide whether deeper internal reimplementation is still worth doing

That is the clearest, safest, and most educational route from the current codebase to real image-conditioned VL inference.
