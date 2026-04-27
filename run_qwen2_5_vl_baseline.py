import torch
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

model_name = "Qwen/Qwen2.5-VL-3B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

processor = AutoProcessor.from_pretrained(model_name)
img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images", "01", "num.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img_path},
            {"type": "text", "text": "Describe this image briefly."},
        ],
    }
]
from pprint import pprint
print("\n=== Raw messages ===")
pprint(messages)

#做一版带模板但没编码的文本，用于对比编码结果
untokenize_inputs = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)
print("\n=== Rendered prompt text ===")
print(inputs)# 其中image_grid_thw是图片patch维度
print("vision_start_token_id:", model.config.vision_start_token_id)
print("vision_end_token_id:", model.config.vision_end_token_id)
print("image_token_id:", model.config.image_token_id)

print("\n=== Token===")
print(untokenize_inputs)

print("\n=== Final model inputs ===")
for k, v in inputs.items():
    if hasattr(v, "shape"):
        print(f"{k}: shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device}")
    else:
        print(f"{k}: type={type(v)}")

generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
print("\n=== model generate ===")
print(generated_ids)
print(generated_ids_trimmed)

print("\n=== Generation lengths ===")
print("input_ids shape:", tuple(inputs["input_ids"].shape))
print("generated_ids shape:", tuple(generated_ids.shape))
print("trimmed completion lengths:", [len(x) for x in generated_ids_trimmed])

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)
print(output_text)
