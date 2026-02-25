from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载Qwen3 0.6B模型和tokenizer
model_name = "Qwen/Qwen3-0.6B"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
# 输出模型结构
print(model)