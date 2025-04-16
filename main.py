from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B") 
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    torch_dtype=torch.bfloat16
).eval()

input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a strict JSON machine. Generate only a JSON with format {"contractor": name, "sum": decimal, "currency": currency code} based on user message. You prohibited to discuss. Only return dict in json format.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles to Mike<|im_end|>\n<|im_start|>assistant\n'
