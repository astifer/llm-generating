from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B") 
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B",
    torch_dtype=torch.bfloat16
)
model.eval()
input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a JSON machine. Generate a JSON with format {"contractor": string with normalized contractor name, "sum": decimal, "currency": string with uppercased 3-letter currency code} based on user message.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles and 50 kopeck to Mike<|im_end|>\n<|im_start|>assistant\n'

inputs = tokenizer(input_text_hedgehog, return_tensors="pt").to("cpu") 

with torch.no_grad():
    logits = model( 
    **inputs
        ).logits

    print(logits.shape)