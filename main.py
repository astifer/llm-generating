from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_choice = "rugpt"  # Change this to "qwen" or "rugpt" as needed

if model_choice == "qwen":
    model_name = "Qwen/Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    model.eval()

    input_text = (
        "<|im_end|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n"
        "<|im_end|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n"
        "<|im_end|>assistant\n"
    )

elif model_choice == "rugpt":
    model_name = "ai-forever/rugpt3.5"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    input_text = "Сгенерируй короткую историю про маленького ежа по имени Соник:\n"

else:
    raise ValueError("Unknown model choice. Use 'qwen' or 'rugpt'.")

inputs = tokenizer(input_text, return_tensors="pt").to("cpu")

with torch.no_grad():
    logits = model(**inputs).logits
    print(f"Logits shape: {logits.shape}")
