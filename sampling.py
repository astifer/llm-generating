from main import model, tokenizer, input_text_hedgehog, input_text_json
import torch
import torch.nn.functional as F

@torch.no_grad()
def sampling_decode(prompt_text, eos_token_id=151645, max_new_tokens=1000, temperature=1.0):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    generated = inputs['input_ids'].clone()
    mask = inputs['attention_mask']

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=mask)
        logits = outputs.logits[:, -1, :] 


        logits = logits / temperature


        probabilities = F.softmax(logits, dim=-1)


        next_token_id = torch.multinomial(probabilities, num_samples=1)

        generated = torch.cat([generated, next_token_id], dim=-1)


        if next_token_id.item() == eos_token_id or next_token_id.item() == 151643:
            break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    return generated_text[len(prompt_text):].strip()

for t in [0.001, 0.1, 0.5, 1.0, 10.0]:
    res = sampling_decode(input_text_hedgehog, temperature=t)
    print(f"RESULT FOR T={t}")
    print(res)

for t in [0.001, 0.1, 0.5, 1.0, 10.0]:
    res = sampling_decode(input_text_json, temperature=t)
    print(f"RESULT FOR T={t}")
    print(res)
