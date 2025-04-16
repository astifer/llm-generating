from main import model, tokenizer, input_text_hedgehog, input_text_json
import torch

@torch.no_grad()
def greedy_decode(prompt_text, eos_token_id=151645, max_new_tokens=1000):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    generated = inputs['input_ids'].clone()
    mask = inputs['attention_mask']

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=mask)
        next_token_logits = outputs.logits[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # (1, 1)


        generated = torch.cat([generated, next_token_id], dim=-1)

        if next_token_id.item() == eos_token_id or next_token_id.item() == 151643:
            break

    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    return generated_text[len(prompt_text):].strip()

for _ in range(3):
    res = greedy_decode(input_text_hedgehog)
    print(res)
    

for _ in range(3):
    res = greedy_decode(input_text_json)
    print(res)