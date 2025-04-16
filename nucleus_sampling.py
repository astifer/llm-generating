from main import model, tokenizer, input_text_hedgehog, input_text_json
import torch

import torch.nn.functional as F

@torch.no_grad()
def nucleus_sampling(prompt_text, top_p=0.9, temperature=1.0, max_new_tokens=10000, eos_token_id=151645):
    inputs = tokenizer(prompt_text, return_tensors="pt")
    
    generated = inputs['input_ids'].clone()
    attention_mask = inputs['attention_mask']

    for _ in range(max_new_tokens):
        outputs = model(input_ids=generated, attention_mask=attention_mask)
        logits = outputs.logits[:, -1, :]
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)

        # сорь по убыванию
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # оставляем только топ P
        mask = cumulative_probs <= top_p

        """
        Когда мы применяем nucleus sampling, мы сортируем токены по убыванию вероятности и включаем в выборку только те, чья кумулятивная вероятность не превышает top_p. Например, если top_p = 0.9, мы включаем только те токены, суммарная вероятность которых ≤ 90%.
        НО! Иногда бывает, что первый (наиболее вероятный) токен сам по себе уже больше top_p (например, он имеет вероятность 0.95). В таком случае, если строго следовать правилу <= top_p, мы бы исключили даже самый вероятный токен, и список допустимых токенов оказался бы пустым.
        Всегда включаем самый вероятный токен
        """
        mask[..., 0] = True  


        filtered_probs = sorted_probs[mask]
        filtered_indices = sorted_indices[mask]

        # норм
        normalized_probs = filtered_probs / filtered_probs.sum()

        sampled_index = torch.multinomial(normalized_probs, num_samples=1)
        next_token_id = filtered_indices[sampled_index].unsqueeze(0)

        generated = torch.cat([generated, next_token_id], dim=-1)

        if next_token_id.item() == eos_token_id or next_token_id.item() == 151643:
            break

    full_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    return full_text[len(prompt_text):].strip()


for t, p in [(1, 0.9), (1, 0.15), (0.5, 0.9), (0.5, 0.15)]:
    res = nucleus_sampling(input_text_json, temperature=t, top_p=p)
    print(f"RESULT FOR t={t}")
    print(res)

for t, p in [(1, 0.9), (1, 0.15), (0.5, 0.9), (0.5, 0.15)]:
    res = nucleus_sampling(input_text_hedgehog, temperature=t, top_p=p)
    print(f"RESULT FOR t={t}")
    print(res)