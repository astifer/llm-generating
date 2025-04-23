import torch
from typing import List

from main import model, tokenizer

class BeamCandidate:
    def __init__(self, input_ids: torch.Tensor, score: float):
        self.input_ids = input_ids
        self.score = score

    def __len__(self):
        return self.input_ids.size(-1)

    def get_normalized_score(self, length_penalty: float) -> float:
        return self.score / (len(self) ** length_penalty)

def beam_search_generate(
    prompt: str,
    eos_token_id: int = 151645,
    num_beams: int = 4,
    length_penalty: float = 1.0,
    max_tokens: int = 1000
) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")

    generated = inputs['input_ids'].clone()
    mask = inputs['attention_mask']


    input_len = generated.shape[-1]

    # Первоначальные кандидаты
    with torch.no_grad():

        outputs = model(input_ids=generated, attention_mask=mask)
        logits = outputs.logits[:, -1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        topk_probs, topk_ids = torch.topk(log_probs, num_beams, dim=-1)

    candidates: List[BeamCandidate] = []
    finished = []

    for i in range(num_beams):
        new_input_ids = torch.cat([generated, topk_ids[:, i].unsqueeze(-1)], dim=-1)
        candidates.append(BeamCandidate(new_input_ids, topk_probs[0, i].item()))

        tok = topk_ids[0, i].item()
        if tok == eos_token_id or tok == 151643:
            finished.append(candidates[-1])

    # Основной цикл
    while candidates and len(finished) < num_beams:
        new_candidates = []
        for candidate in candidates:
            with torch.no_grad():
                outputs = model(candidate.input_ids)
                logits = outputs.logits[:, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                topk_probs, topk_ids = torch.topk(log_probs, num_beams, dim=-1)

            for i in range(num_beams):
                next_token_id = topk_ids[:, i].unsqueeze(-1)
                new_input_ids = torch.cat([candidate.input_ids, next_token_id], dim=-1)
                score = candidate.score + topk_probs[0, i].item()

                new_candidate = BeamCandidate(new_input_ids, score)
                if next_token_id.item() == eos_token_id or next_token_id.item() == 151643:
                    finished.append(new_candidate)
                else:
                    new_candidates.append(new_candidate)

        # Ранжируем все кандидаты
        sorted_candidates = sorted(
            new_candidates, 
            key=lambda c: c.get_normalized_score(length_penalty), 
            reverse=True
        )
        candidates = sorted_candidates[:num_beams]

        if all(len(c.input_ids[0]) >= input_len + max_tokens for c in candidates):
            break  # Прекращаем по макс. длине

    # Выбираем лучший завершённый кандидат
    if finished:
        best = max(finished, key=lambda c: c.get_normalized_score(length_penalty))
    else:
        best = max(candidates, key=lambda c: c.get_normalized_score(length_penalty))

    return tokenizer.decode(best.input_ids[0][input_len:], skip_special_tokens=False)
