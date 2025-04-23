# LLM-generating

Using model Qwen2-0.5B

Vocab size is 151936

## Prompts

```python
input_text_hedgehog = '<|im_start|>system\nYou are a storyteller. Generate a story based on user message.<|im_end|>\n<|im_start|>user\nGenerate me a short story about a tiny hedgehog named Sonic.<|im_end|>\n<|im_start|>assistant\n'
input_text_json = '<|im_start|>system\nYou are a strict JSON machine. Generate only a JSON with format {"contractor": name, "sum": decimal, "currency": currency code} based on user message. You prohibited to discuss. Only return dict in json format.<|im_end|>\n<|im_start|>user\nTransfer 100 rubles to Mike<|im_end|>\n<|im_start|>assistant\n'
```

Везде генерим текст, до тех пор пока
- сгенерировался EOS-токен с ID = 151645
- длина генерации превысила 1000 токенов.

## Greedy Decoding

На каждом этапе выбираем самый вероятный токен
- Результаты не отличаются
- Сказка получается однообразная:

- При генерации json, модель слишком долго рассуждает и вываливает много не нужно текста, даже при прямом указании ей этого не делать. Как итог в будущем: в неоднозначных запросах модель может неверно "прилипнуть" к некорректной интерпретации. Если чуть поизменять промт, можно добиться нужного результата


Source: [code](greedy_decoding.py)

## Sampling

На каждом шаге генерации получаем из модели распределение вероятностей для следующего токена. Выбираем не самый вероятный токен, а делаем сэмплирование среди всех токенов из этого распределения

- Крайне не устойичвая вещь со множеством мусора и кучей непонятных символов
- json так и не получили

Source: [code](sampling.py)

## Samplin with temperature

Используем идею семплинга, но добавляем температуру из диапазона [0.001, 0.1, 0.5, 1.0, 10.0]

Source: [code](sampling.py)

## Nucleus sampling

Как прошлая задача, но сэмплируем не из всего распределения, а только среди самых вероятных токенов.

- Имеем распределение вероятностей по токенам, полученное от модели с применённой температурой.
- Оставляем только самые вероятные токены, кумулятивная вероятность которых не превышает top_p. Остальные выбрасываем. Если так получилось, что самый вероятный токен уже имеет вероятность больше, чем top_p, - оставляем только его.
- Так как вектор из вероятностей теперь не совсем распределение (значения не суммируются в 1), то отмасштабируем каждую вероятность, разделив её на сумму всех вероятностей. Теперь значения суммируется в 1.
- Выполняем сэмплирование

Source: [code](nucleus_sampling.py)

## Beam Search

Source [code](early_stopped_beam_search.py)
