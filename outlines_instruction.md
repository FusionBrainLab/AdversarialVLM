# Инструкция по использованию Outlines для структурированной генерации в LL модели

## Основные компоненты

1. **Импорт необходимых библиотек**:
```python
from pydantic import BaseModel, Field
import outlines
from outlines.models import Transformers
from outlines.generate import json as generate_json
```

2. **Определение схемы данных с Pydantic**:
```python
class ExtractedWords(BaseModel):
    """Schema for extracted words output"""
    words: list[str] = Field(description="List of extracted English words, phrases or collocations")
```

3. **Инициализация модели для структурированного вывода**:
```python
# Предполагается, что model и tokenizer уже загружены
outlines_model = Transformers(model, tokenizer)
extract_words_generator = generate_json(outlines_model, ExtractedWords)
```

4. **Использование модели для генерации структурированного вывода**:
```python
messages = [
    {"role": "system", "content": "System prompt..."},
    {"role": "user", "content": "User message..."}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
result = extract_words_generator(prompt)
# Доступ к данным через атрибуты
extracted_words = result.words
```

## Пример полного цикла

```python
# 1. Определите схему данных
class WordInfo(BaseModel):
    word: str = Field(description="The word")
    definition: str = Field(description="Definition of the word")
    examples: list[str] = Field(description="Examples of usage")

# 2. Создайте генератор
word_info_generator = generate_json(outlines_model, WordInfo)

# 3. Подготовьте промпт
messages = [
    {"role": "system", "content": "Provide information about the word"},
    {"role": "user", "content": f"Word: {word}"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# 4. Получите структурированный результат
result = word_info_generator(prompt)

# 5. Используйте данные
print(f"Word: {result.word}")
print(f"Definition: {result.definition}")
print("Examples:")
for example in result.examples:
    print(f"- {example}")
```

## Обработка ошибок

```python
try:
    result = word_info_generator(prompt)
    # Доступ к данным
    return result
except Exception as e:
    logger.error(f"Ошибка при генерации структурированной информации: {e}")
    # Fallback на обычный метод генерации
    return None
```

Outlines позволяет получать структурированный вывод от LLM, что упрощает обработку ответов и делает работу с данными более надежной и предсказуемой. 