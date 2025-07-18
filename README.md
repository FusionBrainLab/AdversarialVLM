# MLLM-adversarial

Исследовательский проект для создания и тестирования адверсариальных атак на мультимодальные языковые модели (MLLM).

## 🎯 Описание

**MLLM-adversarial** позволяет обучать адверсариальные изображения, которые заставляют мультимодальные модели генерировать небезопасный контент, и оценивать эффективность таких атак. Проект поддерживает атаки на популярные модели: Phi-3.5, Qwen2-VL, Llama-3.2, LLaVA и использует Gemma-3 в качестве модели-судьи для оценки безопасности.

## 🚀 Быстрый старт

### Настройка WandB

```bash
# Добавьте ваш API ключ WandB в файл
echo "your_wandb_key" > wandb_key.txt
```

### Запуск атаки

```bash
# Одиночная атака на Llama-3.2
./scripts/attacks/attack_clamp_tanh_llama.sh

# Кросс-атака на несколько моделей
./scripts/attacks/attack_cross.sh
```

### Оценка результатов

```bash
# 1. Поиск лучшей итерации
./scripts/evaluation/find_best_iter.sh

# 2. Тестирование на SafeBench
./scripts/evaluation/safebench_test.sh experiment_name iteration model_suffix cuda_num

# 3. Оценка результатов судьей
./scripts/evaluation/guard_eval.sh /path/to/results cuda_num
```

## 📁 Структура проекта

```
MLLM-adversarial/
├── src/                          # Основной код проекта
│   ├── attack_model.py           # Атака на одну модель
│   ├── crossattack_models.py     # Кросс-атака на несколько моделей
│   ├── questions.py              # Наборы вопросов
│   ├── answers.py                # Целевые ответы
│   ├── processors/               # Процессоры для разных моделей
│   ├── judge/                    # Модуль оценки безопасности
│   └── evaluation/               # Модули оценки и тестирования
├── scripts/                      # Скрипты запуска
│   ├── attacks/                  # Скрипты атак
│   └── evaluation/               # Скрипты оценки
├── datasets/                     # Датасеты (SafeBench, MM-SafetyBench, FigStep)
├── runs/                         # Результаты обучения атак
├── tests/                        # Результаты экспериментов
└── images/                       # Исходные изображения
```

## 🔧 Основные компоненты

### Модуль атак

- **`attack_model.py`** - Атака на одну модель с поддержкой различных методов ограничения и маскирования
- **`crossattack_models.py`** - Универсальные атаки на несколько моделей одновременно

### Система процессоров

Специализированные процессоры для каждой модели:
- **Phi-3.5** (`phi3processor.py`)
- **Qwen2-VL** (`qwen2VLprocessor.py`) 
- **Llama-3.2** (`llama32processor.py`)
- **LLaVA** (`llavaprocessor.py`)
- **Gemma-3** (`gemma3processor.py`) - только для оценки

### Модуль судьи

`SafetyChecker` класс использует Gemma-3 для оценки безопасности сгенерированного текста с структурированным выводом.

### Модуль оценки

- **`find_best_iter_gemma.py`** - Поиск лучшей итерации атаки
- **`SafeBench_universal.py`** - Универсальное тестирование на SafeBench
- **`guard_eval_gemma.py`** - Оценка результатов судьей
- **`experiment_tracker.py`** - Отслеживание и анализ экспериментов

## 📊 Рабочий процесс

1. **Обучение адверсариального изображения** - оптимизация пикселей для максимизации вероятности генерации целевого текста
2. **Поиск лучшей итерации** - использование модели-судьи для выбора наиболее эффективного изображения  
3. **Тестирование на SafeBench** - оценка атаки на стандартном датасете безопасности
4. **Финальная оценка судьей** - количественная оценка успешности атаки (ASR)

## 🛠️ Параметры атак

### Методы ограничения
- `tanh` - ограничение через tanh функцию (по умолчанию)
- `clamp` - прямое ограничение значений

### Локализация атаки
- `corner` - атака только на угол изображения n×n
- `bottom_lines` - атака только на k нижних строк  
- `random_square` - атака на случайно расположенный квадрат n×n

### Аугментация
- `--use_gaussian_blur` - размытие по Гауссу для повышения робастности
- `--use_local_crop` - случайное кропирование для универсальности

## 📈 Отслеживание экспериментов

Используйте `ExperimentTracker` для анализа результатов:

```python
from src.evaluation.experiment_tracker import ExperimentTracker

# Инициализация трекера
tracker = ExperimentTracker()

# Список всех экспериментов
experiments = tracker.list_experiments()

# Информация о конкретном эксперименте
info = tracker.get_experiment_info("gray_Llama_20250121_110131")

# График динамики ASR
tracker.plot_asr_dynamics("gray_Llama_20250121_110131")

# Сводные таблицы
runs_summary = tracker.get_runs_summary()
tests_summary = tracker.get_tests_summary()
```

## 🎯 Поддерживаемые модели

- **Microsoft Phi-3.5-vision-instruct**
- **Qwen/Qwen2-VL-*B-Instruct** 
- **Llama-3.2-11B-Vision-Instruct**
- **llava-hf/llava-1.5-7b-hf**
- **google/gemma-3-4b-it** (судья)

## 📋 Датасеты

- **SafeBench** - основной датасет для оценки безопасности
- **MM-SafetyBench** - расширенный датасет безопасности
- **FigStep** - датасет для тестирования понимания изображений

## 🔍 Примеры использования

### Атака с маскированием угла

```bash
python src/attack_model.py \
    --model_name "microsoft/Phi-3.5-vision-instruct" \
    --epsilon 0.3 \
    --mask_type corner \
    --mask_size 50 \
    --clamp_method tanh
```

### Кросс-атака с размытием

```bash
python src/crossattack_models.py \
    --models "microsoft/Phi-3.5-vision-instruct,Qwen/Qwen2-VL-7B-Instruct" \
    --epsilon 0.4 \
    --use_gaussian_blur \
    --clamp_method tanh
```

### Поиск лучшей итерации

```bash
python src/evaluation/find_best_iter_gemma.py \
    --experiment_name "gray_Llama_20250121_110131"
```

### Тестирование на SafeBench

```bash
python src/evaluation/SafeBench_universal.py \
    --exp "gray_Llama_20250121_110131" \
    --iter 3500 \
    --model_suf "Llama32" \
    --cuda_num 0
```

## 📚 Документация

- **[DOC.md](DOC.md)** - Подробная техническая документация
- **[src/evaluation/README.md](src/evaluation/README.md)** - Документация модуля оценки
- **[src/judge/README.md](src/judge/README.md)** - Документация модуля судьи

## 🤝 Вклад в проект

1. Форкните репозиторий
2. Создайте ветку для новой функции (`git checkout -b feature/amazing-feature`)
3. Зафиксируйте изменения (`git commit -m 'Add amazing feature'`)
4. Отправьте в ветку (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## ⚠️ Этические соображения

Этот проект предназначен исключительно для исследовательских целей в области безопасности ИИ. Использование для создания вредоносного контента строго запрещено.

## 📄 Лицензия

Проект предназначен для академических исследований. Подробности использования уточняйте у авторов.

## 🙏 Благодарности

- Команде Hugging Face за предоставление моделей
- Разработчикам датасетов SafeBench, MM-SafetyBench и FigStep
- Сообществу исследователей безопасности ИИ 