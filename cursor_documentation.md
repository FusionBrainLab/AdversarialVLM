# Документация проекта Universal Adversarial Attack

## Описание проекта
Исследовательская framework для создания и оценки адверсарных атак против мультимодальных больших языковых моделей (MLLMs). Проект направлен на обнаружение уязвимостей в системах ИИ для повышения их безопасности.

## Структура проекта

### Основные директории
- `src/` - основной исходный код
- `scripts/` - скрипты для запуска атак и оценки
- `tests/` - результаты тестов и экспериментов
- `runs/` - результаты запусков атак
- `datasets/` - наборы данных
- `logs/` - логи экспериментов
- `SafeBench_Text/` - тестовые данные SafeBench

### Основные компоненты

#### Основные модули атак (`src/`)
- `attack_model.py` - атаки на отдельные модели с различными методами ограничения и локализации
- `crossattack_models.py` - универсальные атаки, работающие на нескольких моделях одновременно
- `questions.py` - обработка вопросов и промптов
- `answers.py` - обработка ответов моделей

#### Система процессоров (`src/processors/`)
Специализированные обработчики для каждой модели:
- `phi3processor.py` - Phi-3.5 (динамическая обработка тайлов)
- `qwen2VLprocessor.py` - Qwen2-VL (динамическое масштабирование)
- `llama32processor.py` - Llama-3.2 (адаптивное разбиение на тайлы)
- `llavaprocessor.py` - LLaVA (простое масштабирование)
- `gemma3processor.py` - Gemma-3 (для оценки безопасности)
- `abstract_processor.py` - базовый класс для процессоров

#### Система оценки (`src/evaluation/`)
- `SafeBench_universal.py` - стандартная оценка на benchmark SafeBench
- `SafeBench_direct_paths.py` - версия с прямыми путями к изображениям (**НОВЫЙ**)
- `MM_SafetyBench_baseline.py` - оценка на MM-SafetyBench
- `FigStep_baseline.py` - оценка на FigStep dataset
- `find_best_iter_gemma.py` - автоматический поиск лучших итераций атак
- `guard_eval_gemma.py` - оценка безопасности с помощью Gemma-3
- `experiment_tracker.py` - анализ и визуализация результатов экспериментов

#### Судейская система (`src/judge/`)
SafetyChecker класс использует Gemma-3 со структурированным выходом для оценки безопасности контента.

### Поддерживаемые модели
- Qwen/Qwen2-VL-2B-Instruct (`qwenVL`)
- microsoft/Phi-3.5-vision-instruct (`phi35`)
- alpindale/Llama-3.2-11B-Vision-Instruct (`Llama32`)
- llava-hf/llava-1.5-7b-hf (`llava-hf`)

### Методы атак

#### Техники ограничения
- **Tanh Clamping** - сохраняющее градиент ограничение через tanh (по умолчанию)
- **Direct Clamping** - жесткие ограничения значений пикселей

#### Стратегии локализации
- **Corner Attack** - возмущения в угловых регионах n×n
- **Bottom Lines** - атаки на нижние k строк изображений
- **Random Square** - подвижные патчи n×n
- **Full Image** - традиционные возмущения всего изображения

#### Повышение робастности
- **Gaussian Blur** - размытие для улучшения переносимости
- **Local Cropping** - случайная обрезка
- **Multi-Model Training** - кросс-модельная оптимизация

### Рабочий процесс оценки
1. **Адверсарное обучение** - оптимизация пиксельных возмущений
2. **Выбор лучшей итерации** - использование Gemma-3 судьи
3. **Тестирование на benchmark** - оценка на SafeBench, MM-SafetyBench, FigStep
4. **Оценка безопасности** - количественная оценка через Attack Success Rate (ASR)

## Последние изменения

### 2024-12-XX: Решение проблемы градиентов в image processors для adversarial атак

Обнаружена и решена критическая проблема с потерей градиентов при использовании оригинальных image processors из Hugging Face в adversarial атаках. Проблема возникала из-за того, что processors преобразуют torch тензоры в numpy arrays, разрывая вычислительный граф.

#### Предложенные решения:

**1. SimpleGradientWrapper (рекомендуемый для продакшена)**
- Простой wrapper над оригинальным processor
- Автоматически определяет наличие torch тензоров с градиентами  
- Для torch тензоров использует F.interpolate + torch нормализацию
- Для обычных изображений делегирует оригинальному processor
- Полная обратная совместимость через `__getattr__`

**2. GradientAwareQwen2VLProcessor (для продвинутых случаев)**
- Наследует от оригинального `Qwen2VLImageProcessor`
- Реализует torch-версию препроцессинга с сохранением всех параметров
- Использует оригинальную логику `smart_resize` но с torch операциями
- Поддерживает все настройки оригинального процессора

**3. Manual Processing Functions**
- Прямые torch функции для resize и normalize
- Максимальный контроль над операциями
- Подходит для экспериментов и отладки

#### Использование:

```python
# Создание wrapper
from transformers import Qwen2VLImageProcessor
image_processor = Qwen2VLImageProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
gradient_processor = SimpleGradientWrapper(image_processor)

# Использование в adversarial атаке
tensor_image.requires_grad = True
processed = gradient_processor([tensor_image], return_tensors="pt")
loss = compute_loss(processed["pixel_values"])
loss.backward()  # Теперь градиенты корректно передаются в tensor_image
```

#### Интеграция в существующие процессоры:
Рекомендуется обновить существующие процессоры в `src/processors/` для использования `SimpleGradientWrapper`, что обеспечит корректную работу градиентов без изменения API.

### 2024-12-XX: Добавлен SafeBench_direct_paths.py
Создан новый скрипт `src/evaluation/SafeBench_direct_paths.py` для работы с прямыми путями:
- Принимает `--image_path` (полный путь к изображению) вместо `--exp` и `--iter`
- Принимает `--output_folder` (полная папка для сохранения) вместо построения из названия эксперимента
- Упрощает использование скрипта с произвольными изображениями и папками
- Исправлен импорт модулей для корректной работы при запуске из любой директории
- Исправлена проблема с phi35 моделью: добавлены `use_cache=False` и корректный `pad_token_id` для генерации

### 2024-12-XX: Добавлены скрипты тестирования jailbreaking baselines
Созданы автоматические скрипты тестирования baseline экспериментов:

#### `scripts/evaluation/test_jailbreaking_baselines.sh` (универсальный)
- Автоматически тестирует изображения из jailbreaking baseline экспериментов
- Создает 3 эксперимента с названиями `jailbreaking_attack_norm_{16,64,127}`
- В каждом эксперименте тестирует 4 модели (llama32, llava, phi3, qwen2vl)
- Использует изображения с разными нормами атак (16, 64, 127)
- Сохраняет результаты в структурированном виде в папке tests/

#### `scripts/evaluation/test_jailbreaking_baselines_gpu0.sh` (GPU 0)
- Тестирует только модели Llama32 и llava-hf на GPU 0
- Использует `CUDA_VISIBLE_DEVICES=0`
- Для параллельного выполнения с gpu1 скриптом

#### `scripts/evaluation/test_jailbreaking_baselines_gpu1.sh` (GPU 1)
- Тестирует только модели phi35 и qwenVL на GPU 1
- Использует `CUDA_VISIBLE_DEVICES=1`
- Для параллельного выполнения с gpu0 скриптом

#### `scripts/evaluation/test_jailbreaking_baselines_parallel.sh` (автоматический параллельный)
- Автоматически запускает оба GPU скрипта одновременно в фоне
- Сохраняет логи в папку logs/
- Ожидает завершения обеих частей и выводит итоговый статус

### Команды запуска

#### Основные атаки
```bash
# Атака на одну модель
./scripts/attacks/attack_clamp_tanh_llama.sh

# Универсальная кросс-модельная атака
./scripts/attacks/attack_cross.sh

# Локализованная атака с Gaussian blur
./scripts/attacks/attack_cross_gblur.sh
```

#### Оценка
```bash
# Найти лучшую итерацию
./scripts/evaluation/find_best_iter.sh

# Тест на SafeBench (старый способ)
./scripts/evaluation/safebench_test.sh experiment_name iteration model_suffix cuda_num

# Тест на SafeBench с прямыми путями (новый способ)
python src/evaluation/SafeBench_direct_paths.py \
    --image_path /path/to/image.png \
    --output_folder /path/to/output \
    --model_suf phi35 \
    --cuda_num 0

# Автоматическое тестирование jailbreaking baseline экспериментов
./scripts/evaluation/test_jailbreaking_baselines.sh [cuda_num]

# Параллельное тестирование на двух GPU (запускать в разных терминалах)
./scripts/evaluation/test_jailbreaking_baselines_gpu0.sh  # GPU 0: Llama32, llava-hf
./scripts/evaluation/test_jailbreaking_baselines_gpu1.sh  # GPU 1: phi35, qwenVL

# Если GPU 0 занята, можно запустить только на GPU 1:
./scripts/evaluation/test_jailbreaking_baselines_gpu1.sh

# Автоматический параллельный запуск (в одном терминале)
./scripts/evaluation/test_jailbreaking_baselines_parallel.sh

# Оценка безопасности
./scripts/evaluation/guard_eval.sh /path/to/results cuda_num
```

## Этические соображения
Проект предназначен исключительно для исследований безопасности ИИ. Использование для создания вредоносного контента строго запрещено.

## Технические требования
- Python 3.9+
- PyTorch
- Transformers
- Pandas, NumPy
- PIL/Pillow
- tqdm
- wandb (для логирования)

## Настройка окружения
1. Установить зависимости: `pip install -r requirements.txt`
2. Настроить WandB: `echo "your_wandb_key" > wandb_key.txt`
3. Убедиться в наличии CUDA для GPU-ускорения 