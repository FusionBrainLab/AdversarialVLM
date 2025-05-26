# Evaluation Module

Модуль оценки содержит универсальные скрипты для тестирования и анализа результатов адверсариальных атак.

## Скрипты

### `find_best_iter_gemma.py`
**Назначение:** Поиск лучшей итерации атаки с помощью модели-судьи Gemma-3.

**Функциональность:**
- Анализирует все итерации обучения атаки из папки runs/
- Использует Gemma-3-4b-it для оценки безопасности сгенерированных ответов
- Находит итерацию с максимальным ASR (Attack Success Rate)
- Создает графики динамики метрик
- Сохраняет детальные результаты оценки

**Использование:**
```bash
python src/evaluation/find_best_iter_gemma.py
```

### `SafeBench_universal.py`
**Назначение:** Универсальное тестирование атакованных изображений на датасете SafeBench.

**Функциональность:**
- Поддерживает все основные модели (Phi-3.5, Qwen2-VL, Llama-3.2, LLaVA)
- Генерирует ответы атакованной модели на вопросы SafeBench
- Сохраняет результаты в CSV формате

**Использование:**
```bash
python src/evaluation/SafeBench_universal.py --exp experiment_name --iter iteration_number --model_suf model_suffix --cuda_num gpu_number
```

**Параметры:**
- `--exp`: Имя эксперимента
- `--iter`: Номер итерации
- `--model_suf`: Суффикс модели (phi35, qwenVL, Llama32, llava-hf)
- `--cuda_num`: Номер GPU

### `guard_eval_gemma.py`
**Назначение:** Оценка результатов SafeBench с помощью модели-судьи.

**Функциональность:**
- Оценивает результаты тестирования SafeBench
- Использует Gemma-3-4b-it для классификации безопасности
- Вычисляет финальный ASR
- Сохраняет детальные результаты оценки

**Использование:**
```bash
python src/evaluation/guard_eval_gemma.py /path/to/safebench/results cuda_number
```

### `MM_SafetyBench_baseline.py`
**Назначение:** Базовое тестирование на датасете MM-SafetyBench.

**Функциональность:**
- Тестирование моделей на MM-SafetyBench датасете
- Поддержка различных типов изображений (SD, TYPO, SD_TYPO)
- Генерация ответов по категориям

**Использование:**
```bash
python src/evaluation/MM_SafetyBench_baseline.py --model_suf model_suffix --cuda_num gpu_number --image_type image_type
```

### `FigStep_baseline.py`
**Назначение:** Базовое тестирование на датасете FigStep.

**Функциональность:**
- Тестирование моделей на FigStep датасете
- Использует стандартный промпт FigStep
- Генерация ответов по категориям

**Использование:**
```bash
python src/evaluation/FigStep_baseline.py --model_suf model_suffix --cuda_num gpu_number
```

### `benchmarkign.py`
**Назначение:** Бенчмаркинг моделей в различных режимах.

**Функциональность:**
- Тестирование в режимах: reference, shii, gcg
- Поддержка различных моделей
- Анализ защищенности моделей

**Использование:**
```bash
python src/evaluation/benchmarkign.py --model model_name --mode test_mode --device gpu_device
```

## Зависимости

Все скрипты требуют:
- Доступ к папке `src/` для импорта процессоров и судьи
- Настроенное окружение с установленными зависимостями
- Доступ к соответствующим датасетам

## Примечания

- Скрипты автоматически добавляют путь к `src/` в sys.path
- Результаты сохраняются в соответствующих папках экспериментов
- Для работы с WandB требуется настроенный API ключ 