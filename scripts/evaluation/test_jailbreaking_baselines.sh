#!/bin/bash

# Скрипт для тестирования изображений из jailbreaking baseline экспериментов
# Каждый эксперимент содержит тесты для всех 4 моделей с 3 разными нормами атак

# Базовые пути
BASE_DIR="/home/jovyan/rahmatullaev/adversarial_baselines/Jailbreaking-Attack-against-Multimodal-Large-Language-Model"
OUTPUT_BASE="/home/jovyan/rahmatullaev/adversarial/tests"
SCRIPT_PATH="/home/jovyan/rahmatullaev/adversarial/src/evaluation/SafeBench_direct_paths.py"

# Соответствие между папками и model_suf
declare -A MODEL_MAPPING
MODEL_MAPPING["llama32"]="Llama32"
MODEL_MAPPING["llava"]="llava-hf"
MODEL_MAPPING["phi3"]="phi35"
MODEL_MAPPING["qwen2vl"]="qwenVL"

# Нормы атак
declare -A NORM_MAPPING
NORM_MAPPING["16"]="16"
NORM_MAPPING["64"]="64"
NORM_MAPPING["127"]="127"

# CUDA устройство (можно изменить)
CUDA_NUM=${1:-0}

echo "=== Запуск тестирования jailbreaking baseline экспериментов ==="
echo "Используется CUDA устройство: $CUDA_NUM"
echo ""

# Проходим по каждой норме
for exp_norm in 16 64 127; do
    file_norm=${NORM_MAPPING[$exp_norm]}
    exp_name="jailbreaking_attack_norm_${exp_norm}"
    
    echo "=== Обработка эксперимента: $exp_name ==="
    
    # Проходим по каждой модели
    for model_dir in llama32 llava phi3 qwen2vl; do
        model_suf=${MODEL_MAPPING[$model_dir]}
        
        # Пути к файлам
        image_path="${BASE_DIR}/${model_dir}_Mprompt_adv_img/train_size_25_PGD_${file_norm}_200_iters_pure_noise_A.png"
        output_folder="${OUTPUT_BASE}/${exp_name}/${model_suf}"
        
        echo "  Тестирование модели: $model_suf"
        echo "    Изображение: $image_path"
        echo "    Выходная папка: $output_folder"
        
        # Проверяем существование изображения
        if [[ ! -f "$image_path" ]]; then
            echo "    ОШИБКА: Изображение не найдено: $image_path"
            continue
        fi
        
        # Создаем выходную папку
        mkdir -p "$output_folder"
        
        # Запускаем тестирование
        echo "    Запуск SafeBench_direct_paths.py..."
        python "$SCRIPT_PATH" \
            --image_path "$image_path" \
            --output_folder "$output_folder" \
            --model_suf "$model_suf" \
            --cuda_num "$CUDA_NUM"
        
        if [[ $? -eq 0 ]]; then
            echo "    ✓ Тестирование $model_suf завершено успешно"
        else
            echo "    ✗ Ошибка при тестировании $model_suf"
        fi
        echo ""
    done
    
    echo "=== Эксперимент $exp_name завершен ==="
    echo ""
done

echo "=== Все тестирования завершены ==="
echo ""
echo "Результаты сохранены в:"
for exp_norm in 16 64 127; do
    echo "  - ${OUTPUT_BASE}/jailbreaking_attack_norm_${exp_norm}/"
done 