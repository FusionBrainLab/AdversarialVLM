#!/bin/bash

# Скрипт для параллельного запуска тестирования jailbreaking baseline экспериментов на двух GPU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/home/jovyan/rahmatullaev/adversarial/logs"

# Создаем папку для логов если не существует
mkdir -p "$LOG_DIR"

echo "=== Запуск параллельного тестирования jailbreaking baseline экспериментов ==="
echo "GPU 0: Llama32, llava-hf"
echo "GPU 1: phi35, qwenVL"
echo ""
echo "Логи сохраняются в:"
echo "  - GPU 0: $LOG_DIR/jailbreaking_gpu0.log"
echo "  - GPU 1: $LOG_DIR/jailbreaking_gpu1.log"
echo ""

# Запускаем оба скрипта в фоне
echo "Запуск тестирования на GPU 0..."
"$SCRIPT_DIR/test_jailbreaking_baselines_gpu0.sh" > "$LOG_DIR/jailbreaking_gpu0.log" 2>&1 &
GPU0_PID=$!

echo "Запуск тестирования на GPU 1..."
"$SCRIPT_DIR/test_jailbreaking_baselines_gpu1.sh" > "$LOG_DIR/jailbreaking_gpu1.log" 2>&1 &
GPU1_PID=$!

echo ""
echo "Процессы запущены:"
echo "  - GPU 0 PID: $GPU0_PID"
echo "  - GPU 1 PID: $GPU1_PID"
echo ""
echo "Для отслеживания прогресса:"
echo "  tail -f $LOG_DIR/jailbreaking_gpu0.log"
echo "  tail -f $LOG_DIR/jailbreaking_gpu1.log"
echo ""

# Ждем завершения обоих процессов
echo "Ожидание завершения тестирования..."
wait $GPU0_PID
GPU0_STATUS=$?

wait $GPU1_PID
GPU1_STATUS=$?

echo ""
echo "=== Результаты выполнения ==="
if [ $GPU0_STATUS -eq 0 ]; then
    echo "✓ GPU 0 (Llama32, llava-hf): Завершено успешно"
else
    echo "✗ GPU 0 (Llama32, llava-hf): Завершено с ошибкой (код: $GPU0_STATUS)"
fi

if [ $GPU1_STATUS -eq 0 ]; then
    echo "✓ GPU 1 (phi35, qwenVL): Завершено успешно"
else
    echo "✗ GPU 1 (phi35, qwenVL): Завершено с ошибкой (код: $GPU1_STATUS)"
fi

echo ""
echo "Результаты сохранены в /home/jovyan/rahmatullaev/adversarial/tests/"
echo "Логи доступны в $LOG_DIR/" 