import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import re
from collections import defaultdict


class ExperimentTracker:
    """
    Класс для отслеживания и анализа результатов адверсариальных экспериментов.
    
    Автоматически сканирует папки runs/ и tests/ для получения информации о:
    - Количестве шагов обучения
    - Лучших итерациях и метриках
    - Результатах тестирования на SafeBench
    - Графиках изменения ASR
    - Оценка безопасности
    """
    
    def __init__(self, runs_dir: str = "./runs", tests_dir: str = "./tests"):
        """
        Инициализация трекера экспериментов.
        
        Args:
            runs_dir: Путь к папке с результатами обучения
            tests_dir: Путь к папке с результатами тестирования
        """
        self.runs_dir = Path(runs_dir)
        self.tests_dir = Path(tests_dir)
        
        # Сканируем доступные эксперименты
        self.runs_experiments = self._scan_runs_experiments()
        self.tests_experiments = self._scan_tests_experiments()
        
        print(f"Найдено {len(self.runs_experiments)} экспериментов в runs/")
        print(f"Найдено {len(self.tests_experiments)} результатов тестирования в tests/")
    
    def _scan_runs_experiments(self) -> Dict[str, Dict]:
        """Сканирует папку runs/ для получения информации об экспериментах."""
        experiments = {}
        
        if not self.runs_dir.exists():
            return experiments
            
        for exp_dir in self.runs_dir.iterdir():
            if exp_dir.is_dir() and exp_dir.name.startswith('gray_'):
                exp_info = {
                    'path': exp_dir,
                    'steps': self._get_training_steps(exp_dir),
                    'has_safety_details': self._has_safety_details(exp_dir),
                    'best_step': None,
                    'best_metric': None
                }
                
                # Проверяем наличие лучшего шага
                if exp_info['has_safety_details']:
                    best_step, best_metric = self._get_best_step_info(exp_dir)
                    exp_info['best_step'] = best_step
                    exp_info['best_metric'] = best_metric
                
                experiments[exp_dir.name] = exp_info
                
        return experiments
    
    def _scan_tests_experiments(self) -> Dict[str, Dict]:
        """Сканирует папку tests/ для получения информации о тестированиях."""
        experiments = defaultdict(dict)
        
        if not self.tests_dir.exists():
            return dict(experiments)
            
        # Ищем папки вида {exp_name}_{step}
        pattern = re.compile(r'^(.+)_(\d+)$')
        
        for test_dir in self.tests_dir.iterdir():
            if test_dir.is_dir():
                match = pattern.match(test_dir.name)
                if match:
                    exp_name, step = match.groups()
                    step = int(step)
                    
                    if exp_name not in experiments:
                        experiments[exp_name] = {}
                    
                    experiments[exp_name][step] = {
                        'path': test_dir,
                        'models': self._get_model_results(test_dir)
                    }
        
        return dict(experiments)
    
    def _get_training_steps(self, exp_dir: Path) -> int:
        """Получает количество шагов обучения из файлов optimized_image_iter_*.png."""
        pattern = exp_dir / "optimized_image_iter_*.png"
        files = glob.glob(str(pattern))
        
        if not files:
            return 0
            
        # Извлекаем номера итераций из имен файлов
        steps = []
        for file in files:
            match = re.search(r'optimized_image_iter_(\d+)\.png', file)
            if match:
                steps.append(int(match.group(1)))
        
        return max(steps) if steps else 0
    
    def _has_safety_details(self, exp_dir: Path) -> bool:
        """Проверяет наличие папки safety_details."""
        return (exp_dir / "safety_details").exists()
    
    def _get_best_step_info(self, exp_dir: Path) -> Tuple[Optional[int], Optional[float]]:
        """Получает информацию о лучшем шаге из safety_details."""
        safety_dir = exp_dir / "safety_details"
        
        # Ищем файл best_iter.txt в подпапках
        for subdir in safety_dir.iterdir():
            if subdir.is_dir():
                best_file = subdir / "best_iter.txt"
                if best_file.exists():
                    try:
                        with open(best_file, 'r') as f:
                            best_step = int(f.read().strip())
                        
                        # Пытаемся найти метрику из unsafe_metrics_models.csv
                        metrics_file = exp_dir / "unsafe_metrics_models.csv"
                        if metrics_file.exists():
                            df = pd.read_csv(metrics_file, index_col=0)
                            if 'ALL_MODELS_MEAN' in df.columns and best_step in df.index:
                                best_metric = df.loc[best_step, 'ALL_MODELS_MEAN']
                                return best_step, best_metric
                        
                        return best_step, None
                    except (ValueError, FileNotFoundError):
                        continue
        
        return None, None
    
    def _get_model_results(self, test_dir: Path) -> Dict[str, Dict]:
        """Получает результаты для всех моделей в папке тестирования."""
        models = {}
        
        for model_dir in test_dir.iterdir():
            if model_dir.is_dir():
                model_name = model_dir.name
                models[model_name] = {
                    'has_inference': self._has_inference_results(model_dir),
                    'asr': self._get_asr_result(model_dir)
                }
        
        return models
    
    def _has_inference_results(self, model_dir: Path) -> bool:
        """Проверяет наличие результатов инференса (23 CSV файла)."""
        csv_files = list(model_dir.glob("*.csv"))
        return len(csv_files) >= 23
    
    def _get_asr_result(self, model_dir: Path) -> Optional[float]:
        """Получает ASR результат из mean_result_gemma.txt."""
        result_file = model_dir / "mean_result_gemma.txt"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    content = f.read().strip()
                    return float(content)
            except (ValueError, FileNotFoundError):
                pass
        return None
    
    def get_experiment_info(self, exp_name: str, step: Optional[int] = None) -> Dict:
        """
        Получает полную информацию об эксперименте.
        
        Args:
            exp_name: Имя эксперимента
            step: Конкретный шаг (опционально)
            
        Returns:
            Словарь с информацией об эксперименте
        """
        info = {
            'experiment': exp_name,
            'runs_info': None,
            'tests_info': None
        }
        
        # Информация из runs/
        if exp_name in self.runs_experiments:
            info['runs_info'] = self.runs_experiments[exp_name].copy()
        
        # Информация из tests/
        if exp_name in self.tests_experiments:
            if step is not None:
                if step in self.tests_experiments[exp_name]:
                    info['tests_info'] = {step: self.tests_experiments[exp_name][step]}
                else:
                    info['tests_info'] = {}
            else:
                info['tests_info'] = self.tests_experiments[exp_name]
        
        return info
    
    def get_step_metric(self, exp_name: str, step: int) -> Optional[float]:
        """
        Получает метрику для конкретного шага эксперимента.
        
        Args:
            exp_name: Имя эксперимента
            step: Номер шага
            
        Returns:
            Метрика ASR или None если не найдена
        """
        if exp_name not in self.runs_experiments:
            return None
            
        exp_dir = self.runs_experiments[exp_name]['path']
        metrics_file = exp_dir / "unsafe_metrics_models.csv"
        
        if not metrics_file.exists():
            return None
            
        try:
            df = pd.read_csv(metrics_file, index_col=0)
            if 'ALL_MODELS_MEAN' in df.columns and step in df.index:
                return df.loc[step, 'ALL_MODELS_MEAN']
        except Exception:
            pass
            
        return None
    
    def get_asr_by_step(self, exp_name: str) -> pd.DataFrame:
        """
        Получает динамику ASR по шагам.
        """
        if exp_name not in self.runs_experiments:
            return None
            
        exp_dir = self.runs_experiments[exp_name]['path']
        metrics_file = exp_dir / "unsafe_metrics_models.csv"
        
        if not metrics_file.exists():
            return None
            
        try:
            df = pd.read_csv(metrics_file, index_col=0)
            return df
        except Exception:
            pass
            
        return None
        
    
    def plot_asr_dynamics(self, exp_name: str, save_path: Optional[str] = None) -> bool:
        """
        Строит график изменения ASR по шагам.
        
        Args:
            exp_name: Имя эксперимента
            save_path: Путь для сохранения графика (опционально)
            
        Returns:
            True если график построен успешно
        """
        if exp_name not in self.runs_experiments:
            print(f"Эксперимент {exp_name} не найден в runs/")
            return False
            
        exp_dir = self.runs_experiments[exp_name]['path']
        metrics_file = exp_dir / "unsafe_metrics_models.csv"
        
        if not metrics_file.exists():
            print(f"Файл метрик не найден: {metrics_file}")
            return False
            
        try:
            df = pd.read_csv(metrics_file, index_col=0)
            
            plt.figure(figsize=(12, 6))
            for col in df.columns:
                plt.plot(df.index, df[col], marker='o', label=col)
            
            plt.title(f"ASR Dynamics for {exp_name}")
            plt.xlabel("Iteration")
            plt.ylabel("ASR")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                print(f"График сохранен: {save_path}")
            else:
                plt.show()
                
            return True
            
        except Exception as e:
            print(f"Ошибка при построении графика: {e}")
            return False
    
    def get_runs_summary(self) -> pd.DataFrame:
        """
        Возвращает сводную таблицу экспериментов из runs/.
        
        Returns:
            DataFrame с колонками: experiment, steps, has_safety_details, best_step, best_metric
        """
        data = []
        for exp_name, info in self.runs_experiments.items():
            data.append({
                'experiment': exp_name,
                'steps': info['steps'],
                'has_safety_details': info['has_safety_details'],
                'best_step': info['best_step'],
                'best_metric': info['best_metric']
            })
        
        return pd.DataFrame(data)
    
    def get_runs_with_best_steps(self) -> pd.DataFrame:
        """
        Возвращает таблицу экспериментов с посчитанными лучшими шагами.
        
        Returns:
            DataFrame только с экспериментами, где есть best_step
        """
        df = self.get_runs_summary()
        return df[df['best_step'].notna()]
    
    def get_tests_summary(self) -> pd.DataFrame:
        """
        Возвращает сводную таблицу результатов тестирования.
        
        Returns:
            DataFrame с колонками: experiment, step, phi35_asr, qwenVL_asr, Llama32_asr, llava-hf_asr
        """
        data = []
        model_names = ['phi35', 'qwenVL', 'Llama32', 'llava-hf']
        
        for exp_name, steps_info in self.tests_experiments.items():
            for step, step_info in steps_info.items():
                row = {
                    'experiment': exp_name,
                    'step': step
                }
                
                # Добавляем ASR для каждой модели
                for model in model_names:
                    if model in step_info['models']:
                        row[f'{model}_asr'] = step_info['models'][model]['asr']
                        row[f'{model}_has_inference'] = step_info['models'][model]['has_inference']
                    else:
                        row[f'{model}_asr'] = None
                        row[f'{model}_has_inference'] = False
                
                data.append(row)
        
        return pd.DataFrame(data)
    
    def get_experiment_status(self, exp_name: str) -> Dict:
        """
        Получает статус эксперимента (что готово, что нет).
        
        Args:
            exp_name: Имя эксперимента
            
        Returns:
            Словарь со статусом различных этапов
        """
        status = {
            'training_completed': False,
            'safety_analysis_completed': False,
            'best_step_found': False,
            'safebench_testing': {},
            'guard_evaluation': {}
        }
        
        # Проверяем обучение
        if exp_name in self.runs_experiments:
            info = self.runs_experiments[exp_name]
            status['training_completed'] = info['steps'] > 0
            status['safety_analysis_completed'] = info['has_safety_details']
            status['best_step_found'] = info['best_step'] is not None
        
        # Проверяем тестирование
        if exp_name in self.tests_experiments:
            for step, step_info in self.tests_experiments[exp_name].items():
                status['safebench_testing'][step] = {}
                status['guard_evaluation'][step] = {}
                
                for model, model_info in step_info['models'].items():
                    status['safebench_testing'][step][model] = model_info['has_inference']
                    status['guard_evaluation'][step][model] = model_info['asr'] is not None
        
        return status
    
    def list_experiments(self) -> List[str]:
        """Возвращает список всех доступных экспериментов."""
        all_experiments = set(self.runs_experiments.keys()) | set(self.tests_experiments.keys())
        return sorted(list(all_experiments))
    
    def search_experiments(self, pattern: str) -> List[str]:
        """
        Ищет эксперименты по паттерну в названии.
        
        Args:
            pattern: Паттерн для поиска (регулярное выражение)
            
        Returns:
            Список найденных экспериментов
        """
        import re
        regex = re.compile(pattern, re.IGNORECASE)
        all_experiments = self.list_experiments()
        return [exp for exp in all_experiments if regex.search(exp)]
    
    def load_test_generations(self, exp_name: str, step: int) -> Optional[pd.DataFrame]:
        """
        Загружает результаты генерации тестов для конкретного эксперимента и шага.
        
        Args:
            exp_name: Имя эксперимента
            step: Номер шага/итерации
            
        Returns:
            DataFrame с колонками: question, model1, model2, ... или None если файл не найден
        """
        if exp_name not in self.runs_experiments:
            print(f"Эксперимент {exp_name} не найден в runs/")
            return None
            
        exp_dir = self.runs_experiments[exp_name]['path']
        test_file = exp_dir / f"test_results_iter_{step}.csv"
        
        if not test_file.exists():
            print(f"Файл тестовых результатов не найден: {test_file}")
            return None
            
        try:
            df = pd.read_csv(test_file)
            print(f"Загружен файл: {test_file}")
            print(f"Размер: {df.shape[0]} вопросов, {df.shape[1]-1} моделей")
            return df
            
        except Exception as e:
            print(f"Ошибка при загрузке файла {test_file}: {e}")
            return None
    
    def get_available_test_steps(self, exp_name: str) -> List[int]:
        """
        Возвращает список доступных шагов с тестовыми результатами для эксперимента.
        
        Args:
            exp_name: Имя эксперимента
            
        Returns:
            Список номеров шагов
        """
        if exp_name not in self.runs_experiments:
            return []
            
        exp_dir = self.runs_experiments[exp_name]['path']
        
        # Ищем файлы вида test_results_iter_*.csv
        pattern = exp_dir / "test_results_iter_*.csv"
        files = glob.glob(str(pattern))
        
        steps = []
        for file in files:
            match = re.search(r'test_results_iter_(\d+)\.csv', file)
            if match:
                steps.append(int(match.group(1)))
        
        return sorted(steps)
    
    def load_safety_evaluation(self, exp_name: str, step: int, evaluator_model: str = "gemma-3-4b-it", verbose: bool = True) -> Optional[pd.DataFrame]:
        """
        Загружает результаты оценки безопасности для конкретного эксперимента и шага.
        
        Args:
            exp_name: Имя эксперимента
            step: Номер шага/итерации
            evaluator_model: Модель-судья для оценки безопасности (по умолчанию "gemma-3-4b-it")
            
        Returns:
            DataFrame с колонками: question, model, text_answer, is_safe, safety_status или None если файл не найден
        """
        if exp_name not in self.runs_experiments:
            print(f"Эксперимент {exp_name} не найден в runs/")
            return None
            
        exp_dir = self.runs_experiments[exp_name]['path']
        safety_file = exp_dir / "safety_details" / evaluator_model / f"safety_details_iter_{step}.csv"
        
        if not safety_file.exists():
            if verbose:
                print(f"Файл оценки безопасности не найден: {safety_file}")
            return None
            
        try:
            # Используем разделитель ";" и обрабатываем булевы значения
            df = pd.read_csv(safety_file, sep=';')
            
            # Конвертируем is_safe в булев тип, если это строки
            if df['is_safe'].dtype == 'object':
                df['is_safe'] = df['is_safe'].map({'True': True, 'False': False})
            
            if verbose:
                print(f"Загружен файл: {safety_file}")
                print(f"Размер: {df.shape[0]} оценок")
                print(f"Unsafe responses: {(~df['is_safe']).sum()} ({(~df['is_safe']).mean():.2%})")
            
            return df
            
        except Exception as e:
            print(f"Ошибка при загрузке файла {safety_file}: {e}")
            return None
    
    def get_available_safety_steps(self, exp_name: str, evaluator_model: str = "gemma-3-4b-it") -> List[int]:
        """
        Возвращает список доступных шагов с оценкой безопасности для эксперимента.
        
        Args:
            exp_name: Имя эксперимента
            evaluator_model: Модель-судья для оценки безопасности
            
        Returns:
            Список номеров шагов
        """
        if exp_name not in self.runs_experiments:
            return []
            
        exp_dir = self.runs_experiments[exp_name]['path']
        safety_dir = exp_dir / "safety_details" / evaluator_model
        
        if not safety_dir.exists():
            return []
        
        # Ищем файлы вида safety_details_iter_*.csv
        pattern = safety_dir / "safety_details_iter_*.csv"
        files = glob.glob(str(pattern))
        
        steps = []
        for file in files:
            match = re.search(r'safety_details_iter_(\d+)\.csv', file)
            if match:
                steps.append(int(match.group(1)))
        
        return sorted(steps)
    
    def get_safety_summary(self, exp_name: str, evaluator_model: str = "gemma-3-4b-it") -> pd.DataFrame:
        """
        Возвращает сводную таблицу по всем доступным оценкам безопасности для эксперимента.
        
        Args:
            exp_name: Имя эксперимента
            evaluator_model: Модель-судья для оценки безопасности
            
        Returns:
            DataFrame с колонками: step, total_evaluations, unsafe_count, asr, models
        """
        available_steps = self.get_available_safety_steps(exp_name, evaluator_model)
        
        if not available_steps:
            return pd.DataFrame()
        
        summary_data = []
        
        for step in available_steps:
            safety_df = self.load_safety_evaluation(exp_name, step, evaluator_model, verbose=False)
            
            if safety_df is not None:
                total_evaluations = len(safety_df)
                unsafe_count = (~safety_df['is_safe']).sum()
                asr = unsafe_count / total_evaluations if total_evaluations > 0 else 0.0
                models = safety_df['model'].unique().tolist()
                
                summary_data.append({
                    'step': step,
                    'total_evaluations': total_evaluations,
                    'unsafe_count': unsafe_count,
                    'asr': asr,
                    'models': models
                })
        
        return pd.DataFrame(summary_data) 