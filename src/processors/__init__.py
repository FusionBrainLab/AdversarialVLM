import importlib
from typing import Tuple

# Карта соответствия весов и модулей
MODEL_MAP = {
    "microsoft/Phi-3.5-vision-instruct": {
        "module": "processors.phi3processor",
        "input_class": "AdvPhiInputs",
        "processor_class": "DifferentiablePhi3VImageProcessor",
    },
    "Qwen/Qwen2-VL-2B-Instruct": {
        "module": "processors.qwen2VLprocessor",
        "input_class": "AdvQwen2VLInputs",
        "processor_class": "DifferentiableQwen2VLImageProcessor",
    },
    "Qwen/Qwen2-VL-7B-Instruct": {
        "module": "processors.qwen2VLprocessor",
        "input_class": "AdvQwen2VLInputs",
        "processor_class": "DifferentiableQwen2VLImageProcessor",
    },
    "alpindale/Llama-3.2-11B-Vision-Instruct": {
        "module": "processors.llama32processor",
        "input_class": "AdvMllamaInputs",
        "processor_class": "DifferentiableMllamaImageProcessor",
    },
    "alpindale/Llama-3.2-11B-Vision": {
        "module": "processors.llama32processor",
        "input_class": "AdvMllamaInputs",
        "processor_class": "DifferentiableMllamaImageProcessor",
    },
    "SinclairSchneider/Llama-Guard-3-11B-Vision": {
        "module": "processors.llama32processor",
        "input_class": "AdvMllamaInputs",
        "processor_class": "DifferentiableMllamaImageProcessor",
    },
    "llava-hf/llava-1.5-7b-hf": {
        "module": "processors.llavaprocessor",
        "input_class": "AdvLlavaInputs",
        "processor_class": "DifferentiableLlavaImageProcessor",
    },
    # pqlet: google/gemma-3-12b-it. Only for evaluation.
    "google/gemma-3-12b-it": {
        "module": "processors.gemma3processor",
        "input_class": "AdvGemma3Inputs",
        "processor_class": None,
    },
    "Lexius/Phi-3.5-vision-instruct": {
        "module": "processors.phi3processor",
        "input_class": "AdvPhiInputs",
        "processor_class": "DifferentiablePhi3VImageProcessor",
    },
}

def load_components(model_name: str, any_support: bool = False) -> Tuple[object, object, object]:
    """
    Загружает load_model_and_processor, AdvInputs и DifferentiableImageProcessor 
    на основе названия весов модели.

    :param model_name: Название весов модели.
    :return: Tuple из функций/классов: (load_model_and_processor, AdvInputs, DifferentiableImageProcessor).
    :raises ValueError: Если модель не найдена в карте.
    """
    # if model_name not in MODEL_MAP:
    #     raise ValueError(f"Model {model_name} not found in MODEL_MAP. Please add it to the map.")

    if model_name in MODEL_MAP and not any_support:
        model_info = MODEL_MAP[model_name]
        module_name = model_info["module"]
    
        # Импортируем модуль через importlib
        module = importlib.import_module(module_name)

        # Получаем необходимые компоненты
        load_model_and_processor = getattr(module, "load_model_and_processor")
        AdvInputs = getattr(module, model_info["input_class"])
        if model_info["processor_class"] is not None:
            DifferentiableImageProcessor = getattr(module, model_info["processor_class"])
        else:
            # i.e. inference only 
            DifferentiableImageProcessor = None
    else:
        load_model_and_processor = load_model_and_processor_any
        AdvInputs = AdvAnyInputs
        DifferentiableImageProcessor = DifferentiableAnyImageProcessor
        

    return load_model_and_processor, AdvInputs, DifferentiableImageProcessor
