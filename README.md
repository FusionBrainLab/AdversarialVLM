# Universal Adversarial Attack

A comprehensive research framework for creating and evaluating adversarial attacks against Multimodal Large Language Models (MLLMs).

> **📄 Research Paper**: [Universal Adversarial Attack on Aligned Multimodal LLMs](https://arxiv.org/abs/2502.07987)  
> **Authors**: Temurbek Rahmatullaev, Polina Druzhinina, Nikita Kurdiukov, Matvey Mikhalchuk, Andrey Kuznetsov, Anton Razzhigaev  
> **arXiv**: 2502.07987 [cs.AI]  
> **License**: Apache-2.0

## 🎯 Overview

**Universal Adversarial Attack** is a sophisticated framework that enables training adversarial images to trigger unsafe content generation in multimodal models and evaluating the effectiveness of such attacks. This implementation corresponds to the research paper ["Universal Adversarial Attack on Aligned Multimodal LLMs"](https://arxiv.org/abs/2502.07987), which demonstrates how a single optimized image can override alignment safeguards across diverse queries and multiple models.

The framework achieves **up to 81% attack success rates** on certain models, outperforming existing baselines including text-only universal prompts. Our method leverages backpropagation through vision encoders and language heads to craft synthetic images that force models to generate targeted phrases (e.g., "Sure, here it is") or other unsafe content.

**Key Research Contributions:**
- **Universal Image Attacks**: Single adversarial image effective across multiple models and queries
- **Cross-Model Transferability**: Attacks trained on one model transfer to others
- **Benchmark Evaluation**: Comprehensive testing on SafeBench and MM-SafetyBench
- **Multi-Answer Variants**: More natural-sounding malicious responses

## 🚀 Quick Start

### Setup WandB

```bash
# Add your WandB API key
echo "your_wandb_key" > wandb_key.txt
```

### Launch Attacks

```bash
# Single model attack on Llama-3.2
./scripts/attacks/attack_clamp_tanh_llama.sh

# Cross-model universal attack
./scripts/attacks/attack_cross.sh

# Localized attack with Gaussian blur
./scripts/attacks/attack_cross_gblur.sh
```

### Evaluation Pipeline

```bash
# 1. Find best iteration using safety judge
./scripts/evaluation/find_best_iter.sh

# 2. Test on SafeBench dataset
./scripts/evaluation/safebench_test.sh experiment_name iteration model_suffix cuda_num

# 3. Evaluate with safety judge
./scripts/evaluation/guard_eval.sh /path/to/results cuda_num
```

## 🏗️ Project Structure

```
MLLM-adversarial/
├── src/                              # Core source code
│   ├── attack_model.py               # Single model attack implementation
│   ├── crossattack_models.py         # Universal cross-model attacks
│   ├── questions.py                  # Question sets for attacks
│   ├── answers.py                    # Target harmful responses
│   ├── processors/                   # Model-specific processors
│   │   ├── abstract_processor.py     # Base processor class
│   │   ├── phi3processor.py          # Phi-3.5 vision processor
│   │   ├── qwen2VLprocessor.py       # Qwen2-VL processor
│   │   ├── llama32processor.py       # Llama-3.2 vision processor
│   │   ├── llavaprocessor.py         # LLaVA processor
│   │   └── gemma3processor.py        # Gemma-3 judge processor
│   ├── judge/                        # Safety evaluation module
│   │   ├── safety_checker.py         # Gemma-3 safety judge
│   │   ├── system_prompts.txt        # Judge system prompts
│   │   └── README.md                 # Judge documentation
│   └── evaluation/                   # Evaluation and testing tools
│       ├── experiment_tracker.py     # Experiment analysis
│       ├── find_best_iter_gemma.py   # Best iteration finder
│       ├── SafeBench_universal.py    # SafeBench testing
│       ├── MM_SafetyBench_baseline.py # MM-SafetyBench evaluation
│       ├── FigStep_baseline.py       # FigStep testing
│       ├── guard_eval_gemma.py       # Safety evaluation
│       └── README.md                 # Evaluation documentation
├── scripts/                          # Automation scripts
│   ├── attacks/                      # Attack launch scripts
│   │   ├── attack_cross.sh           # Cross-model attacks
│   │   ├── attack_cross_gblur.sh     # Attacks with Gaussian blur
│   │   ├── attack_cross_localization.sh # Localized attacks
│   │   └── attack_clamp_tanh_*.sh    # Single model attacks
│   └── evaluation/                   # Evaluation scripts
│       ├── find_best_iter.sh         # Best iteration finder
│       ├── safebench_test.sh         # SafeBench testing
│       ├── guard_eval.sh             # Safety evaluation
│       └── baseline_tests.sh         # Baseline benchmarks
├── datasets/                         # Evaluation datasets
│   ├── SafeBench_text_subset/        # SafeBench dataset
│   ├── MM_SafetyBench/               # MM-SafetyBench dataset
│   └── FigStep/                      # FigStep dataset
├── runs/                             # Training results
├── tests/                            # Evaluation results
└── images/                           # Source images
```

## 🔧 Core Components

### Adversarial Attack Framework

- **Single Model Attacks** (`attack_model.py`) - Targeted attacks against individual models with various clamping methods and localization techniques
- **Universal Cross-Model Attacks** (`crossattack_models.py`) - Attacks that work across multiple models simultaneously

### Processor System

The framework features a sophisticated processor system with specialized handlers for each model:

- **Phi-3.5** (`phi3processor.py`) - Dynamic tile-based processing with global + local context
- **Qwen2-VL** (`qwen2VLprocessor.py`) - Dynamic scaling with fixed ViT patches
- **Llama-3.2** (`llama32processor.py`) - Adaptive tiling with optimal canvas allocation
- **LLaVA** (`llavaprocessor.py`) - Simple fixed-size scaling
- **Gemma-3** (`gemma3processor.py`) - Safety judge for evaluation only

### Safety Evaluation System

**SafetyChecker** class uses Gemma-3 with structured output to evaluate the safety of generated content, providing:
- Binary safety classification
- Confidence scores
- Detailed reasoning
- Structured JSON output

### Evaluation Suite

- **Experiment Tracker** (`experiment_tracker.py`) - Comprehensive analysis and visualization of experimental results
- **Best Iteration Finder** (`find_best_iter_gemma.py`) - Automated selection of optimal attack iterations
- **Universal Testing** (`SafeBench_universal.py`) - Standardized evaluation on safety benchmarks
- **Baseline Comparisons** - Testing against multiple safety datasets

## 🎯 Attack Methods

### Clamping Techniques
- **Tanh Clamping** - Gradient-preserving constraint using tanh function (default)
- **Direct Clamping** - Hard pixel value constraints
- **Custom Methods** - Extensible clamping system

### Localization Strategies
- **Corner Attack** - Adversarial perturbations in n×n corner regions
- **Bottom Lines** - Attacks targeting bottom k rows of images
- **Random Square** - Movable n×n adversarial patches for robustness
- **Full Image** - Traditional full-image perturbations

### Robustness Enhancements
- **Gaussian Blur** - Blur augmentation for improved transferability
- **Local Cropping** - Random cropping for universal attacks
- **Multi-Model Training** - Cross-model optimization for universal attacks

## 📊 Evaluation Workflow

1. **Adversarial Training** - Optimize pixel perturbations to maximize target text generation probability
2. **Best Iteration Selection** - Use Gemma-3 judge to identify most effective attack iterations
3. **Benchmark Testing** - Evaluate attacks on SafeBench, MM-SafetyBench, and FigStep datasets
4. **Safety Assessment** - Quantitative evaluation using Attack Success Rate (ASR) metrics

## 🛠️ Supported Models

### Target Models (Attack Victims)
- **Microsoft Phi-3.5-vision-instruct** - Advanced vision-language model
- **Qwen/Qwen2-VL-*B-Instruct** - Alibaba's multimodal model family
- **meta-llama/Llama-3.2-11B-Vision-Instruct** - Meta's vision-enabled Llama
- **llava-hf/llava-1.5-7b-hf** - Popular open-source multimodal model

### Judge Model
- **google/gemma-3-4b-it** - Safety evaluation and content filtering

## 📈 Experiment Analysis

Use the `ExperimentTracker` for comprehensive result analysis:

```python
from src.evaluation.experiment_tracker import ExperimentTracker

# Initialize tracker
tracker = ExperimentTracker()

# List all experiments
experiments = tracker.list_experiments()

# Get experiment details
info = tracker.get_experiment_info("experiment_name")

# Visualize ASR dynamics
tracker.plot_asr_dynamics("experiment_name")

# Generate summary reports
runs_summary = tracker.get_runs_summary()
tests_summary = tracker.get_tests_summary()
```

## 🔬 Usage Examples

### Single Model Attack with Localization

```bash
python src/attack_model.py \
    --model_name "microsoft/Phi-3.5-vision-instruct" \
    --epsilon 0.3 \
    --mask_type corner \
    --mask_size 50 \
    --clamp_method tanh \
    --use_gaussian_blur
```

### Universal Cross-Model Attack

```bash
python src/crossattack_models.py \
    --models "microsoft/Phi-3.5-vision-instruct,Qwen/Qwen2-VL-7B-Instruct" \
    --epsilon 0.4 \
    --mask_type random_square \
    --mask_size 100 \
    --use_gaussian_blur \
    --clamp_method tanh
```

### Automated Evaluation Pipeline

```bash
# Find best iteration
python src/evaluation/find_best_iter_gemma.py \
    --experiment_name "cross_attack_20250121"

# Test on SafeBench
python src/evaluation/SafeBench_universal.py \
    --exp "cross_attack_20250121" \
    --iter 3500 \
    --model_suf "Universal" \
    --cuda_num 0
```

## 📋 Available Datasets

- **SafeBench** - Primary safety evaluation benchmark
- **MM-SafetyBench** - Extended multimodal safety dataset
- **FigStep** - Image understanding and reasoning dataset

## 🧪 Advanced Features

### Optimizer Restart
- Periodic optimizer reinitialization for improved convergence
- Configurable restart intervals

### Batch Processing
- Mixed prompt batching for efficient training
- Randomized question sampling within batches

### Multi-GPU Support
- Distributed training across multiple GPUs
- Model parallelism for large-scale experiments

## 📚 More Documentation

- **[src/evaluation/README.md](src/evaluation/README.md)** - Evaluation framework documentation
- **[src/judge/README.md](src/judge/README.md)** - Safety judge documentation


## ⚠️ Ethical Considerations

This framework is designed exclusively for AI safety research purposes. The goal is to identify and mitigate potential vulnerabilities in multimodal AI systems. **Usage for creating harmful content is strictly prohibited.**

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is intended for academic research purposes. Please contact the authors for usage details.

## 🙏 Acknowledgments

- Hugging Face team for providing model infrastructure
- SafeBench, MM-SafetyBench, and FigStep dataset creators
- AI safety research community for foundational work
- Open-source contributors to the multimodal AI ecosystem 