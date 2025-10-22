# 🧠 Continual Learning with Experience Replay

A comprehensive implementation of Experience Replay techniques for Continual Learning scenarios, designed to mitigate catastrophic forgetting in deep neural networks.

## 📋 Overview

This repository provides a modular framework for implementing and evaluating Experience Replay methods in continual learning settings. Experience Replay is a powerful technique that stores and replays past experiences to maintain previously learned knowledge while learning new tasks.

## ✨ Features

- 🔄 **Multiple Replay Strategies**: Implementation of various experience replay methods including:
  - Reservoir Sampling
  - Gradient-based Selection
  - Herding Selection
  - Random Selection
- 📊 **Comprehensive Evaluation**: Built-in metrics for measuring catastrophic forgetting and knowledge retention
- 🎯 **Task-Incremental Learning**: Support for both class-incremental and task-incremental scenarios
- 🔧 **Modular Architecture**: Easy to extend and customize for different models and datasets
- 📈 **Visualization Tools**: Track and visualize learning progress across tasks
- ⚡ **GPU Acceleration**: Optimized for efficient training with CUDA support

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch 1.10 or higher
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/Nakshatra1729yuvi/CL-Experience_Replay.git
cd CL-Experience_Replay
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📦 Requirements

```
torch>=1.10.0
torchvision>=0.11.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
tqdm>=4.62.0
tensorboard>=2.7.0
```

## 💻 Usage

### Basic Example

```python
from models import ExperienceReplayModel
from datasets import get_continual_dataset
from replay import ReservoirSampling

# Initialize model and replay buffer
model = ExperienceReplayModel(input_size=784, hidden_size=256, num_classes=10)
replay_buffer = ReservoirSampling(memory_size=500)

# Load continual learning dataset
train_tasks, test_tasks = get_continual_dataset('MNIST', num_tasks=5)

# Train on sequential tasks
for task_id, (train_data, test_data) in enumerate(zip(train_tasks, test_tasks)):
    print(f"Training on Task {task_id + 1}")
    model.train_task(train_data, replay_buffer)
    
    # Evaluate on all previous tasks
    avg_accuracy = model.evaluate_all_tasks(test_tasks[:task_id+1])
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
```

### Training with Custom Configuration

```python
python train.py --dataset CIFAR100 \
                --num_tasks 10 \
                --memory_size 2000 \
                --replay_strategy reservoir \
                --epochs 50 \
                --batch_size 32 \
                --learning_rate 0.001
```

### Evaluation

```python
python evaluate.py --checkpoint checkpoints/model_best.pth \
                   --dataset CIFAR100 \
                   --num_tasks 10
```

## 📊 Supported Datasets

- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- TinyImageNet
- Custom datasets (with provided data loaders)

## 🔬 Experimental Results

The repository includes scripts to reproduce key results from continual learning literature:

- **Catastrophic Forgetting Analysis**: Measure forgetting across tasks
- **Memory Efficiency**: Compare replay strategies with different buffer sizes
- **Scalability**: Evaluate performance with varying number of tasks

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork the repository** and create your branch from `main`
2. **Write clear commit messages** describing your changes
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with a comprehensive description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 .
black .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This work builds upon research in continual learning and experience replay:

- **Experience Replay**: Inspired by the seminal work in reinforcement learning and its adaptation to supervised continual learning
- **Continual Learning Benchmarks**: Thanks to the continual learning community for establishing evaluation protocols
- **PyTorch Community**: For providing an excellent deep learning framework

Special thanks to researchers whose work has influenced this implementation:
- Rolnick et al. (2019) - "Experience Replay for Continual Learning"
- Chaudhry et al. (2019) - "Efficient Lifelong Learning with A-GEM"
- Riemer et al. (2019) - "Learning to Learn without Forgetting"

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{cl_experience_replay,
  author = {Nakshatra1729yuvi},
  title = {Continual Learning with Experience Replay},
  year = {2025},
  url = {https://github.com/Nakshatra1729yuvi/CL-Experience_Replay}
}
```

## 📧 Contact

For questions, issues, or suggestions, please:
- Open an issue on GitHub
- Start a discussion in the Discussions tab

## 🌟 Star History

If you find this project helpful, please consider giving it a star! ⭐

---

**Happy Continual Learning!** 🚀🧠
