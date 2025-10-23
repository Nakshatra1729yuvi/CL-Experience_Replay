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

## 🎓 Usage

### Basic Example
```python
from cl_experience_replay import ExperienceReplayModel, DatasetSequence

# Initialize model and replay buffer
model = ExperienceReplayModel(
    model_name='resnet18',
    buffer_size=1000,
    replay_strategy='reservoir'
)

# Train on sequential tasks
for task_id, task_data in enumerate(task_sequence):
    model.train_task(task_data, task_id)
    model.evaluate()
```

### Advanced Configuration
```python
config = {
    'model': 'resnet18',
    'buffer_size': 2000,
    'replay_strategy': 'gradient_based',
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs_per_task': 10
}

model = ExperienceReplayModel(**config)
```

## 🔬 Experiments

We provide several pre-configured experiments:

### Split MNIST
```bash
python experiments/run_split_mnist.py --buffer_size 500 --strategy reservoir
```

### Permuted MNIST
```bash
python experiments/run_permuted_mnist.py --n_tasks 10 --buffer_size 1000
```

### Split CIFAR-10
```bash
python experiments/run_split_cifar10.py --buffer_size 2000 --strategy herding
```

## 📊 Evaluation Metrics

The framework tracks several key metrics:
- **Average Accuracy**: Performance across all learned tasks
- **Forgetting Measure**: Quantifies catastrophic forgetting
- **Forward Transfer**: Measures positive transfer to new tasks
- **Backward Transfer**: Measures knowledge retention

## 🏗️ Architecture

```
CL-Experience_Replay/
├── cl_experience_replay/
│   ├── models/          # Neural network architectures
│   ├── buffers/         # Experience replay buffer implementations
│   ├── strategies/      # Sample selection strategies
│   ├── datasets/        # Dataset loaders and processors
│   └── utils/           # Utility functions
├── experiments/         # Experiment scripts
├── notebooks/           # Jupyter notebooks for analysis
├── tests/              # Unit tests
└── configs/            # Configuration files
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

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

## Results 📊

**Final Validation Accuracies:**
- FashionMNIST: **87.68%**
- MNIST: **93.82%**

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
