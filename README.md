# ResNet-18 Fine-Tuning & Hyperparameter Analysis on CIFAR-10

## Project Objective
This project implements a reproducible machine-learning training pipeline using PyTorch to fine-tune a pre-trained ResNet-18 architecture on the CIFAR-10 dataset. The primary objective is to evaluate the isolated impacts of varying hyperparameters (batch size, learning rate, and training epochs) on model convergence, computational efficiency, and class-specific classification accuracy.

 **Read the full Technical Whitepaper:** https://docs.google.com/document/d/1Z0Hor9MGAPU4MSx5NY6p_HZov6v0-lrNEatYBKrtXPQ/edit?usp=sharing
## Tech Stack
* **Framework:** PyTorch, Torchvision
* **Language:** Python
* **Data Visualization:** Matplotlib, Seaborn
* **Metrics & Evaluation:** Scikit-Learn
* **Environment:** Google Colab (GPU Acceleration)

## Methodology & Architecture
* **Model:** ResNet-18 (Pre-trained on ImageNet1K_V1)
* **Dataset:** CIFAR-10 (augmented with `RandomResizedCrop` and `RandomHorizontalFlip` for training stability).
* **Optimization:** Stochastic Gradient Descent (SGD) with Momentum (0.9).
* **Scheduling:** StepLR (Decay by a factor of 0.1 every 7 epochs).
* **Evaluation Strategy:** Conducted four distinct training configurations to isolate and analyze the effects of learning rate scaling, batch size increases, and epoch extensions. 

## Key Results & Error Analysis
The baseline model achieved a **95.8% validation accuracy** at 20 epochs. 

While overall accuracy was exceptionally high, a rigorous class-specific error analysis revealed persistent vulnerabilities when differentiating visually similar categories (specifically `cats` vs. `dogs`). Due to shared quadrupedal features and the low 32x32 spatial resolution of the CIFAR-10 dataset, the model generalized these biological traits, leading to distinct confusion matrix clusters.

### Training & Validation Loss
`![Training Curves](LearningCurves.png)`

### Confusion Matrix
`![Confusion Matrix](ConfusionMatrix.png)`

## How to Run the Code
1. Clone this repository:
   ```bash
   git clone [https://github.com/RowanAlhallak/resnet18-cifar10-finetuning.git](https://github.com/RowanAlhallak/resnet18-cifar10-finetuning.git)
