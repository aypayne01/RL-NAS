# RL-NAS
Reinforcement Learning (RL)-based Neural Architecture Search (NAS)

## Introduction
This project implements a Reinforcement Learning (RL)-based Neural Architecture Search (NAS) algorithm. The goal is to automatically search for the optimal neural network architecture for a given task, such as image classification, using an RL agent. The agent learns to propose architectures, and the performance of these architectures is used to guide the search.

## Features
- **Reinforcement Learning Agent**: Uses policy gradient (REINFORCE) to control the search for neural architectures.
- **Policy Network**: An LSTM-based controller that generates sequences of architectural decisions.
- **Parallel NAS Support**: Can leverage parallel environments for faster architecture evaluation.
- **Automatic Dataset Handling**: The project dynamically downloads and processes the CIFAR-10 dataset.
- **Modular Design**: Easily extendable to other tasks or search spaces.

## Setup and Installation

### Prerequisites
Make sure you have Python 3.12 (or lower) installed on your machine.

### Installation
1. **Clone the repository**:
    ```bash
    git clone https://github.com/aypayne01/RL-NAS.git
    cd RL-NAS
    ```
2. **Set up a virtual environment** (recommended):
    ```bash
    pyenv virtualenv 3.12.0 rl-nas-env
    pyenv activate rl-nas-env
    ```
3. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

### CIFAR-10 Dataset
The CIFAR-10 dataset will be automatically downloaded when the code is run. There's no need to manually manage or download the dataset.

## Usage

1. **Train the RL-based NAS algorithm**:
    To start the NAS process, simply run the following command:
    ```bash
    python src/main.py
    ```
    The RL agent will start generating and evaluating architectures on the CIFAR-10 dataset.

2. **Monitor Progress**:
    The output will show the architectures generated and their respective rewards (validation performance).

## Project Structure
RL-NAS/ ├── distributed.py # Optional: Parallel NAS implementation ├── env.py # Task environment for evaluating architectures ├── main.py # Main script to run the NAS process ├── policy.py # Policy network for generating architectures ├── README.md # Project documentation ├── requirements.txt # List of dependencies └── utils.py # Utility functions (e.g., for architecture representation)


## How It Works
1. **Architecture Search**:
   - The RL agent generates architectural decisions (e.g., types of layers, layer sizes) sequentially.
   - Each decision corresponds to an action sampled from a policy network.
   - The architecture is constructed based on these decisions and trained on the CIFAR-10 dataset.

2. **Reward Mechanism**:
   - After training each architecture, its performance (e.g., validation accuracy) is used as the reward for the RL agent.
   - The RL agent uses this reward to update its policy, favoring architectures that perform better.

3. **Policy Optimization**:
   - The policy network (controller) is trained using the REINFORCE algorithm, a policy gradient method that optimizes the likelihood of actions that lead to high rewards.

## Future Improvements
- **Search Space Expansion**: Add more architectural decisions, such as different types of regularization layers, activation functions, and more.
- **Support for Additional Datasets**: Extend the environment to support tasks beyond CIFAR-10, such as NLP tasks (e.g., text classification).
- **Advanced RL Techniques**: Incorporate techniques like Proximal Policy Optimization (PPO) or Actor-Critic for more efficient exploration of the search space.
