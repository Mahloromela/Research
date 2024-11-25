---

# Morabaraba AI Agents

This repository contains the implementation of Artificial Intelligence (AI) agents for **Morabaraba**, a traditional African board game. Developed as part of an Honours Research Project in Computer Science and Applied Mathematics at Wits University, the project explores multiple AI approaches, including:

- **Monte Carlo Tree Search (MCTS)**
- **Minimax Tree Search**
- **Neural Networks**

The aim is to develop AI agents capable of playing Morabaraba with strategic proficiency, preserving the cultural significance of the game while advancing AI research in traditional board games.

---

## Table of Contents
- [Background](#background)
- [Features](#features)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Results](#results)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Background
**Morabaraba**, also known as Umlabalaba or Twelve Men's Morris, is a traditional African strategy board game. Played on a grid of 24 points, the game involves strategic placement, movement, and removal of pieces to form "mills" (three aligned pieces).

This project investigates how AI techniques can replicate or enhance human-like gameplay through:
- **Monte Carlo simulations**
- **Tree-based search algorithms**
- **Learning-based neural networks**

---

## Features
- **Custom Game Environment**: Implements Morabaraba game rules and phases (placement, movement, and jumping).
- **AI Agents**:
    - **Random Agent**: Baseline control agent.
    - **MCTS Agent**: Combines exploration and exploitation for decision-making.
    - **Minimax Agent**: Uses alpha-beta pruning for strategic depth.
    - **Neural Network Agent**: Employs attention mechanisms for nuanced decision-making.
- **Simulation and Data Collection**: Automated games generate data for training and evaluation.
- **Feature Engineering**: Extracts strategic metrics like mills, mobility, and configuration advantages for neural network training.

---

## Project Structure
```
├── src
│   ├── game_environment.py   # Core Morabaraba game logic
│   ├── agents
│   │   ├── random_agent.py   # Random move generator
│   │   ├── mcts_agent.py     # Monte Carlo Tree Search implementation
│   │   ├── minimax_agent.py  # Minimax algorithm with alpha-beta pruning
│   │   └── nn_agent.py       # Neural network agent with attention mechanisms
│   └── utils
│       ├── feature_engineering.py # Feature extraction for training
│       ├── data_preprocessing.py  # Data cleaning and normalization
│       └── visualization.py       # Game state visualization with Pygame
├── data
│   ├── generated_data/       # Game states and results from simulations
│   └── trained_models/       # Saved neural network models
├── notebooks
│   └── exploratory_analysis.ipynb # Neural network training analysis
├── results
│   ├── evaluation_metrics.csv # AI agent performance summaries
├── README.md
└── requirements.txt
```

---

## How to Run
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mahloromela/Research.git
   cd Research
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Simulate Games**:
   Run matches between agents and collect data:
   ```bash
   python src/game_environment.py
   ```

4. **Train Neural Network**:
   ```bash
   python src/agents/nn_agent.py --train
   ```

5. **Visualize Gameplay**:
   ```bash
   python src/utils/visualization.py
   ```

---

## Results
The AI agents were tested across multiple training iterations with the following outcomes:

| Agent          | Wins Against Random | Wins Against MCTS | Wins Against Minimax |
|----------------|---------------------|-------------------|-----------------------|
| Neural Network | 30                 | 18                | 15                    |
| MCTS           | 30                 | -                 | 16                    |
| Minimax        | 30                 | 12                | -                     |

Neural networks displayed strong performance after training but showed room for improvement against Monte Carlo and Minimax agents.

---

## Future Work
- **Reinforcement Learning**: Implement self-play for neural network agents to improve adaptability.
- **Enhanced Training Data**: Include edge-case scenarios for better generalization.
- **Optimization**: Parallel processing and efficient state copying for faster simulations.

---

## Acknowledgements
Special thanks to:
- **Professor Clint Van Alten** for supervision and guidance.
- **Wits University** for providing the resources to conduct this research.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

Feel free to modify this template to align with your preferences or additional project details.
