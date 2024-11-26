# 🎲 Morabaraba AI Agents 🤖

## 🌍 Overview

Welcome to an exciting exploration of **Morabaraba**, a traditional African board game reimagined through cutting-edge Artificial Intelligence! 🚀

Developed as an Honours Research Project at Wits University, this project brings together **Computer Science**, **Applied Mathematics**, and **Cultural Heritage** in a unique technological journey.

## 🎯 About Morabaraba

**Morabaraba** (aka Umlabalaba or Twelve Men's Morris) is more than just a game - it's a strategic masterpiece 🏆 deeply rooted in African cultural traditions. Played on a grid of 24 points, players engage in a thrilling dance of strategy, placing, moving, and capturing pieces to form "mills".

## 🧠 Research Approaches

Our AI investigates game intelligence through:

- 🌐 **Monte Carlo Tree Search (MCTS)**
- 📊 **Minimax Tree Search with Alpha-Beta Pruning**
- 🤖 **Neural Networks with Attention Mechanisms**

## ✨ Key Features

### 🎮 Game Environment
- 🏁 Full Morabaraba game rules implementation
- 🔄 Supports all game phases:
    - 🥇 Piece Placement
    - 🚶 Piece Movement
    - 🦘 Jumping Phase

### 🤹 AI Agents
1. 🎲 **Random Agent**: Baseline control
2. 🌳 **MCTS Agent**: Exploration meets exploitation
3. 🧩 **Minimax Agent**: Strategic tree search
4. 🧠 **Neural Network Agent**: Machine learning magic

### 🔬 Advanced Capabilities
- 🤖 Automated game simulation
- 📊 Data collection and preprocessing
- 🎯 Strategic feature engineering
- 📈 Performance visualization

## 📂 Project Structure

```
Research/ 🗂️
│
├── src/ 💻
│   ├── Game.py               # Core game logic
│   ├── Algorithms/ 🧮
│   │   ├── Random_Agent.py
│   │   ├── Monte_Carlo_Agent.py
│   │   ├── Minimax.py
│   │   └── Augmented_Minimax.py
│   └── utils/ 🛠️
│       ├── Preprocessing.py
│       └── Check_Performance.py
│
├── Models/ 🤖         # Saved neural network models
├── Data/ 📊           # Simulation game states
├── game_performance_data/ 📈  # AI agent performance
└── requirements.txt
```

## 📊 Performance Results

Our AI agents' battle results! 💥

| Agent 🤖          | Wins vs Random 🎲 | Wins vs MCTS 🌐 | Wins vs Minimax 🧩 |
|------------------|------------------|----------------|-------------------|
| Neural Network 🧠 | 30 🏆            | 18 🥈           | 15 🥉              |
| MCTS 🌳           | 30 🏆            | -              | 16 🥈              |
| Minimax 📊        | 30 🏆            | 12 🥉           | -                 |

## 🚀 Getting Started

### 📋 Prerequisites
- 🐍 Python 3.8+
- 📦 pip package manager

### 🔧 Installation
1. Clone the repository
   ```bash
   git clone https://github.com/Mahloromela/Research.git
   cd Research
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 🎮 Running the Project

#### 🎲 Simulate Games
```bash
python Simulate.py
```

#### 🧠 Train Neural Network
```bash
python Training/Collect_Data.py
```

#### 👀 Visualize Gameplay
```bash
# Check AI game performance
python Check_Performance.py

# Play against AI or another human
python Human_vs_AI.py
```

## 🗺️ Roadmap and Future Work

- 🔄 Implement reinforcement learning for self-play
- 🧩 Enhance training data with edge-case scenarios
- 🚀 Optimize simulation performance
- 🤖 Develop sophisticated neural network architectures

## 🤝 Contributing

Contributions are welcome! 🌟 Check out our [issues page](https://github.com/Mahloromela/Research/issues).

## License
This project is licensed under the [MIT License](LICENSE.txt).
## 🙏 Acknowledgements
Special thanks to:
- 👨‍🏫 **Professor Clint Van Alten**: Project supervision
- 🏫 **Wits University**: Research support

## 📜 License

MIT Licensed 🆓 - See [LICENSE.txt](LICENSE.txt) for details.

## 📞 Contact

Got questions? 🤔 Reach out to the project maintainers!

**Made with 🧠 at Wits University**