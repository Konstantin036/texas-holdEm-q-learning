# Texas Hold'em · Q-Learning Lab

> A high-fidelity Python simulation of a **simplified heads-up Texas Hold'em MDP** (post-flop) solved with **tabular Q-Learning**.

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture (MVC)](#architecture-mvc)
3. [MDP Formulation](#mdp-formulation)
4. [Hand Ranking Engine](#hand-ranking-engine)
5. [Q-Learning Agent](#q-learning-agent)
6. [GUI Features](#gui-features)
7. [Installation](#installation)
8. [Usage](#usage)
9. [Convergence to Nash Equilibrium](#convergence-to-nash-equilibrium)
10. [File Descriptions](#file-descriptions)
11. [Customisation](#customisation)
12. [Troubleshooting](#troubleshooting)

---

## Overview

This project implements a simplified **heads-up** (1 v 1) Texas Hold'em starting from the **flop** with a fixed hero hand:

| Component | Value |
|-----------|-------|
| **Hero** | 8♥ 9♥ |
| **Flop** | J♥ Q♥ 2♣ |
| **Stacks** | $150 each |
| **Pot** | $100 (pre-flop action already concluded) |

Hero holds a **double-gutter straight draw** (needs T or K) **and** a **flush draw** (9 heart outs).  Combined unique outs ≈ 15 cards, giving roughly **54 % equity** to improve by the river.

The opponent is modelled as a **fixed stochastic policy** (analogous to the dealer in Blackjack).  A **Q-Learning** agent learns, through thousands of episodes, to exploit this sub-game.

---

## Architecture (MVC)

```
┌────────────────────────────┐
│          main.py           │  ← Entry point
└─────────────┬──────────────┘
              │ imports
┌─────────────▼──────────────┐
│           ui.py            │  ← View + Controller (CustomTkinter)
│  • Card animations         │
│  • Q-Table heatmap         │
│  • Win-rate live graph     │
│  • Human-vs-AI mode        │
└─────────────┬──────────────┘
              │ imports
┌─────────────▼──────────────┐
│          agent.py          │  ← Controller (ε-greedy Q-Learning)
│  • Q-table management      │
│  • ε-greedy action select  │
│  • TD update rule          │
└─────────────┬──────────────┘
              │ imports
┌─────────────▼──────────────┐
│         engine.py          │  ← Model (MDP environment)
│  • Card, HandEvaluator     │
│  • GameState (NamedTuple)  │
│  • PokerEnv (transitions)  │
│  • OpponentPolicy          │
└─────────────┬──────────────┘
              │ imports
┌─────────────▼──────────────┐
│         config.py          │  ← Constants & theme palette
│  • Poker rules & defaults  │
│  • Q-Learning defaults     │
│  • GUI theme colours       │
└────────────────────────────┘
```

Strict **separation of concerns** — the environment knows nothing about the GUI, the agent knows nothing about rendering, and the GUI orchestrates both.

---

## MDP Formulation

### State Space

```
S = (street, hero_stack, pot)
```

Where `street ∈ {flop, turn, river, showdown}`.  Community cards are implicitly encoded because the hero hand and flop are fixed (only the turn and river are stochastic).

### Action Space

```
A = {fold, call, raise_100, raise_150}
```

`raise_150` is an all-in.

### Transition Dynamics

After both players act on a street, the next community card is dealt **uniformly at random** from the 47 remaining unknown cards (52 − 2 hero − 3 flop).

```
P(s' | s, a)  ∝  Uniform over remaining deck
```

### Reward Function

```
R = ΔStack_hero = stack_final − stack_initial
```

No intermediate shaping — the agent receives reward **only** when the hand concludes (fold, opponent fold, or showdown).

### Pot Splitting

On a tied showdown, the pot is split evenly.  **Odd chips go to the hero** (player left of the dealer), per BGC rules:

```python
hero_share = (pot + 1) // 2
```

---

## Hand Ranking Engine

Strictly follows the BGC Texas Hold'em hierarchy:

| Rank | Hand | Numeric |
|------|------|---------|
| 10 | Royal Flush | A♠K♠Q♠J♠T♠ |
| 9 | Straight Flush | 9♥8♥7♥6♥5♥ |
| 8 | Four of a Kind | K♠K♥K♦K♣5♠ |
| 7 | Full House | K♠K♥K♦5♠5♥ |
| 6 | Flush | K♥9♥7♥5♥2♥ |
| 5 | Straight | K♠Q♥J♦T♣9♠ |
| 4 | Three of a Kind | 7♠7♥7♦K♣2♠ |
| 3 | Two Pair | K♠K♥7♦7♣2♠ |
| 2 | One Pair | A♠A♥7♦5♣2♠ |
| 1 | High Card | A♠K♥9♦5♣2♠ |

### Ace Dynamics

The Ace plays **both** roles:

- **High**: A-K-Q-J-T straight (the Broadway)
- **Low**: 5-4-3-2-A straight (the Wheel)

The evaluator considers all C(7,5) = 21 five-card combinations from the 7-card pool and returns the best.

---

## Q-Learning Agent

### Update Rule

$$Q(s, a) \leftarrow Q(s, a) + \alpha \bigl[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \bigr]$$

### Exploration Strategy

**ε-greedy**: with probability ε choose a random valid action; otherwise choose the action with the highest Q-value (ties broken randomly).

### Hyperparameters

| Symbol | Parameter | Default | Range |
|--------|-----------|---------|-------|
| α | `learning_rate` | 0.10 | 0.01 – 0.5 |
| γ | `discount_factor` | 0.95 | 0.90 – 0.99 |
| ε | `epsilon` | 0.20 | 0.05 – 0.40 |

All three are **exposed** in the GUI and in the constructor for programmatic tuning.

---

## GUI Features

Built with **CustomTkinter** for a modern dark-mode aesthetic.

| Feature | Description |
|---------|-------------|
| 🃏 **Card widgets** | Visual card rendering with suit colours & smooth deal animation |
| 📊 **Q-Table Heatmap** | Real-time colour-coded matrix (states × actions) using RdYlGn colourmap |
| 📈 **Win-Rate Graph** | 50-episode rolling average updated after training |
| 📉 **Reward Graph** | Raw + moving-average reward curve |
| 🧠 **AI Thought Process** | Live Q-value display for the current game state |
| 🎮 **Human vs AI** | Play manually while seeing what the AI *would* choose |
| 🤖 **Watch AI** | Step-by-step AI play with 1.2 s delays |
| ⚙️ **Hyperparameter Tuning** | Adjust α, γ, ε from the GUI before training |
| 📊 **Progress Bar** | Real-time training progress indicator |

---

## Installation

### Prerequisites

- **Python 3.9+**
- **tkinter** (usually bundled with Python)

### Steps

```bash
# 1. Clone / navigate to the project
cd TexasHold\'em

# 2. (Optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

If `tkinter` is missing on Linux:

```bash
sudo apt-get install python3-tk
```

---

## Usage

### GUI (recommended)

```bash
python main.py
```

1. **Train** — enter episode count & hyperparameters → click *Start Training*.
2. **Inspect** — switch between the *Win Rate*, *Reward*, *Q-Table Heatmap*, and *Q-Values* tabs.
3. **Play** — click *New Game (Manual)* and use the action buttons.
4. **Watch** — click *Watch AI Play* to see the trained agent in action.

### Command-line demo

```bash
python demo.py
```

Runs the full validation suite: hand evaluator, pot splitting, outs analysis, training convergence, and demonstration games.

### Programmatic

```python
from engine import PokerEnv
from agent import QLearningAgent

env = PokerEnv()
agent = QLearningAgent(actions=env.actions, learning_rate=0.1,
                       discount_factor=0.95, epsilon=0.2)
agent.train(env, num_episodes=5000, verbose_every=1000)
agent.save("my_agent.pkl")

# Play one hand
state = env.reset()
done = False
while not done:
    action = agent.get_action(state, env.get_valid_actions(), training=False)
    result = env.step(action)
    state, reward, done, info = result

print(f"Winner: {info['winner']}  |  Reward: ${reward:+.0f}")
```

---

## Convergence to Nash Equilibrium

### Why This Sub-game Has a Clear Optimal Strategy

In this **fixed sub-game**, hero always starts with the same hand (8♥9♥) against a **stationary stochastic opponent**.  Because:

1. The opponent's policy is **fixed** (not adapting).
2. The state space is **finite** and **fully observable** to the agent.
3. Transitions are **Markovian** (next state depends only on current state + action + random card).

…the Q-Learning algorithm is **guaranteed to converge** to the optimal Q-function $Q^*(s, a)$ as $t \to \infty$, provided:

- Every (state, action) pair is visited infinitely often (ensured by ε-greedy).
- The learning rate satisfies the Robbins-Monro conditions (constant α works in practice for finite MDPs).

### What the Agent Learns

| Street | Optimal Action | Reasoning |
|--------|---------------|-----------|
| **Flop** | Call / Raise | 15 outs ≈ 54% equity.  Folding leaves money on the table. |
| **Turn** (hit) | Raise | Made hand (flush or straight).  Extract value. |
| **Turn** (miss) | Call | Still 15 outs with 1 card to come ≈ 30% equity.  Pot odds justify calling. |
| **River** (hit) | Raise | Value bet the made hand. |
| **River** (miss) | Fold | No equity remaining.  Minimise losses. |

After 5 000+ episodes, the Q-values clearly reflect this pattern — `Q(flop, call) >> Q(flop, fold)` and `Q(river_miss, fold) > Q(river_miss, call)`.

### Approximation of Nash Equilibrium

Against a fixed opponent, the converged Q-policy is the **best response** to that opponent's strategy.  In two-player zero-sum games, a pair of best responses constitutes a **Nash Equilibrium**.  Since the opponent is fixed, the agent's converged policy is the NE *for this specific sub-game*.

For a truly adaptive opponent, one would need **fictitious play**, **CFR (Counterfactual Regret Minimisation)**, or **Nash-Q** — extensions left as future work.

---

## File Descriptions

| File | Role | Key Classes |
|------|------|-------------|
| [config.py](config.py) | **Config** — Constants & palette | All poker, RL, and theme constants |
| [engine.py](engine.py) | **Model** — MDP environment | `Card`, `HandEvaluator`, `HandRank`, `GameState`, `OpponentPolicy`, `PokerEnv` |
| [agent.py](agent.py) | **Controller** — RL agent | `QLearningAgent` |
| [ui.py](ui.py) | **View** — CustomTkinter GUI | `CardWidget`, `PokerGUI` |
| [main.py](main.py) | **Entry point** | — |
| [demo.py](demo.py) | CLI validation suite | `test_hand_evaluator`, `test_training`, … |
| [QUICKSTART.py](QUICKSTART.py) | Quick-start guide | — |
| [requirements.txt](requirements.txt) | Dependencies | — |

---

## Customisation

### Change starting hand

In `engine.py`:

```python
self.hero_cards = [Card("A", "s"), Card("K", "s")]
self.flop = [Card("A", "h"), Card("K", "h"), Card("Q", "h")]
```

### Adjust opponent aggression

```python
self.opponent_policy = OpponentPolicy(aggression=0.5, fold_prob=0.15)
```

### Tune hyperparameters

```python
agent = QLearningAgent(
    actions=env.actions,
    learning_rate=0.05,    # slower, more stable
    discount_factor=0.99,  # values future rewards more
    epsilon=0.10,          # less exploration
)
```

Or adjust them directly in the GUI before clicking *Start Training*.

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: customtkinter` | `pip install customtkinter` |
| `ModuleNotFoundError: tkinter` | `sudo apt-get install python3-tk` |
| GUI blank / crash | Ensure `matplotlib` backend is `TkAgg` (set automatically) |
| Agent not learning | Train ≥ 5 000 episodes; try α=0.05, ε=0.15 |
| Slow training | 5 000 episodes ≈ 3 s; 50 000 ≈ 30 s |

---

## Expected Results

| Episodes | Win Rate | Avg Reward | Q-Table States |
|----------|----------|------------|----------------|
| 1 000 | 40–50 % | $0–15 | 5–10 |
| 5 000 | 45–55 % | $10–25 | 10–20 |
| 20 000 | 50–60 % | $15–30 | 15–25 |

---

## License

MIT — see individual file headers.

---

*Built as a reinforcement-learning research prototype.  Not intended for real-money gambling.*
