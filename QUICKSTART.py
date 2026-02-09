"""
QUICK START GUIDE
Texas Hold'em Q-Learning Lab
=============================

# ============================================================================
# INSTALLATION
# ============================================================================

pip install -r requirements.txt

# Note: tkinter usually comes with Python. If missing:
#   Ubuntu/Debian:  sudo apt-get install python3-tk
#   macOS/Windows:  Already included


# ============================================================================
# RUNNING THE APPLICATION
# ============================================================================

# Option 1: GUI Application (Recommended — CustomTkinter dark-mode UI)
python main.py

# Option 2: Command-line Demo & Validation Suite
python demo.py


# ============================================================================
# USING THE GUI
# ============================================================================

# TRAINING THE AGENT:
#   1. Set episodes (5000 recommended), α, γ, ε
#   2. Click "🚀 Start Training"
#   3. Watch progress bar & stats update live
#   4. Inspect Win-Rate graph, Reward graph, and Q-Table Heatmap tabs

# PLAYING MANUALLY (Human vs AI):
#   1. Click "▶ New Game (Manual)"
#   2. Use action buttons (Fold, Call, Raise $100, All-In $150)
#   3. Watch the AI Thought Process panel for Q-values
#   4. See results after showdown

# WATCHING AI:
#   1. Train the agent first (at least 1000 episodes)
#   2. Click "🤖 Watch AI Play"
#   3. Observe step-by-step AI decisions with 1.2 s delays


# ============================================================================
# PROGRAMMATIC USAGE
# ============================================================================
"""

from engine import PokerEnv, Card, HandEvaluator
from agent import QLearningAgent

# Create environment
env = PokerEnv()

# Create and train agent
agent = QLearningAgent(
    actions=env.actions,
    learning_rate=0.1,
    discount_factor=0.95,
    epsilon=0.2,
)

# Train for 5000 episodes
agent.train(env, num_episodes=5000, verbose_every=1000)

# Save trained agent
agent.save("my_poker_agent.pkl")

# Play one game
state = env.reset()
done = False
total_reward = 0.0

while not done:
    valid_actions = env.get_valid_actions()
    action = agent.get_action(state, valid_actions, training=False)
    print(f"  {state.street}  →  {action}")
    result = env.step(action)
    state, reward, done, info = result
    total_reward += reward

print(f"\nGame result : {info['winner']}")
print(f"Total reward: ${total_reward:+.2f}")


"""
# ============================================================================
# TESTING HAND EVALUATOR
# ============================================================================
"""

cards = [
    Card("A", "h"), Card("K", "h"), Card("Q", "h"),
    Card("J", "h"), Card("T", "h"), Card("2", "c"), Card("3", "d"),
]
hr = HandEvaluator.evaluate_hand(cards)
print(f"Hand: {hr.name}  (rank {hr.rank})")  # Royal Flush, rank 10

# Compare two hands
hero_pool = [
    Card("8", "h"), Card("9", "h"),
    Card("J", "h"), Card("Q", "h"), Card("2", "c"),
    Card("T", "h"), Card("7", "c"),
]
opp_pool = [
    Card("A", "s"), Card("K", "s"),
    Card("J", "h"), Card("Q", "h"), Card("2", "c"),
    Card("T", "h"), Card("7", "c"),
]
result = HandEvaluator.compare_hands(hero_pool, opp_pool)
print("Hero wins!" if result > 0 else "Opponent wins!" if result < 0 else "Tie!")


"""
# ============================================================================
# EXPECTED RESULTS AFTER TRAINING
# ============================================================================

# After 5000 episodes:
#   Win Rate:  45–55 %
#   Avg Reward: $10–25
#   Q-table states: 10–20
#
# The agent learns to:
#   ✓ Call / Raise on the flop (high equity from draws)
#   ✓ Value-bet when flush or straight completes
#   ✓ Fold on the river when draws miss
#   ✓ Manage pot odds across streets

# ============================================================================
# NEXT STEPS
# ============================================================================
#   1. Train with different hyper-parameters
#   2. Test against various OpponentPolicy aggression levels
#   3. Extend to pre-flop betting
#   4. Implement Deep Q-Network (DQN) with neural-net function approx
#   5. Add CFR (Counterfactual Regret Minimisation) for true Nash equilibrium
#   6. Multi-table tournament mode
"""

print("\nReady to play poker with Q-Learning! 🎰")
print("Run:  python main.py")
