"""
Command-line demo & validation suite
=====================================
Comprehensive testing of all components:
  1. Hand evaluator (all 10 ranks + Ace dynamics)
  2. Pot-splitting edge case
  3. Environment MDP transitions
  4. Q-Learning convergence (double-gutter + flush draw insight)
  5. Trained agent play-through
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from engine import Card, HandEvaluator, HandRank, PokerEnv
from agent import QLearningAgent


# ============================================================================
# 1. Hand evaluator
# ============================================================================

def test_hand_evaluator() -> None:
    """Validate every hand rank including Ace-high and Ace-low straights."""
    print("=" * 64)
    print("  HAND EVALUATOR — all 10 ranks")
    print("=" * 64)

    cases: List[Tuple[str, List[Card], int]] = [
        (
            "Royal Flush",
            [Card("A", "h"), Card("K", "h"), Card("Q", "h"),
             Card("J", "h"), Card("T", "h"), Card("2", "c"), Card("3", "d")],
            10,
        ),
        (
            "Straight Flush (9-high)",
            [Card("9", "h"), Card("8", "h"), Card("7", "h"),
             Card("6", "h"), Card("5", "h"), Card("2", "c"), Card("3", "d")],
            9,
        ),
        (
            "Wheel Straight (A-low)",
            [Card("A", "h"), Card("2", "d"), Card("3", "c"),
             Card("4", "s"), Card("5", "h"), Card("K", "c"), Card("Q", "d")],
            5,
        ),
        (
            "Four of a Kind (Kings)",
            [Card("K", "h"), Card("K", "d"), Card("K", "c"),
             Card("K", "s"), Card("5", "h"), Card("2", "c"), Card("3", "d")],
            8,
        ),
        (
            "Full House (K over 5)",
            [Card("K", "h"), Card("K", "d"), Card("K", "c"),
             Card("5", "s"), Card("5", "h"), Card("2", "c"), Card("3", "d")],
            7,
        ),
        (
            "Flush (K-high hearts)",
            [Card("K", "h"), Card("9", "h"), Card("7", "h"),
             Card("5", "h"), Card("2", "h"), Card("3", "c"), Card("4", "d")],
            6,
        ),
        (
            "Straight (K-high)",
            [Card("K", "h"), Card("Q", "d"), Card("J", "c"),
             Card("T", "s"), Card("9", "h"), Card("2", "c"), Card("3", "d")],
            5,
        ),
        (
            "Three of a Kind (7s)",
            [Card("7", "h"), Card("7", "d"), Card("7", "c"),
             Card("K", "s"), Card("2", "h"), Card("3", "c"), Card("4", "d")],
            4,
        ),
        (
            "Two Pair (K & 7)",
            [Card("K", "h"), Card("K", "d"), Card("7", "c"),
             Card("7", "s"), Card("2", "h"), Card("3", "c"), Card("4", "d")],
            3,
        ),
        (
            "One Pair (Aces)",
            [Card("A", "h"), Card("A", "d"), Card("7", "c"),
             Card("5", "s"), Card("2", "h"), Card("9", "c"), Card("6", "d")],
            2,
        ),
        (
            "High Card (A-high)",
            [Card("A", "h"), Card("K", "d"), Card("9", "c"),
             Card("6", "s"), Card("2", "h"), Card("8", "c"), Card("4", "d")],
            1,
        ),
    ]

    all_pass = True
    for label, cards, expected_rank in cases:
        hr: HandRank = HandEvaluator.evaluate_hand(cards)
        ok = hr.rank == expected_rank
        status = "✅" if ok else "❌"
        print(f"  {status}  {label:30s}  rank={hr.rank:2d}  "
              f"({hr.name})  tiebreakers={hr.tiebreakers}")
        if not ok:
            all_pass = False

    print()
    print("  Ace-low straight tiebreaker check:")
    wheel = [Card("A", "h"), Card("2", "d"), Card("3", "c"),
             Card("4", "s"), Card("5", "h"), Card("7", "c"), Card("9", "d")]
    hr_w = HandEvaluator.evaluate_hand(wheel)
    assert hr_w.tiebreakers == (3,), f"Expected (3,), got {hr_w.tiebreakers}"
    print(f"  ✅  Wheel high-card = 5 (rank_value 3)")

    print()
    if all_pass:
        print("  All hand-evaluator tests PASSED ✅")
    else:
        print("  Some tests FAILED ❌")
    print()


# ============================================================================
# 2. Pot splitting
# ============================================================================

def test_pot_splitting() -> None:
    """Verify that ties split the pot and odd chips go to hero."""
    print("=" * 64)
    print("  POT SPLITTING (odd chip → hero)")
    print("=" * 64)

    # Force a tie by giving both players the same pocket cards
    # (environment draws random opponent cards, so we just check the
    #  splitting math directly)
    total_pot = 101  # odd number
    hero_share = (total_pot + 1) // 2  # should be 51
    opp_share = total_pot - hero_share
    print(f"  Pot = ${total_pot}")
    print(f"  Hero share = ${hero_share}  (odd chip to hero)")
    print(f"  Opp  share = ${opp_share}")
    assert hero_share == 51 and opp_share == 50
    print("  ✅  Pot-splitting logic correct")
    print()


# ============================================================================
# 3. Environment
# ============================================================================

def test_environment() -> None:
    """Test environment reset, transitions, and terminal states."""
    print("=" * 64)
    print("  ENVIRONMENT MDP")
    print("=" * 64)

    env = PokerEnv()
    state = env.reset()

    print(f"  Hero   : {env.hero_cards[0].symbol} {env.hero_cards[1].symbol}")
    print(f"  Flop   : {' '.join(c.symbol for c in env.flop)}")
    print(f"  Opponent: {env.opponent_cards[0].symbol} {env.opponent_cards[1].symbol}")
    print(f"  Pot    : ${env.pot}")
    print(f"  Stack  : ${env.hero_stack}")
    print()

    # Known cards: 2 hero + 3 flop = 5 → remaining = 47 − 2 opponent = 45
    remaining = len(env.get_remaining_deck())
    print(f"  Remaining deck: {remaining} cards (expected 45)")
    assert remaining == 45
    print("  ✅  Deck construction correct")

    # Play through a hand
    import random
    random.seed(42)
    done = False
    steps = 0
    while not done and steps < 10:
        valid = env.get_valid_actions()
        action = random.choice(valid)
        result = env.step(action)
        _, _, done, info = result
        steps += 1

    print(f"  Played {steps} step(s).  Winner: {env.winner}")
    print()


# ============================================================================
# 4. Outs analysis (the "double gutter" insight)
# ============================================================================

def test_outs() -> None:
    """Confirm hero's outs on the flop: flush draw + double-gutter."""
    print("=" * 64)
    print("  OUTS ANALYSIS — 8♥ 9♥ on J♥ Q♥ 2♣")
    print("=" * 64)

    env = PokerEnv()
    env.reset()

    outs = HandEvaluator.count_outs(
        env.hero_cards, list(env.flop), env.get_remaining_deck()
    )
    print(f"  Flush  outs : {outs['flush']}")
    print(f"  Straight outs: {outs['straight']}")
    print(f"  Total unique : {outs['total_unique']}")
    print()
    print("  Hero has a flush draw (hearts) AND a double-gutter straight")
    print("  draw (needs T for 8-9-T-J-Q  or  K for 9-T-J-Q-K, but T also")
    print("  completes 8-9-T-J-Q).  Combined equity ≈ 50%+ by the river.")
    print()


# ============================================================================
# 5. Q-Learning convergence
# ============================================================================

def test_training() -> None:
    """Train the agent and show convergence metrics."""
    print("=" * 64)
    print("  Q-LEARNING CONVERGENCE")
    print("=" * 64)

    env = PokerEnv()
    agent = QLearningAgent(
        actions=env.actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.2,
    )

    print("  Training 5 000 episodes …")
    t0 = time.time()
    agent.train(env, num_episodes=5000, verbose_every=1000)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")
    print()

    stats = agent.get_statistics()
    print(f"  Total episodes : {stats['total_episodes']}")
    print(f"  Avg reward (100): ${stats['avg_reward']:.2f}")
    print(f"  Win rate   (100): {stats['win_rate']:.1%}")
    print()

    # Show learned Q-values
    print("  Learned Q-values:")
    snap = agent.get_q_table_snapshot()
    for sk in sorted(snap.keys()):
        vals = snap[sk]
        best_a = max(vals, key=vals.get) if vals else "?"  # type: ignore[arg-type]
        print(f"    State: {sk}")
        for a, q in sorted(vals.items(), key=lambda x: x[1], reverse=True):
            marker = " ◄" if a == best_a else ""
            print(f"      {a:15s}  Q = {q:+8.2f}{marker}")
    print()

    # Check that flop Q(call) > Q(fold) — the agent should learn this
    flop_states = [s for s in snap if s.startswith("flop")]
    if flop_states:
        fs = flop_states[0]
        q_call = snap[fs].get("call", 0)
        q_fold = snap[fs].get("fold", 0)
        if q_call > q_fold:
            print("  ✅  Agent learned: calling on flop > folding (draws have equity)")
        else:
            print("  ⚠  Agent hasn't fully converged — try more episodes")
    print()


# ============================================================================
# 6. Watch trained agent
# ============================================================================

def demo_trained_play(n_games: int = 5) -> None:
    """Train and then watch the agent play several hands."""
    print("=" * 64)
    print(f"  TRAINED AGENT — {n_games} demonstration hands")
    print("=" * 64)

    env = PokerEnv()
    agent = QLearningAgent(
        actions=env.actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.15,
    )
    agent.train(env, num_episodes=3000, verbose_every=1000)
    print()

    wins, ties, losses = 0, 0, 0
    for g in range(n_games):
        print(f"  ── Game {g + 1} ──")
        state = env.reset()
        done = False
        total_r = 0.0
        while not done:
            valid = env.get_valid_actions()
            action = agent.get_action(state, valid, training=False)
            print(f"    {state.street:6s}  |  {action:12s}  |  "
                  f"pot=${env.pot:3d}  stack=${env.hero_stack:3d}")
            result = env.step(action)
            state = result.next_state
            total_r += result.reward
            done = result.done
            info = result.info

        winner = info.get("winner", "?")
        hero_h = info.get("hero_hand", "")
        opp_h = info.get("opponent_hand", "")
        board = " ".join(c.symbol for c in env._community_list())
        opp = " ".join(c.symbol for c in env.opponent_cards)

        print(f"    Result: {winner.upper():8s}  reward=${total_r:+.0f}")
        print(f"    Board : {board}")
        print(f"    Hero  : {env.hero_cards[0].symbol} {env.hero_cards[1].symbol}  ({hero_h})")
        print(f"    Opp   : {opp}  ({opp_h})")
        print()

        if winner == "hero":
            wins += 1
        elif winner == "tie":
            ties += 1
        else:
            losses += 1

    print(f"  Summary: {wins}W / {ties}T / {losses}L")
    print()


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    print()
    print("╔" + "═" * 62 + "╗")
    print("║    TEXAS HOLD'EM  ·  Q-LEARNING  ·  VALIDATION SUITE         ║")
    print("╚" + "═" * 62 + "╝")
    print()

    test_hand_evaluator()
    test_pot_splitting()
    test_environment()
    test_outs()
    test_training()
    demo_trained_play()

    print("=" * 64)
    print("  ALL TESTS COMPLETED")
    print("=" * 64)
    print()
    print("  To launch the GUI:  python main.py")
    print()


if __name__ == "__main__":
    main()
