"""
Poker Engine — Simplified Heads-Up Post-Flop MDP
=================================================
Implements the **Model** layer (MVC) for a simplified heads-up Texas Hold'em
starting from the Flop with fixed hero cards.

Key design decisions
--------------------
* **Hand ranking engine** – robust 5-of-7 evaluator following the strict
  BGC hierarchy: Royal Flush > Straight Flush > 4-of-a-Kind > Full House >
  Flush > Straight > 3-of-a-Kind > 2 Pair > 1 Pair > High Card.
* **Ace dynamics** – Ace is high (A K Q J T) *and* low (5 4 3 2 A).
* **Pot splitting** – on tied ranks the pot is split; odd chips go to hero
  (player left of the dealer, as per BGC rules).
* **Opponent** – treated as a *fixed, stochastic* part of the environment
  (analogous to the Dealer in Blackjack).
* **State space** – ``(street, hero_stack, community_cards_tuple)``.
* **Reward** – ``R = ΔStack_hero``.

Public API
----------
* :class:`Card`
* :class:`HandRank`
* :class:`HandEvaluator`
* :class:`OpponentPolicy`
* :class:`GameState`
* :class:`StepResult`
* :class:`PokerEnv`
"""

from __future__ import annotations

import itertools
import random
from collections import Counter
from typing import (
    Any,
    Dict,
    Final,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
)

from config import (
    ACTIONS,
    HAND_NAMES,
    INITIAL_POT,
    INITIAL_STACK,
    OPPONENT_AGGRESSION,
    OPPONENT_FOLD_PROB,
    RANKS,
    SUIT_SYMBOLS,
    SUITS,
)


# ============================================================================
# Card
# ============================================================================

class Card:
    """Immutable representation of a standard playing card.

    Parameters
    ----------
    rank : str
        One of ``2 3 4 5 6 7 8 9 T J Q K A``.
    suit : str
        One of ``h d c s`` (hearts, diamonds, clubs, spades).
    """

    RANKS: Final[List[str]] = RANKS
    SUITS: Final[List[str]] = SUITS
    SUIT_SYMBOLS: Final[Dict[str, str]] = SUIT_SYMBOLS

    __slots__ = ("rank", "suit", "rank_value")

    def __init__(self, rank: str, suit: str) -> None:
        if rank not in self.RANKS:
            raise ValueError(f"Invalid rank: {rank!r}")
        if suit not in self.SUITS:
            raise ValueError(f"Invalid suit: {suit!r}")
        self.rank: str = rank
        self.suit: str = suit
        self.rank_value: int = self.RANKS.index(rank)

    # -- Display helpers -----------------------------------------------------

    @property
    def symbol(self) -> str:
        """Unicode pretty-print, e.g. ``A♠``."""
        return f"{self.rank}{self.SUIT_SYMBOLS[self.suit]}"

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def __repr__(self) -> str:
        return f"Card({self.rank!r}, {self.suit!r})"

    # -- Equality / hashing --------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

    def __hash__(self) -> int:
        return hash((self.rank, self.suit))


# ============================================================================
# Hand Evaluator
# ============================================================================

class HandRank(NamedTuple):
    """Comparable hand evaluation result.

    Attributes
    ----------
    rank : int
        1 (High Card) … 10 (Royal Flush).
    tiebreakers : Tuple[int, ...]
        Descending kicker values for tie-breaking.
    name : str
        Human-readable hand name.
    """
    rank: int
    tiebreakers: Tuple[int, ...]
    name: str


class HandEvaluator:
    """Evaluate poker hands strictly following the BGC Texas Hold'em rules.

    Hierarchy (high → low)::

        10  Royal Flush
         9  Straight Flush
         8  Four of a Kind
         7  Full House
         6  Flush
         5  Straight
         4  Three of a Kind
         3  Two Pair
         2  One Pair
         1  High Card

    Ace dynamics:
        • High straight: A K Q J T
        • Low  straight: 5 4 3 2 A  (the *wheel*)
    """

    HAND_NAMES: Final[Dict[int, str]] = HAND_NAMES

    # -- Public API ----------------------------------------------------------

    @staticmethod
    def evaluate_hand(cards: Sequence[Card]) -> HandRank:
        """Evaluate the best 5-card hand from *cards* (5–7 cards).

        Returns a :class:`HandRank` that is directly comparable via ``<``,
        ``>``, ``==``.
        """
        if len(cards) < 5:
            raise ValueError(f"Need >= 5 cards, got {len(cards)}")

        best: Optional[HandRank] = None
        for combo in itertools.combinations(cards, 5):
            hr = HandEvaluator._evaluate_five(list(combo))
            if best is None or hr > best:
                best = hr
        assert best is not None
        return best

    @staticmethod
    def compare_hands(
        hand1_cards: Sequence[Card],
        hand2_cards: Sequence[Card],
    ) -> int:
        """Compare two 7-card pools.

        Returns
        -------
        int
            ``1`` if *hand1* wins, ``-1`` if *hand2* wins, ``0`` on tie.
        """
        hr1 = HandEvaluator.evaluate_hand(hand1_cards)
        hr2 = HandEvaluator.evaluate_hand(hand2_cards)
        if hr1 > hr2:
            return 1
        if hr1 < hr2:
            return -1
        return 0

    @staticmethod
    def hand_name(rank: int) -> str:
        """Human-readable name for a numeric hand rank."""
        return HandEvaluator.HAND_NAMES.get(rank, "Unknown")

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _evaluate_five(cards: List[Card]) -> HandRank:
        """Classify exactly five cards and return a :class:`HandRank`."""
        is_flush = HandEvaluator._is_flush(cards)
        is_straight, straight_high = HandEvaluator._is_straight(cards)
        rank_counts = Counter(c.rank_value for c in cards)
        counts_sorted = sorted(rank_counts.values(), reverse=True)

        # Tiebreakers: grouped by frequency then by rank value (desc).
        sorted_groups = sorted(
            rank_counts.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        tiebreakers = tuple(rv for rv, _ in sorted_groups)

        # --- Royal Flush (A-K-Q-J-T suited) ---
        if is_flush and is_straight and straight_high == 12:
            return HandRank(10, (12,), "Royal Flush")

        # --- Straight Flush ---
        if is_flush and is_straight:
            return HandRank(9, (straight_high,), "Straight Flush")

        # --- Four of a Kind ---
        if counts_sorted == [4, 1]:
            return HandRank(8, tiebreakers, "Four of a Kind")

        # --- Full House ---
        if counts_sorted == [3, 2]:
            return HandRank(7, tiebreakers, "Full House")

        # --- Flush ---
        if is_flush:
            desc = tuple(sorted((c.rank_value for c in cards), reverse=True))
            return HandRank(6, desc, "Flush")

        # --- Straight ---
        if is_straight:
            return HandRank(5, (straight_high,), "Straight")

        # --- Three of a Kind ---
        if counts_sorted == [3, 1, 1]:
            return HandRank(4, tiebreakers, "Three of a Kind")

        # --- Two Pair ---
        if counts_sorted == [2, 2, 1]:
            return HandRank(3, tiebreakers, "Two Pair")

        # --- One Pair ---
        if counts_sorted == [2, 1, 1, 1]:
            return HandRank(2, tiebreakers, "One Pair")

        # --- High Card ---
        desc = tuple(sorted((c.rank_value for c in cards), reverse=True))
        return HandRank(1, desc, "High Card")

    @staticmethod
    def _is_flush(cards: List[Card]) -> bool:
        return len({c.suit for c in cards}) == 1

    @staticmethod
    def _is_straight(cards: List[Card]) -> Tuple[bool, int]:
        """Return ``(is_straight, high_card_rank_value)``.

        Handles the *wheel* (A-2-3-4-5) where Ace plays low.
        """
        ranks = sorted(c.rank_value for c in cards)

        # Normal straight
        if ranks == list(range(ranks[0], ranks[0] + 5)):
            return True, ranks[-1]

        # Wheel (A-2-3-4-5): rank values [0,1,2,3,12]
        if ranks == [0, 1, 2, 3, 12]:
            return True, 3  # 5-high

        return False, 0

    # -- Outs / draw helpers (used by GUI thought-process display) -----------

    @staticmethod
    def count_outs(
        hero_cards: List[Card],
        community: List[Card],
        remaining_deck: List[Card],
    ) -> Dict[str, int]:
        """Count the number of outs that improve hero's hand rank.

        Returns a dict mapping improvement type to count, e.g.
        ``{'flush': 9, 'straight': 6, 'total_unique': 15}``.
        """
        current_pool = hero_cards + community
        current_hr = (
            HandEvaluator.evaluate_hand(current_pool)
            if len(current_pool) >= 5
            else None
        )

        flush_outs: int = 0
        straight_outs: int = 0
        improving_cards: List[Card] = []

        for card in remaining_deck:
            test_pool = current_pool + [card]
            if len(test_pool) < 5:
                continue
            test_hr = HandEvaluator.evaluate_hand(test_pool)
            if current_hr is None or test_hr > current_hr:
                improving_cards.append(card)
                if test_hr.name in ("Flush", "Royal Flush", "Straight Flush"):
                    flush_outs += 1
                if test_hr.name in ("Straight", "Straight Flush"):
                    straight_outs += 1

        return {
            "flush": flush_outs,
            "straight": straight_outs,
            "total_unique": len(set(improving_cards)),
        }


# ============================================================================
# Opponent Policy
# ============================================================================

class OpponentPolicy:
    """Fixed stochastic opponent — part of the environment, not an agent.

    Parameters
    ----------
    aggression : float
        Probability of raising (vs. calling) when not folding.  0.0–1.0.
    fold_prob : float
        Base probability of folding (before adjustments).
    """

    def __init__(
        self,
        aggression: float = OPPONENT_AGGRESSION,
        fold_prob: float = OPPONENT_FOLD_PROB,
    ) -> None:
        self.aggression: float = aggression
        self.fold_prob: float = fold_prob

    def get_action(
        self,
        state: Dict[str, Any],
        valid_actions: List[str],
    ) -> str:
        """Select an action stochastically."""
        if "fold" in valid_actions and random.random() < self.fold_prob:
            return "fold"
        if "raise_100" in valid_actions and random.random() < self.aggression:
            return "raise_100"
        if "call" in valid_actions:
            return "call"
        return valid_actions[0]


# ============================================================================
# State representation
# ============================================================================

class GameState(NamedTuple):
    """Observable game state exposed to the agent.

    The ``community`` tuple is ordered (flop1, flop2, flop3[, turn[, river]]).
    """
    street: str
    hero_cards: Tuple[Card, Card]
    community: Tuple[Card, ...]
    pot: int
    hero_stack: int
    opponent_stack: int
    hero_invested: int
    opponent_invested: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to plain dict (useful for opponent policy)."""
        return {
            "street": self.street,
            "hero_cards": list(self.hero_cards),
            "community": list(self.community),
            "pot": self.pot,
            "hero_stack": self.hero_stack,
            "opponent_stack": self.opponent_stack,
            "hero_invested": self.hero_invested,
            "opponent_invested": self.opponent_invested,
        }

    @property
    def state_key(self) -> str:
        """Compact hashable string for the Q-table."""
        return f"{self.street}_{self.pot}_{self.hero_stack}"


# ============================================================================
# Step result
# ============================================================================

class StepResult(NamedTuple):
    """Value returned by :meth:`PokerEnv.step`."""
    next_state: GameState
    reward: float
    done: bool
    info: Dict[str, Any]


# ============================================================================
# Poker Environment (MDP)
# ============================================================================

class PokerEnv:
    """Simplified Heads-Up Texas Hold'em — post-flop MDP.

    Initial conditions (per spec)
    -----------------------------
    * Hero:  8h 9h
    * Flop:  Jh Qh 2c
    * Stacks: $150 each
    * Pot:   $100

    Hero holds a *double-gutter straight draw* (needs T or K) **and** a
    heart flush draw (9 outs).  The agent should learn the combined ~50%
    equity and adjust its Q-values for 'call'/'raise' accordingly.

    Transition dynamics
    -------------------
    After hero and opponent act on each street, the next community card is
    dealt uniformly from the 47 remaining unknowns (52 - 2 hero - 3 flop).

    Reward
    ------
    ``R = Delta_Stack_hero = hero_stack_final - hero_stack_initial``

    Pot splitting
    -------------
    On a tie the pot is split evenly; the **odd chip** goes to hero (player
    left of the dealer).
    """

    ACTIONS: Final[List[str]] = ACTIONS

    def __init__(self) -> None:
        # Fixed starting hand & flop
        self.hero_cards: List[Card] = [Card("8", "h"), Card("9", "h")]
        self.flop: List[Card] = [Card("J", "h"), Card("Q", "h"), Card("2", "c")]

        # Mutable game state
        self.deck: List[Card] = []
        self.turn: Optional[Card] = None
        self.river: Optional[Card] = None
        self.opponent_cards: List[Card] = []

        self.pot: int = INITIAL_POT
        self.hero_stack: int = INITIAL_STACK
        self.opponent_stack: int = INITIAL_STACK
        self.hero_invested: int = 0
        self.opponent_invested: int = 0

        self.street: str = "flop"
        self.done: bool = False
        self.winner: Optional[str] = None

        self.opponent_policy: OpponentPolicy = OpponentPolicy(
            aggression=OPPONENT_AGGRESSION,
        )

        # Expose action list for agent constructor
        self.actions: List[str] = list(self.ACTIONS)

    # -- Public API ----------------------------------------------------------

    def reset(self) -> GameState:
        """Reset for a new episode.  Returns the initial state."""
        self.deck = [
            Card(r, s)
            for r in Card.RANKS
            for s in Card.SUITS
            if Card(r, s) not in self.hero_cards
            and Card(r, s) not in self.flop
        ]
        random.shuffle(self.deck)

        self.opponent_cards = [self.deck.pop(), self.deck.pop()]

        self.turn = None
        self.river = None
        self.pot = INITIAL_POT
        self.hero_stack = INITIAL_STACK
        self.opponent_stack = INITIAL_STACK
        self.hero_invested = 0
        self.opponent_invested = 0
        self.street = "flop"
        self.done = False
        self.winner = None

        return self._get_state()

    def get_valid_actions(self) -> List[str]:
        """Actions available to hero in the current state."""
        if self.done:
            return []
        valid: List[str] = ["fold", "call"]
        if self.hero_stack >= 100:
            valid.append("raise_100")
        if self.hero_stack >= 150:
            valid.append("raise_150")
        return valid

    def step(self, action: str) -> StepResult:
        """Execute *action* and return ``(next_state, reward, done, info)``.

        The reward is ``R = Delta_Stack_hero``.
        """
        if self.done:
            return StepResult(self._get_state(), 0.0, True, {})

        initial_stack: int = self.hero_stack

        # ---- Hero acts ------------------------------------------------
        if action == "fold":
            self.done = True
            self.winner = "opponent"
            reward = float(-(initial_stack - self.hero_stack) - self.hero_invested)
            return StepResult(
                self._get_state(), reward, True, {"winner": "opponent"}
            )

        if action == "call":
            to_call = max(0, self.opponent_invested - self.hero_invested)
            actual = min(to_call, self.hero_stack)
            self.hero_stack -= actual
            self.hero_invested += actual
            self.pot += actual

        elif action == "raise_100":
            total_bet = 100 + max(0, self.opponent_invested - self.hero_invested)
            actual = min(total_bet, self.hero_stack)
            self.hero_stack -= actual
            self.hero_invested += actual
            self.pot += actual

        elif action == "raise_150":
            actual = self.hero_stack
            self.hero_invested += actual
            self.pot += actual
            self.hero_stack = 0

        # ---- Opponent responds ----------------------------------------
        opp_valid: List[str] = ["fold", "call"]
        if self.opponent_stack >= 100:
            opp_valid.append("raise_100")

        opp_action = self.opponent_policy.get_action(
            self._get_state().to_dict(), opp_valid
        )

        if opp_action == "fold":
            self.done = True
            self.winner = "hero"
            reward = float(self.pot - (initial_stack - self.hero_stack))
            return StepResult(
                self._get_state(), reward, True, {"winner": "hero"}
            )

        if opp_action == "call":
            to_call = max(0, self.hero_invested - self.opponent_invested)
            actual = min(to_call, self.opponent_stack)
            self.opponent_stack -= actual
            self.opponent_invested += actual
            self.pot += actual

        elif opp_action == "raise_100":
            total_bet = 100 + max(0, self.hero_invested - self.opponent_invested)
            actual = min(total_bet, self.opponent_stack)
            self.opponent_stack -= actual
            self.opponent_invested += actual
            self.pot += actual
            # Hero auto-calls the raise (simplified)
            to_call = max(0, self.opponent_invested - self.hero_invested)
            actual_call = min(to_call, self.hero_stack)
            self.hero_stack -= actual_call
            self.hero_invested += actual_call
            self.pot += actual_call

        # ---- Advance street -------------------------------------------
        if self.street == "flop":
            self.turn = self.deck.pop()
            self.street = "turn"
        elif self.street == "turn":
            self.river = self.deck.pop()
            self.street = "river"
        elif self.street == "river":
            return self._showdown(initial_stack)

        return StepResult(self._get_state(), 0.0, False, {})

    def get_remaining_deck(self) -> List[Card]:
        """Cards remaining in the deck (for outs calculation)."""
        return list(self.deck)

    # -- Display helper ------------------------------------------------------

    def render(self) -> str:
        """Pretty-print the current game state."""
        community = self._community_list()
        board = " ".join(c.symbol for c in community) if community else "---"
        hero = " ".join(c.symbol for c in self.hero_cards)
        lines = [
            f"Street : {self.street.upper()}",
            f"Hero   : {hero}",
            f"Board  : {board}",
            f"Pot    : ${self.pot}  |  Hero Stack: ${self.hero_stack}",
        ]
        return "\n".join(lines)

    # -- Internal helpers ----------------------------------------------------

    def _community_list(self) -> List[Card]:
        community = list(self.flop)
        if self.turn is not None:
            community.append(self.turn)
        if self.river is not None:
            community.append(self.river)
        return community

    def _get_state(self) -> GameState:
        return GameState(
            street=self.street,
            hero_cards=(self.hero_cards[0], self.hero_cards[1]),
            community=tuple(self._community_list()),
            pot=self.pot,
            hero_stack=self.hero_stack,
            opponent_stack=self.opponent_stack,
            hero_invested=self.hero_invested,
            opponent_invested=self.opponent_invested,
        )

    def _showdown(self, initial_stack: int) -> StepResult:
        """Resolve the hand at showdown with proper pot splitting."""
        self.street = "showdown"
        self.done = True

        community = self._community_list()
        hero_pool = list(self.hero_cards) + community
        opp_pool = list(self.opponent_cards) + community

        result = HandEvaluator.compare_hands(hero_pool, opp_pool)

        if result > 0:
            self.winner = "hero"
            reward = float(self.pot - (initial_stack - self.hero_stack))
        elif result < 0:
            self.winner = "opponent"
            reward = float(-(initial_stack - self.hero_stack))
        else:
            # Tie — split pot.  Odd chip to hero (left of dealer).
            self.winner = "tie"
            hero_share = (self.pot + 1) // 2  # odd chip -> hero
            reward = float(hero_share - (initial_stack - self.hero_stack))

        hero_hr = HandEvaluator.evaluate_hand(hero_pool)
        opp_hr = HandEvaluator.evaluate_hand(opp_pool)

        info: Dict[str, Any] = {
            "winner": self.winner,
            "hero_hand": hero_hr.name,
            "opponent_hand": opp_hr.name,
            "hero_rank": hero_hr.rank,
            "opponent_rank": opp_hr.rank,
        }
        return StepResult(self._get_state(), reward, True, info)
