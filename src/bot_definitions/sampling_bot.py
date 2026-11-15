"""Level 4: bounded, weighted determinization with level-2 continuation policies.

Contributors: Grace Guo (85375651b), OpenAI Codex (implementation assistance).

Sample unseen cards without replacement into opponent hands and the deck, and
shuffle only the publicly known remaining auction-card multiset. A world's
importance weight is exp(evidence_weight * level-3 remaining-hand log evidence).
Compare the SAME worlds/randomness for all candidate bids and maximize weighted
mean OWN final coin score, not the number of rollout wins. Every simulated
player uses current level 2 with its own private view and reconstructed learning
history. All non-inference level-3 experiments stay off; actual reveals are the
level-2 first-card rule. Candidate and time limits make this an approximate
search, with an explicit level-2 fallback if fewer than two worlds finish.
"""

from collections import Counter
from math import exp, isfinite
from random import Random, getrandbits
from time import perf_counter

from bot_definitions.cash_fraction_model import available_budget
from bot_definitions.sampled_rollout import (
    RolloutDeadline,
    SampledWorld,
    finish_rollout,
    historical_rates,
    interruptible_deadline,
    public_players,
    root_bids,
    sample_world,
    world_game,
)
from bot_definitions.soft_tracking_bot import SoftTrackingBot
from bot_definitions.valuation_bot import ValuationBot
from bots import BotInfo, RoundResult
from constants import STANDARD_GEMS
from harness import GameState


def weighted_means(
    log_weights: list[float],
    outcomes: list[list[int]],
) -> tuple[list[float], float]:
    """Self-normalized importance means and effective sample size, stably scaled."""
    if not log_weights or len(log_weights) != len(outcomes):
        raise ValueError("One nonempty outcome vector is required per world")
    width = len(outcomes[0])
    if not width or any(len(row) != width for row in outcomes):
        raise ValueError("Every world must evaluate the same candidate set")
    maximum = max(log_weights)
    weights = [exp(log_weight - maximum) for log_weight in log_weights]
    total = sum(weights)
    means = [
        sum(w * row[i] for w, row in zip(weights, outcomes)) / total
        for i in range(width)
    ]
    effective = total**2 / sum(w * w for w in weights)
    return means, effective


class SamplingBot(SoftTrackingBot):
    """A separate public level-4 bot; only the level-3 hand tracker is required."""

    use_bid_evidence = True
    use_endgame_rules = False
    use_finance_horizon = False
    use_cash_fraction_model = False
    use_strategic_reveal = False

    def __init__(
        self,
        *,
        decision_seconds: float = 0.85,
        max_worlds: int = 16,
        max_candidates: int = 7,
        evidence_weight: float = 0.1,
    ) -> None:
        if not isfinite(decision_seconds) or not 0 <= decision_seconds <= 1.35:
            raise ValueError("Search allowance must be between zero and 1.35 seconds")
        if type(max_worlds) is not int or max_worlds < 2:
            raise ValueError("Search requires at least two possible completed worlds")
        if type(max_candidates) is not int or max_candidates < 2:
            raise ValueError("Search requires at least two candidate slots")
        self.decision_seconds = decision_seconds
        self.max_worlds = max_worlds
        self.max_candidates = max_candidates
        super().__init__(
            use_bid_evidence=True,
            evidence_weight=evidence_weight,
            use_endgame_rules=False,
            use_finance_horizon=False,
            use_cash_fraction_model=False,
            use_strategic_reveal=False,
            use_bid_model=True,
            use_cash_limits=True,
        )

    def on_game_start(self) -> None:
        super().on_game_start()
        self.baseline = ValuationBot()
        self.history: list[tuple[BotInfo, RoundResult]] = []
        self.search_log: list[dict[str, object]] = []
        self.last_search: dict[str, object] = {}

    def on_round_end(self, info: BotInfo, result: RoundResult) -> None:
        before = self._pending
        self.baseline.on_round_end(info, result)
        super().on_round_end(info, result)
        if before is not None and before.round_number == info.round_number:
            self.history.append((before, result))

    def _prepare(
        self,
        info: BotInfo,
        world: SampledWorld,
        deadline: float,
        rate_cache: dict[tuple[int, tuple[int, ...]], dict[int, float]],
    ) -> tuple[GameState, list[int]]:
        rates = []
        for player in public_players(info):
            if perf_counter() >= deadline:
                raise RolloutDeadline()
            if player.id == info.self_player.id:
                rates.append(dict(self.baseline.bid_rates))
                continue
            counts = Counter(world.hands[player.id])
            key = player.id, tuple(counts[g] for g in STANDARD_GEMS)
            if key not in rate_cache:
                rate_cache[key] = historical_rates(info, world, player.id, self.history)
            rates.append(rate_cache[key])
        game = world_game(info, world, rates, deadline)
        return game, root_bids(game)

    def _candidates(
        self,
        info: BotInfo,
        baseline: int,
        proposals: list[tuple[GameState, list[int]]],
    ) -> list[int]:
        budget = available_budget(info, info.self_player)
        ceiling = min(
            budget, max(available_budget(info, p) for p in info.other_players) + 1
        )
        if ceiling + 1 <= self.max_candidates:
            return list(range(ceiling + 1))
        candidates = [0, baseline, ceiling]
        thresholds: list[int] = []
        for _, bids in proposals:
            maximum = max(bids[p.id] for p in info.other_players)
            tied = [p.id for p in info.other_players if bids[p.id] == maximum]
            previous = info.previous_winner_id
            if previous is None:
                thresholds.extend((maximum, maximum + 1))
            else:
                our_rank = (info.self_player.id - previous - 1) % info.num_players
                wins_tie = all(
                    our_rank < (i - previous - 1) % info.num_players for i in tied
                )
                thresholds.append(maximum + int(not wins_tie))
        thresholds.sort()
        if thresholds:
            candidates.extend(
                thresholds[round(q * (len(thresholds) - 1))] for q in (0.0, 0.5, 1.0)
            )
        candidates.extend((baseline - 1, baseline + 1))
        unique = []
        for bid in candidates:
            if 0 <= bid <= ceiling and bid not in unique:
                unique.append(bid)
        return sorted(unique[: self.max_candidates])

    def get_bid(self, info: BotInfo) -> int:
        started = perf_counter()
        self.observe_auction(info)
        # Preserve real pre-bid memory even if the baseline computation itself
        # is interrupted. Its on_round_end still learns this actual auction.
        self.baseline._pending = info
        allowance = self.decision_seconds
        if self.timeout_seconds is not None:
            allowance = min(allowance, max(0.0, self.timeout_seconds - 0.30))
        diagnostics: dict[str, object] = {
            "round": info.round_number,
            "worlds": 0,
            "rollouts": 0,
            "baseline_bid": 0,
            "chosen_bid": 0,
            "baseline_ready": False,
            "fallback": None,
            "seconds": 0.0,
        }
        if allowance <= 0 or info.current_auction is None:
            baseline = self.baseline.get_bid(info)
            diagnostics.update(
                baseline_bid=baseline,
                chosen_bid=baseline,
                baseline_ready=True,
                fallback="no_search_time",
            )
            return self._record(baseline, diagnostics, started)
        baseline = 0
        try:
            with interruptible_deadline(
                max(1e-6, started + allowance - perf_counter())
            ) as enabled:
                diagnostics["interruptible_deadline"] = enabled
                baseline = self.baseline.get_bid(info)
                diagnostics.update(baseline_bid=baseline, baseline_ready=True)
                if available_budget(info, info.self_player) == 0:
                    diagnostics["trivial"] = "only_legal_bid_zero"
                    chosen = 0
                else:
                    chosen = self._search_bid(
                        info, baseline, started + allowance, diagnostics
                    )
        except RolloutDeadline:
            chosen = baseline
            diagnostics["fallback"] = (
                "search_deadline"
                if diagnostics["baseline_ready"]
                else "baseline_deadline_zero_bid"
            )
        diagnostics["chosen_bid"] = chosen
        # Outside the alarm: statistics and checkpoint transport are never
        # interrupted midway by a leftover search timer.
        return self._record(chosen, diagnostics, started)

    def _search_bid(
        self,
        info: BotInfo,
        baseline: int,
        deadline: float,
        diagnostics: dict[str, object],
    ) -> int:
        rng = Random(getrandbits(128))
        prepared: list[tuple[GameState, list[int]]] = []
        worlds: list[SampledWorld] = []
        cache: dict[tuple[int, tuple[int, ...]], dict[int, float]] = {}
        try:
            for _ in range(min(4, self.max_worlds)):
                world = sample_world(info, self.likelihoods, self.evidence_weight, rng)
                prepared.append(self._prepare(info, world, deadline, cache))
                worlds.append(world)
        except RolloutDeadline:
            diagnostics["fallback"] = "preparation_deadline"
            return baseline
        candidates = self._candidates(info, baseline, prepared)
        # Measure a real continuation before allocating the remaining time.
        # Adapt action COUNT using cost only, never the pilot's score.
        pilot_start = perf_counter()
        try:
            pilot = finish_rollout(
                prepared[0][0], prepared[0][1], info.self_player.id, baseline, deadline
            )[info.self_player.id]
        except RolloutDeadline:
            diagnostics["fallback"] = "pilot_deadline"
            return baseline
        pilot_seconds = max(1e-6, perf_counter() - pilot_start)
        remaining = max(0.0, deadline - perf_counter())
        active_limit = max(2, int((remaining / (1.2 * pilot_seconds) + 1) / 3))
        if len(candidates) > active_limit:
            priorities = [baseline]
            # Root proposals identify likely winning thresholds independently
            # of the pilot continuation's terminal reward.
            target = (
                sorted(max(b[p.id] for p in info.other_players) for _, b in prepared)[
                    len(prepared) // 2
                ]
                + 1
            )
            priorities += sorted(candidates, key=lambda c: (abs(c - target), c))
            keep: list[int] = []
            for bid in priorities:
                if bid in candidates and bid not in keep:
                    keep.append(bid)
            candidates = sorted(keep[:active_limit])
        diagnostics["pilot_seconds"] = pilot_seconds
        diagnostics["candidates"] = candidates
        outcomes: list[list[int]] = []
        weights: list[float] = []
        last_batch = 0.0
        for index in range(self.max_worlds):
            if perf_counter() + last_batch * 1.15 >= deadline:
                break
            batch_start = perf_counter()
            try:
                if index >= len(prepared):
                    world = sample_world(
                        info, self.likelihoods, self.evidence_weight, rng
                    )
                    game, bids = self._prepare(info, world, deadline, cache)
                else:
                    world = worlds[index]
                    game, bids = prepared[index]
                # Commit a world only after ALL its candidate continuations finish.
                scores = [
                    pilot
                    if index == 0 and bid == baseline
                    else finish_rollout(game, bids, info.self_player.id, bid, deadline)[
                        info.self_player.id
                    ]
                    for bid in candidates
                ]
            except RolloutDeadline:
                diagnostics["abandoned_worlds"] = 1
                break
            outcomes.append(scores)
            weights.append(world.log_weight)
            last_batch = perf_counter() - batch_start
        diagnostics["worlds"] = len(outcomes)
        diagnostics["rollouts"] = len(outcomes) * len(candidates)
        if len(outcomes) < 2:
            diagnostics["fallback"] = "fewer_than_two_complete_worlds"
            return baseline
        means, effective = weighted_means(weights, outcomes)
        best = max(range(len(candidates)), key=lambda i: (means[i], -candidates[i]))
        diagnostics["mean_final_scores"] = dict(zip(candidates, means))
        diagnostics["effective_worlds"] = effective
        diagnostics["chosen_bid"] = candidates[best]
        return candidates[best]

    def _record(self, bid: int, diagnostics: dict[str, object], started: float) -> int:
        diagnostics["seconds"] = perf_counter() - started
        self.last_search = diagnostics
        self.search_log.append(diagnostics)
        del self.search_log[:-25]
        return bid
