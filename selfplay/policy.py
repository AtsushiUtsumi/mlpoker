import random

from poker_domain import Bet, Call, Check, Fold, Raise
from poker_domain.game_state import GameState, WaitingFor

_BASE_WEIGHTS = {
    Fold: 0.20,
    Check: 0.55,
    Call: 0.45,
    Bet: 0.35,
    Raise: 0.30,
}


class RandomPolicy:
    """
    合法なアクションから重み付きランダムに選択する自己対戦用の方策。
    将来的に学習済みモデルに差し替えることを想定したプレースホルダ実装。
    """

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng

    def choose(self, state: GameState, waiting_for: WaitingFor):
        player = next(p for p in state.players if p.player_id == waiting_for.player_id)

        # 手番として問い合わせが来るが既にオールイン済み(ショートスタックがブラインド/
        # アンティで全額拠出済みなど)のケースでは、Call() が唯一の実質ノーオペレーションになる。
        # Fold() を選ぶと拠出済みチップの権利を無意味に放棄してしまうため避ける。
        if player.chips.amount == 0:
            return Call()

        big_blind = state.big_blind.amount
        current_bet = state.current_bet.amount
        pot = state.pot.amount

        feasible = list(waiting_for.valid_actions)
        if Raise in feasible:
            min_raise_total = current_bet * 2
            diff = min_raise_total - player.current_bet.amount
            if diff > player.chips.amount:
                feasible.remove(Raise)

        weights = [_BASE_WEIGHTS[action_cls] for action_cls in feasible]
        action_cls = self._rng.choices(feasible, weights=weights, k=1)[0]

        if action_cls is Fold:
            return Fold()
        if action_cls is Check:
            return Check()
        if action_cls is Call:
            return Call()
        if action_cls is Bet:
            amount = self._choose_bet_amount(player.chips.amount, big_blind, pot)
            return Bet(amount=amount)
        if action_cls is Raise:
            max_total = player.current_bet.amount + player.chips.amount
            min_total = current_bet * 2
            amount = self._choose_raise_amount(min_total, max_total, current_bet, pot)
            return Raise(amount=amount)

        raise AssertionError(f"未知のアクション種別です: {action_cls}")

    def _choose_bet_amount(self, stack: int, big_blind: int, pot: int) -> int:
        if stack <= big_blind:
            return stack  # BB未満は全額オールインのみ合法
        candidates = sorted({
            big_blind,
            min(stack, max(big_blind, pot // 2)),
            min(stack, max(big_blind, pot)),
            stack,
        })
        return self._rng.choice(candidates)

    def _choose_raise_amount(self, min_total: int, max_total: int, current_bet: int, pot: int) -> int:
        if min_total >= max_total:
            return max_total
        candidates = sorted({
            min_total,
            min(max_total, max(min_total, current_bet + pot)),
            min(max_total, max(min_total, min_total + pot)),
            max_total,
        })
        return self._rng.choice(candidates)
