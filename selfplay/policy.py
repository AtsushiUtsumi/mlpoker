import random

from poker_domain import Bet, Call, Check, Fold, HandRank, Raise
from poker_domain.game_state import GamePhase, GameState, WaitingFor

from selfplay.cards import card_to_str, cards_to_list
from selfplay.equity import classify, has_draw, is_nuts

_BASE_WEIGHTS = {
    Fold: 0.20,
    Check: 0.55,
    Call: 0.45,
    Bet: 0.35,
    Raise: 0.30,
}

# Chenフォーミュラのハイカード点数 (rank値 2〜14 -> 点数)
_CHEN_HIGH_CARD_POINTS = {
    14: 10, 13: 8, 12: 7, 11: 6, 10: 5,
    9: 4.5, 8: 4, 7: 3.5, 6: 3, 5: 2.5, 4: 2, 3: 1.5, 2: 1,
}

# プリフロップでこのスコア未満のハンドは弱いハンドとしてフォールドする
_PREFLOP_FOLD_THRESHOLD = 5


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

        override = self._heuristic_override(state, player, feasible)
        if override is not None:
            return override

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

    def _heuristic_override(self, state: GameState, player, feasible: list[type]):
        """
        プレイ傾向を制御するための決め打ちルール。
        - プリフロップ: 弱いハンドはフォールドする
        - フロップ: ワンペア以上、またはドローがあればベット(できなければレイズ/コール)する
        - ターン/リバー: ナッツならベット(できなければレイズ/コール)する
        該当しない場合は None を返し、通常のランダム選択にフォールバックする。
        """
        if not player.hole_cards or len(player.hole_cards) < 2:
            return None

        if state.phase == GamePhase.PRE_FLOP:
            # Checkできる(コスト無し)場面でまで弱いハンドを降りる必要はない
            if Check not in feasible and Fold in feasible:
                if self._chen_score(player.hole_cards) < _PREFLOP_FOLD_THRESHOLD:
                    return Fold()
            return None

        if state.phase not in (GamePhase.FLOP, GamePhase.TURN, GamePhase.RIVER):
            return None

        hole = [card_to_str(c) for c in player.hole_cards]
        community = cards_to_list(state.community_cards)

        if state.phase == GamePhase.FLOP:
            is_strong = classify(hole + community) >= HandRank.ONE_PAIR or has_draw(hole + community)
        else:
            is_strong = is_nuts(hole, community)

        if not is_strong:
            return None
        return self._aggressive_action(feasible, player, state)

    def _aggressive_action(self, feasible: list[type], player, state: GameState):
        """Bet > Raise > Call > Check の優先順で、強いハンドを降りずに攻める行動を選ぶ"""
        big_blind = state.big_blind.amount
        current_bet = state.current_bet.amount
        pot = state.pot.amount

        if Bet in feasible:
            amount = self._choose_bet_amount(player.chips.amount, big_blind, pot)
            return Bet(amount=amount)
        if Raise in feasible:
            max_total = player.current_bet.amount + player.chips.amount
            min_total = current_bet * 2
            amount = self._choose_raise_amount(min_total, max_total, current_bet, pot)
            return Raise(amount=amount)
        if Call in feasible:
            return Call()
        if Check in feasible:
            return Check()
        return Fold()

    @staticmethod
    def _chen_score(hole_cards) -> float:
        """Chenフォーミュラによるプリフロップのハンド強度スコア"""
        ranks = sorted((c.rank.value for c in hole_cards), reverse=True)
        high, low = ranks[0], ranks[1]
        suited = hole_cards[0].suit == hole_cards[1].suit
        points = _CHEN_HIGH_CARD_POINTS[high]

        if high == low:
            return max(points * 2, 5)

        score = points
        if suited:
            score += 2

        gap = high - low - 1
        if gap == 1:
            score -= 1
        elif gap == 2:
            score -= 2
        elif gap == 3:
            score -= 4
        elif gap >= 4:
            score -= 5

        if gap <= 1 and high < 12:
            score += 1  # クイーン未満同士の連番/1ギャップはストレート完成ボーナス

        return score

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
