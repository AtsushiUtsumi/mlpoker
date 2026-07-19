import argparse
import random

from poker_domain import Bet, Call, Check, Chips, Fold, PokerTable, Raise
from poker_domain.game_state import EventType, GameState, TableStatus

from selfplay.cards import cards_to_list
from selfplay.logger import JsonlWriter
from selfplay.policy import RandomPolicy


def _action_name_amount(action) -> tuple[str, int | None]:
    if isinstance(action, Fold):
        return "fold", None
    if isinstance(action, Check):
        return "check", None
    if isinstance(action, Call):
        return "call", None
    if isinstance(action, Bet):
        return "bet", action.amount
    if isinstance(action, Raise):
        return "raise", action.amount
    raise AssertionError(f"未知のアクション: {action}")


def _serialize_hand(hand) -> dict:
    return {
        "rank": hand.rank.name,
        "tiebreakers": list(hand.tiebreakers),
        "cards": cards_to_list(hand.cards),
    }


def _serialize_pot(pot) -> dict:
    return {"amount": pot.amount.amount, "eligible_player_ids": list(pot.eligible_player_ids)}


def _player_snapshot(state: GameState) -> list[dict]:
    return [
        {
            "player_id": p.player_id,
            "chips": p.chips.amount,
            "current_bet": p.current_bet.amount,
            "folded": p.folded,
            "is_all_in": p.is_all_in,
        }
        for p in state.players
    ]


def _hand_start_players(table: PokerTable, state: GameState) -> list[dict]:
    """このハンドに参加する各プレイヤーのホールカードを、それぞれの視点で取得する"""
    players = []
    for p in state.players:
        viewer_state = table.get_state(viewer_player_id=p.player_id)
        own = next(vp for vp in viewer_state.players if vp.player_id == p.player_id)
        players.append({
            "player_id": p.player_id,
            "chips": p.chips.amount,
            "hole_cards": cards_to_list(own.hole_cards or ()),
        })
    return players


def run_game(game_id: int, hand_counter: int, args: argparse.Namespace,
             writer: JsonlWriter, policy: RandomPolicy) -> int:
    table = PokerTable(
        table_id=f"game-{game_id}",
        max_players=args.num_players,
        small_blind=args.small_blind,
        big_blind=args.big_blind,
        ante=args.ante,
    )
    player_ids = [f"bot{i}" for i in range(args.num_players)]
    for pid in player_ids:
        table.add_player(pid, Chips(args.starting_chips))

    writer.write({
        "type": "game_start",
        "game_id": game_id,
        "player_ids": player_ids,
        "starting_chips": args.starting_chips,
        "small_blind": args.small_blind,
        "big_blind": args.big_blind,
        "ante": args.ante,
    })

    hand_in_game = 0
    while table.get_table_status() != TableStatus.CLOSED:
        result = table.start_game()
        hand_counter += 1
        hand_in_game += 1

        writer.write({
            "type": "hand_start",
            "game_id": game_id,
            "hand_id": hand_counter,
            "hand_in_game": hand_in_game,
            "dealer_id": result.state.dealer_id,
            "small_blind": result.state.small_blind.amount,
            "big_blind": result.state.big_blind.amount,
            "ante": result.state.ante.amount,
            "level": result.state.level,
            "players": _hand_start_players(table, result.state),
        })

        while result.waiting_for is not None:
            acting_id = result.waiting_for.player_id
            state = table.get_state(viewer_player_id=acting_id)
            action = policy.choose(state, result.waiting_for)
            action_name, amount = _action_name_amount(action)

            writer.write({
                "type": "action",
                "game_id": game_id,
                "hand_id": hand_counter,
                "phase": state.phase.name,
                "player_id": acting_id,
                "action": action_name,
                "amount": amount,
                "pot_before": state.pot.amount,
                "current_bet_before": state.current_bet.amount,
                "community_cards_before": cards_to_list(state.community_cards),
                "players_before": _player_snapshot(state),
            })

            result = table.action(acting_id, action)

        showdown_event = next(
            (e for e in result.events if e.event_type == EventType.SHOWDOWN), None
        )
        payload = showdown_event.payload if showdown_event else {}
        final_state = table.get_state()

        writer.write({
            "type": "hand_end",
            "game_id": game_id,
            "hand_id": hand_counter,
            "winner_id": payload.get("winner_id"),
            "payouts": payload.get("payouts", {}),
            "pots": [_serialize_pot(p) for p in payload.get("pots", ())],
            "rake": payload.get("rake", 0),
            "hands": {pid: _serialize_hand(h) for pid, h in payload.get("hands", {}).items()},
            "chips_after": {p.player_id: p.chips.amount for p in final_state.players},
        })

    final_state = table.get_state()
    remaining_chips = {p.player_id: p.chips.amount for p in final_state.players}
    writer.write({
        "type": "game_end",
        "game_id": game_id,
        "hands_played": hand_in_game,
        # 既にバストして卓の内部ロースターから外れたプレイヤーは 0 として補完する
        "final_chips": {pid: remaining_chips.get(pid, 0) for pid in player_ids},
    })

    return hand_counter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="poker-domain を使った自己対戦プレイログ(JSONL)の生成スクリプト"
    )
    parser.add_argument(
        "--target-hands", type=int, default=100_000,
        help="生成する総ハンド数の目安。このハンド数を超えるまでゲームを繰り返す (既定値: 100000)",
    )
    parser.add_argument("--num-players", type=int, default=6, help="卓の人数 (既定値: 6)")
    parser.add_argument("--starting-chips", type=int, default=1000, help="各プレイヤーの初期チップ")
    parser.add_argument("--small-blind", type=int, default=25)
    parser.add_argument("--big-blind", type=int, default=50)
    parser.add_argument("--ante", type=int, default=0)
    parser.add_argument("--output", type=str, default="logs/playlog.jsonl", help="出力するJSONLファイルのパス")
    parser.add_argument("--seed", type=int, default=None, help="乱数シード (再現性が必要な場合に指定)")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    policy = RandomPolicy(rng)

    hand_counter = 0
    game_id = 0
    with JsonlWriter(args.output) as writer:
        while hand_counter < args.target_hands:
            game_id += 1
            hand_counter = run_game(game_id, hand_counter, args, writer, policy)
            print(f"[game {game_id}] 終了。累計ハンド数: {hand_counter}")

    print(f"完了: {game_id} ゲーム, 合計 {hand_counter} ハンドを {args.output} に出力しました")


if __name__ == "__main__":
    main()
