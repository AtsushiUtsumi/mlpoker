import argparse
import csv
import json
import random

from selfplay.equity import HAND_RANK_ORDER, estimate_hand_rank_probabilities

_PROB_COLUMNS = [f"prob_{r.name.lower()}" for r in HAND_RANK_ORDER]

_HEADER = [
    "game_id", "hand_id", "phase", "player_id", "action", "amount",
    "pot_before", "current_bet_before", "player_chips_before", "spr",
    *_PROB_COLUMNS, "equity_samples",
]


def _spr(chips: int, pot: int) -> float | str:
    if pot <= 0:
        return ""
    return chips / pot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="プレイログ(JSONL)から SPR と役ごとの完成確率を事前計算してCSVに出力する"
    )
    parser.add_argument("--input", type=str, default="logs/playlog.jsonl", help="入力するプレイログ(JSONL)")
    parser.add_argument("--output", type=str, default="logs/features.csv", help="出力するCSVのパス")
    parser.add_argument(
        "--samples", type=int, default=300,
        help="PRE_FLOP(残り5枚)時のモンテカルロ・サンプル数。FLOP以降は厳密に全パターンを列挙する (既定値: 300)",
    )
    parser.add_argument("--seed", type=int, default=None, help="モンテカルロ・サンプリングの乱数シード")
    parser.add_argument("--limit", type=int, default=None, help="デバッグ用: 先頭N件のactionレコードのみ処理する")
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # 同一ハンド内で同じプレイヤーが同じストリートで複数回手番を持つ場合(レイズ合戦など)、
    # ホールカード+コミュニティカードの組み合わせが同一になるため、計算結果をキャッシュして使い回す
    equity_cache: dict[tuple[tuple[str, ...], tuple[str, ...]], tuple[dict, int]] = {}
    hole_cards_by_player: dict[str, tuple[str, ...]] = {}

    processed = 0
    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(_HEADER)

        for line in fin:
            record = json.loads(line)
            record_type = record["type"]

            if record_type == "hand_start":
                hole_cards_by_player = {
                    p["player_id"]: tuple(p["hole_cards"]) for p in record["players"]
                }
                continue

            if record_type != "action":
                continue

            player_id = record["player_id"]
            hole_cards = hole_cards_by_player[player_id]
            community_cards = tuple(record["community_cards_before"])

            player_state = next(
                p for p in record["players_before"] if p["player_id"] == player_id
            )
            chips = player_state["chips"]
            pot = record["pot_before"]

            cache_key = (tuple(sorted(hole_cards)), community_cards)
            if cache_key not in equity_cache:
                equity_cache[cache_key] = estimate_hand_rank_probabilities(
                    hole_cards, community_cards, args.samples, rng
                )
            probabilities, n_samples = equity_cache[cache_key]

            writer.writerow([
                record["game_id"], record["hand_id"], record["phase"], player_id,
                record["action"], record["amount"],
                pot, record["current_bet_before"], chips, _spr(chips, pot),
                *[probabilities[r] for r in HAND_RANK_ORDER],
                n_samples,
            ])

            processed += 1
            if args.limit is not None and processed >= args.limit:
                break

    print(f"完了: {processed} 件のアクションを {args.output} に出力しました "
          f"(ユニークな局面数: {len(equity_cache)})")


if __name__ == "__main__":
    main()
