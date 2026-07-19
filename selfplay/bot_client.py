"""機械学習モデル(selfplay.train_model で学習)を使って mullhouse のポーカーテーブルに
接続し、6人卓でボット同士に対戦させるクライアント。

学習データ(selfplay.precompute_features の出力)と同じ特徴量(SPR・役完成確率)を
その場で計算し、行動分類モデル+ベット/レイズ額回帰モデルで手番を決定する。
HTTP通信は mullhouse/bots/table_bot.py に合わせ、標準ライブラリのみで行う。

Usage:
    python -m selfplay.bot_client [--base-url http://localhost/api/poker]
                                   [--site-url http://localhost]
                                   [--model-dir models] [--num-bots 6]
                                   [--initial-chips 1000] [--small-blind 25]
                                   [--big-blind 50] [--ante 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import joblib

from selfplay.equity import HAND_RANK_ORDER, estimate_hand_rank_probabilities

PHASES = ["PRE_FLOP", "FLOP", "TURN", "RIVER"]


def request(method: str, url: str, body: dict | None = None) -> dict:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} -> {e.code}: {detail}") from e


class ModelPolicy:
    def __init__(self, model_dir: str, samples: int, seed: int | None) -> None:
        model_dir_path = Path(model_dir)
        self.action_model = joblib.load(model_dir_path / "action_model.joblib")
        self.amount_model = joblib.load(model_dir_path / "amount_model.joblib")
        self.metadata = json.loads((model_dir_path / "metadata.json").read_text(encoding="utf-8"))
        self.samples = samples
        self.rng = random.Random(seed)
        self.action_classes = list(self.action_model.classes_)

    def _feature_row(self, state: dict, me: dict) -> dict:
        hole = tuple(me.get("hole_cards") or ())
        community = tuple(state["community_cards"])
        pot = state["pot"]
        chips = me["chips"]

        if hole:
            probs, _ = estimate_hand_rank_probabilities(hole, community, self.samples, self.rng)
        else:
            probs = {r: 0.0 for r in HAND_RANK_ORDER}

        row = {
            "pot_before": pot,
            "current_bet_before": state["current_bet"],
            "player_chips_before": chips,
            "spr": (chips / pot) if pot > 0 else 0.0,
        }
        for r in HAND_RANK_ORDER:
            row[f"prob_{r.name.lower()}"] = probs[r]
        for p in PHASES:
            row[f"phase_{p}"] = 1 if state["phase"] == p else 0
        return row

    @staticmethod
    def _clamp_amount(state: dict, me: dict, action: str, raw_amount: float) -> int | None:
        max_amount = me["current_bet"] + me["chips"]
        if action == "bet":
            min_amount = state["big_blind"]
        else:
            min_amount = max(state["big_blind"], state["current_bet"] * 2)
        if max_amount < min_amount:
            # 手持ち全額を賭けても最小額に届かない(ショートスタック)。
            # bet はオールインなら合法だが、raise は不足額オールインに対応していないため断念する。
            if action == "bet" and max_amount > 0:
                return max_amount
            return None
        return int(min(max(round(raw_amount), min_amount), max_amount))

    def decide(self, state: dict, me: dict, valid_actions: list[str]) -> tuple[str, int | None]:
        row = self._feature_row(state, me)
        x_action = [[row[c] for c in self.metadata["action_feature_columns"]]]
        proba = self.action_model.predict_proba(x_action)[0]

        masked = {
            a: proba[self.action_classes.index(a)]
            for a in valid_actions if a in self.action_classes
        }
        if not masked or max(masked.values()) <= 0:
            chosen = "fold" if "fold" in valid_actions else valid_actions[0]
        else:
            chosen = max(masked, key=masked.get)

        if chosen not in ("bet", "raise"):
            return chosen, None

        row_amount = dict(row)
        row_amount["is_raise"] = 1 if chosen == "raise" else 0
        x_amount = [[row_amount[c] for c in self.metadata["amount_feature_columns"]]]
        ratio = float(self.amount_model.predict(x_amount)[0])
        raw_amount = ratio * max(state["pot"], 1)

        amount = self._clamp_amount(state, me, chosen, raw_amount)
        if amount is None:
            fallback = "call" if "call" in valid_actions else ("check" if "check" in valid_actions else "fold")
            return fallback, None
        return chosen, amount


class BotPlayer:
    def __init__(self, name: str, player_id: str, token: str) -> None:
        self.name = name
        self.player_id = player_id
        self.token = token
        self.last_phase: str | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="localhost", help="対象ホスト")
    parser.add_argument("--base-url", default=None, help="省略時は http://{host}/api/poker")
    parser.add_argument("--site-url", default=None, help="省略時は http://{host}")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--table-name", default="MLボット卓")
    parser.add_argument("--join-table-id", default=None,
                         help="指定した場合、新規に卓を作成せず既存の卓(table_id)にボットを参加させる")
    parser.add_argument("--max-players", type=int, default=6)
    parser.add_argument("--num-bots", type=int, default=6, help="ボット同士のみで対戦する場合は max-players と同数にする")
    # 学習データ(logs/features.csv)と同じブラインド・初期チップに揃えるのが既定値
    # (行動分類・回帰モデルの特徴量はポット/ベット額の絶対チップ数なので、スケールが
    # 学習時と大きく異なると精度が落ちる点に注意)
    parser.add_argument("--initial-chips", type=int, default=1000)
    parser.add_argument("--small-blind", type=int, default=25)
    parser.add_argument("--big-blind", type=int, default=50)
    parser.add_argument("--ante", type=int, default=0)
    parser.add_argument("--samples", type=int, default=300, help="PRE_FLOP時のモンテカルロ・サンプル数")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--think-seconds", type=float, default=0.5)
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}/api/poker"
    site_url = args.site_url or f"http://{args.host}"

    policy = ModelPolicy(args.model_dir, args.samples, args.seed)

    if args.join_table_id:
        table = request("GET", f"{base_url}/tables/{args.join_table_id}")
        table_id = table["table_id"]
        buy_in = table.get("initial_chips") or args.initial_chips
        max_players = table["max_players"]
        print(f"既存の卓に参加します: '{table['name']}' (table_id={table_id})")
        print(f"  既存着席人数={table['seated']}/{max_players} 買い入れ額={buy_in}")
    else:
        table = request(
            "POST",
            f"{base_url}/tables",
            {
                "name": args.table_name,
                "max_players": args.max_players,
                "level_schedule": [[args.small_blind, args.big_blind, args.ante]],
                "require_full_table": True,
                "initial_chips": args.initial_chips,
                # 5人がバストするまでプレイさせるため、リバイは禁止(バストしたら脱落のまま)
                "allow_rebuy": False,
            },
        )
        table_id = table["table_id"]
        buy_in = args.initial_chips
        max_players = args.max_players
        print(f"卓を作成しました: '{args.table_name}' (table_id={table_id})")
        print(f"  初期チップ={args.initial_chips} SB/BB/ante={args.small_blind}/{args.big_blind}/{args.ante}")

    bots: list[BotPlayer] = []
    for i in range(1, args.num_bots + 1):
        name = f"MLBot{i}"
        join = request(
            "POST",
            f"{base_url}/tables/{table_id}/join",
            {"display_name": name, "buy_in": buy_in},
        )
        bots.append(BotPlayer(name, join["player_id"], join["token"]))
        print(f"'{name}' として着席しました (player_id={join['player_id']})")

    print(f"{len(bots)}/{max_players}人のMLボットが着席しました。")
    print(f"観戦する場合はブラウザで開いてください: {site_url}/poker/{table_id}")
    print("対戦を開始します... (Ctrl+Cで終了)")

    while True:
        any_active = False
        for bot in bots:
            payload = request(
                "GET",
                f"{base_url}/tables/{table_id}/state?player_id={bot.player_id}&token={bot.token}",
            )
            state = payload["state"]
            if state["phase"] != bot.last_phase:
                seats = [(p["player_id"][:6], p["chips"]) for p in state["players"]]
                print(f"[{bot.name}] phase={state['phase']} pot={state['pot']} players={seats}")
                bot.last_phase = state["phase"]

            if state["status"] == "CLOSED":
                continue
            any_active = True

            waiting_for = payload["waiting_for"]
            if waiting_for is not None and waiting_for["player_id"] == bot.player_id:
                me = next(p for p in state["players"] if p["player_id"] == bot.player_id)
                time.sleep(args.think_seconds)
                action, amount = policy.decide(state, me, waiting_for["valid_actions"])
                print(f"[{bot.name}] action={action}" + (f" amount={amount}" if amount is not None else ""))
                request(
                    "POST",
                    f"{base_url}/tables/{table_id}/action",
                    {"player_id": bot.player_id, "token": bot.token, "action": action, "amount": amount},
                )

        if not any_active:
            final = request("GET", f"{base_url}/tables/{table_id}")
            print("卓が終了しました。")
            print(json.dumps(final, ensure_ascii=False, indent=2))
            break

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました")
        sys.exit(0)
