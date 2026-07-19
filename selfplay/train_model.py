import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PHASES = ["PRE_FLOP", "FLOP", "TURN", "RIVER"]
PROB_COLUMNS = [
    "prob_high_card", "prob_one_pair", "prob_two_pair", "prob_three_of_a_kind",
    "prob_straight", "prob_flush", "prob_full_house", "prob_four_of_a_kind",
    "prob_straight_flush", "prob_royal_flush",
]
BASE_FEATURE_COLUMNS = ["pot_before", "current_bet_before", "player_chips_before", "spr", *PROB_COLUMNS]
ACTION_CLASSES = ["fold", "check", "call", "bet", "raise"]


def _phase_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {f"phase_{p}": (df["phase"] == p).astype(int) for p in PHASES},
        index=df.index,
    )


def build_action_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df[BASE_FEATURE_COLUMNS], _phase_one_hot(df)], axis=1)


def build_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    features = build_action_features(df)
    features = features.copy()
    features["is_raise"] = (df["action"] == "raise").astype(int)
    return features


def main() -> None:
    parser = argparse.ArgumentParser(
        description="features.csv からアクション模倣モデル(行動分類+ベット/レイズ額の回帰)を学習する"
    )
    parser.add_argument("--input", type=str, default="logs/features.csv")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df["spr"] = pd.to_numeric(df["spr"], errors="coerce").fillna(0.0)

    # ── アクション分類モデル (fold/check/call/bet/raise) ──
    X = build_action_features(df)
    y = df["action"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    action_model = RandomForestClassifier(
        n_estimators=200, max_depth=16, min_samples_leaf=5,
        random_state=args.seed, n_jobs=-1,
    )
    action_model.fit(X_train, y_train)
    action_report = classification_report(y_test, action_model.predict(X_test), zero_division=0)
    print("=== アクション分類モデル: テストデータでの評価 ===")
    print(action_report)

    # ── ベット/レイズ額(ポット比)の回帰モデル ──
    bet_raise_df = df[df["action"].isin(["bet", "raise"]) & (df["pot_before"] > 0)].copy()
    bet_raise_df["amount_ratio"] = bet_raise_df["amount"] / bet_raise_df["pot_before"]

    Xa = build_amount_features(bet_raise_df)
    ya = bet_raise_df["amount_ratio"]
    Xa_train, Xa_test, ya_train, ya_test = train_test_split(
        Xa, ya, test_size=args.test_size, random_state=args.seed
    )

    amount_model = RandomForestRegressor(
        n_estimators=200, max_depth=16, min_samples_leaf=5,
        random_state=args.seed, n_jobs=-1,
    )
    amount_model.fit(Xa_train, ya_train)
    pred = amount_model.predict(Xa_test)
    print("=== ベット/レイズ額(ポット比)回帰モデル: テストデータでの評価 ===")
    print(f"MAE: {mean_absolute_error(ya_test, pred):.4f}  R^2: {r2_score(ya_test, pred):.4f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(action_model, output_dir / "action_model.joblib")
    joblib.dump(amount_model, output_dir / "amount_model.joblib")

    metadata = {
        "phases": PHASES,
        "prob_columns": PROB_COLUMNS,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "action_feature_columns": list(build_action_features(df.head(1)).columns),
        "amount_feature_columns": list(build_amount_features(bet_raise_df.head(1)).columns),
        "action_classes": list(action_model.classes_),
        "trained_rows": int(len(df)),
        "bet_raise_rows": int(len(bet_raise_df)),
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n完了: モデルを {output_dir}/ に保存しました "
          f"(action_model.joblib, amount_model.joblib, metadata.json)")


if __name__ == "__main__":
    main()
