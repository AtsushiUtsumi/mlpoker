---
name: selfplay-pipeline
description: mlpokerでpoker-domainを使ったセルフプレイのプレイログ(JSONL)生成と、SPR・役完成確率を付与したCSVへの変換を実行する。「シミュレーションを実行して」「プレイログを生成して」「学習データを作って」「CSVに変換して」といった依頼で使用する。
---

# セルフプレイ → 特徴量CSV 生成パイプライン

`mlpoker` リポジトリで、poker-domain を使った6人卓のランダム方策セルフプレイから
機械学習用データセットを作るまでの2段階パイプライン。

## 前提

- `.venv` が未作成なら `run.bat` (Windows) / `run.sh` (Linux) が自動で作成し、
  `requirements.txt` (poker-domain) をインストールする。
- 実行はリポジトリルート (`mlpoker/`) で行う。

## Step 1: プレイログ(JSONL)の生成

```
run.bat [--target-hands N] [--num-players 6] [--starting-chips 1000]
        [--small-blind 25] [--big-blind 50] [--ante 0]
        [--output logs/playlog.jsonl] [--seed N]
```

- 6人卓で5人がバスト(チップ0)するまでを1ゲームとし、目標ハンド数を超えるまでゲームを繰り返す。
- 既定値は `--target-hands 100000`。1ハンドあたり実測で約7ms程度(2000ハンドで約1秒未満)。
  10万ハンドで概ね数分〜10分程度。ハンド数が多い場合は `run_in_background: true` で実行し、
  出力行数 (`wc -l logs/playlog.jsonl`) をポーリングして完了を確認するとよい
  (バックグラウンド起動時に `&` を併用すると、シェルスクリプト自体は即座に終了したとみなされ、
  実際の処理が裏で終わる前に「完了」と誤通知されることがあるため、`&` は付けずに
  素のコマンドを `run_in_background: true` で渡すこと)。
- 出力は `logs/playlog.jsonl` (既定)。レコード種別は `game_start` / `hand_start` (ホールカード含む) /
  `action` (手番ごとのポット・スタック・アクション) / `hand_end` (役・配当・レーキ) / `game_end`。
- **既知の注意点**: poker-domain 側に「誰にもコールされなかった超過ベット/レイズをした本人が
  後でフォールドすると、その分のチップが消滅する」というサイドポット計算のバグがある
  (詳細は `poker_domain_bugreport_uncalled_bet_refund.md` を参照)。生成後のゲーム単位の
  チップ総量を検算すると、全体の1割前後のゲームで不一致が出るが、これは既知の事象であり
  スクリプト側の不具合ではない。

## Step 2: SPR・役完成確率のCSV変換

```
.venv/Scripts/python.exe -m selfplay.precompute_features \
    [--input logs/playlog.jsonl] [--output logs/features.csv] \
    [--samples 300] [--seed N] [--limit N]
```

- `logs/playlog.jsonl` の `action` レコード1件ごとに1行のCSVを出力する。
- 列: `game_id, hand_id, phase, player_id, action, amount, pot_before, current_bet_before,
  player_chips_before, spr, prob_high_card 〜 prob_royal_flush (10列, 合計1になる分布),
  equity_samples`
- `spr` = その時点のスタック ÷ ポット。
- 役完成確率は FLOP(残り2枚)/TURN(残り1枚)/RIVER(確定済み)は全パターンを厳密列挙、
  PRE_FLOP(残り5枚)のみ組み合わせ数が膨大なため `--samples` 件のモンテカルロ近似
  (既定300件)。同一ハンド内で同一プレイヤーが同一ストリートで複数回手番を持つ場合は
  結果をキャッシュして使い回す。
- 高速な役判定は `selfplay/equity.py` の `classify()` が担う(`poker_domain.HandEvaluator` と
  20万回のランダム突合せで完全一致を確認済み)。
- 実測: 56万行(playlog.jsonl 100,030ハンド分)で約10〜13分。大規模データの場合は
  `run_in_background: true` で実行し、想定行数 (= `action` レコード数 + 1) に達したかを
  ポーリングして完了を確認するとよい。

## 動作確認済みの実績値 (参考)

- 100,030ハンド / 6,739ゲーム → `playlog.jsonl` 約360MB, 774,804行
- 上記から `features.csv` 561,266行(約94MB)を生成、確率合計1.0の不整合ゼロを確認済み
