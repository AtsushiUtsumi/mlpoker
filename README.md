# mlpoker

[poker-domain](https://github.com/AtsushiUtsumi/poker-domain) を使ってポーカーの自己対戦プレイログを生成し、そこから機械学習モデルを作成して、実際にポーカーサイト上でボット同士に対戦させるまでの一連のパイプラインです。

## 全体の流れ

1. **プレイログ生成** (`selfplay.simulate`)
   6人卓・ランダム方策で自己対戦を行い、5人がバストするまでを1ゲームとして繰り返し、目的のハンド数に達するまでアクションログを JSONL 形式 (`logs/playlog.jsonl`) に出力します。

2. **特徴量計算** (`selfplay.precompute_features`)
   プレイログの各アクション局面について、SPR (Stack to Pot Ratio) と、その時点のホールカード・コミュニティカードからリバーまでに各役 (ハイカード〜ロイヤルフラッシュ) が成立する確率をモンテカルロ/全列挙で計算し、学習用CSV (`logs/features.csv`) に変換します。

3. **モデル学習** (`selfplay.train_model`)
   `features.csv` を読み込み、
   - アクション分類モデル (fold/check/call/bet/raise を予測する RandomForestClassifier)
   - ベット/レイズ額(ポット比)の回帰モデル (RandomForestRegressor)

   の2つを学習し、`models/` ディレクトリに保存します (`action_model.joblib`, `amount_model.joblib`, `metadata.json`)。

4. **ボット対戦** (`selfplay.bot_client`)
   学習済みモデルを読み込み、ポーカーサイト (Mullhouse) のAPIに接続してテーブルを作成/参加し、6人卓でボット同士に対戦させます。局面ごとに学習時と同じ特徴量(SPR・役完成確率)をその場で計算し、モデルの予測に基づいて行動を決定します。

## セットアップ・実行方法

Python の venv (`.venv`) を使用します。初回実行時に依存パッケージが自動でインストールされます。

```bash
# Linux
./run.sh [任意の selfplay.simulate 用引数]
```

```bat
:: Windows
run.bat [任意の selfplay.simulate 用引数]
```

上記スクリプトは `python -m selfplay.simulate` を実行します。他のステップ (特徴量計算・学習・ボット対戦) は venv を有効化した上でそれぞれ以下のように実行します。

```bash
python -m selfplay.precompute_features --input logs/playlog.jsonl --output logs/features.csv
python -m selfplay.train_model --input logs/features.csv --output-dir models
python -m selfplay.bot_client --host <ポーカーサイトのホスト>
```

各スクリプトの主なオプションは `--help` で確認できます。

## ディレクトリ構成

- `selfplay/simulate.py` — 自己対戦プレイログ生成
- `selfplay/precompute_features.py` — SPR・役完成確率の特徴量計算 (JSONL → CSV)
- `selfplay/train_model.py` — アクション模倣モデルの学習
- `selfplay/bot_client.py` — 学習済みモデルでポーカーサイト上のボットを対戦させるクライアント
- `selfplay/equity.py` — 役判定・役成立確率の計算ロジック
- `selfplay/policy.py` — プレイログ生成用のランダム方策
- `selfplay/cards.py`, `selfplay/logger.py` — カード表現・JSONL書き出しのユーティリティ
- `logs/` — 生成されたプレイログ・特徴量CSVの出力先
- `models/` — 学習済みモデルの出力先
