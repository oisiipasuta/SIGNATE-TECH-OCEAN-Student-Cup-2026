"""
実験ID: exp15
実験名: n-gram品詞累積比較（名詞＋動詞＋形容詞・形状詞）
著者: Codex

目的・仮説:
- exp14に形容詞・形状詞を追加し、対象・行動に状態や評価を加えたn-gramの効果を調べる。
- exp13～exp16では抽出品詞だけを変える。

特徴量・前処理:
- 入力は「今後のDX展望」1列。名詞・動詞・形容詞・形状詞を抽出する。
- 抽出後の語列へTfidfVectorizer(ngram_range=(1, 2), min_df=1)を適用する。
- TF-IDFと30次元TruncatedSVDは各学習fold内だけでfitする。

モデル・評価:
- LightGBMのパラメータは固定: n_estimators=500, learning_rate=0.03,
  max_depth=3, num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner=4。閾値は各outer学習部分のinner OOF F1
  だけで選択し、outer validationは最終評価まで使用しない。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp15/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp15/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1、TF-IDF語彙数: 57154, 56841, 56897, 56659, 56811
- SVD後特徴量数: 各fold 30、除外した全欠損特徴量: なし
- fold threshold: 0.215, 0.250, 0.395, 0.450, 0.260
- fold F1: 0.5631, 0.4956, 0.5079, 0.5745, 0.5301
- fold F1 mean ± std: 0.5342 ± 0.0305
- nested OOF F1: 0.5351、最終threshold: 0.3140
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.exp_dx_outlook_lightgbm._common import (
    configure_japanese_font,
    make_f1_figure,
    make_feature_importance_figure,
    print_summary,
    run_experiment,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp15"
EXPERIMENT_NAME = "n-gram品詞累積比較（名詞＋動詞＋形容詞・形状詞）"
PARTS_OF_SPEECH = ("名詞", "動詞", "形容詞", "形状詞")
NGRAM_RANGE = (1, 2)
FEATURE_LABEL = "名詞＋動詞＋形容詞・形状詞・word unigram＋bigram・SVD 30"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# 4. LightGBM
# 5. ネステッド・クロスバリデーション
# ==================================================

def main() -> None:
    result = run_experiment(TRAIN_PATH, PARTS_OF_SPEECH, ngram_range=NGRAM_RANGE)

    # ==================================================
    # 6. 実験結果・図の出力
    # ==================================================
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(
        result.feature_importance, EXPERIMENT_ID, FEATURE_LABEL
    )
    f1_figure = make_f1_figure(
        result.fold_scores, result.nested_oof_f1, EXPERIMENT_ID, FEATURE_LABEL
    )
    feature_figure.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    print(f"ngram_range: {NGRAM_RANGE}")
    print_summary(result, EXPERIMENT_ID, EXPERIMENT_NAME, PARTS_OF_SPEECH, font_name)


if __name__ == "__main__":
    main()
