"""
実験ID: exp17
実験名: 名詞unigram TF-IDF → SVD 30
著者: Codex

目的・仮説:
- 名詞だけのunigram表現を30次元へ圧縮し、bigram単独・別チャネル結合との基準にする。

特徴量・前処理:
- 「今後のDX展望」からMeCab（UniDic）で名詞だけを抽出する。
- TfidfVectorizer(ngram_range=(1, 1), min_df=1)とTruncatedSVD(30次元)を
  各学習fold内だけでfitする。

モデル・評価:
- 固定LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer 5-fold、inner 4-fold。閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp17/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp17/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1、unigram語彙数: 2941, 2915, 2886, 2871, 2930
- SVD後特徴量数: 各fold 30、除外した全欠損特徴量: なし
- fold threshold: 0.125, 0.165, 0.210, 0.280, 0.280
- fold F1: 0.5000, 0.5333, 0.5366, 0.4632, 0.4857
- fold F1 mean ± std: 0.5038 ± 0.0281
- nested OOF F1: 0.5051、最終threshold: 0.2120
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

EXPERIMENT_ID = "exp17"
EXPERIMENT_NAME = "名詞unigram TF-IDF → SVD 30"
PARTS_OF_SPEECH = ("名詞",)
NGRAM_RANGE = (1, 1)
FEATURE_LABEL = "名詞unigram・SVD 30"
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
