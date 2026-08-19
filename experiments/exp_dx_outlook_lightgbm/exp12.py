"""
実験ID: exp12（Stage 4 の Exp09 相当）
実験名: word unigram + bigram（SVD 30・内容語5品詞）
著者: Codex

目的・仮説:
- unigram に bigram を加え、「研修を強化」のような内容語の並びを表現すると
  word unigram のみの exp11 より分類性能が改善するか検証する。
- exp11 との差は ngram_range だけに限定する。

特徴量・前処理:
- 入力特徴量は「今後のDX展望」1列。
- MeCab（UniDic）で名詞・動詞・形容詞・形状詞・副詞を抽出する。
- TfidfVectorizer の ngram_range=(1, 2) とし、各学習fold内だけでfitする。
- TF-IDFを各学習fold内だけで30次元TruncatedSVDへ変換する。
- 助詞は対象外なので、bigram は抽出後の内容語列における隣接2語を表す。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42、importance_type=split。
- outer StratifiedKFold=5、inner=4。各outer foldの閾値は、その学習部分の
  inner OOF F1だけで選ぶ。最終thresholdはouter fold閾値の平均。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp12/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp12/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1、TF-IDF語彙数: 58496, 58197, 58239, 57987, 58156
- SVD生成・選択後特徴量数: 各fold 30、除外した全欠損特徴量: なし
- fold threshold: 0.335, 0.190, 0.235, 0.315, 0.285
- fold F1: 0.5977, 0.4957, 0.5479, 0.4660, 0.5714
- fold F1 mean ± std: 0.5358 ± 0.0484
- nested OOF F1: 0.5302、最終threshold: 0.2720
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

EXPERIMENT_ID = "exp12"
EXPERIMENT_NAME = "word unigram + bigram（SVD 30・内容語5品詞）"
PARTS_OF_SPEECH = ("名詞", "動詞", "形容詞", "形状詞", "副詞")
NGRAM_RANGE = (1, 2)
FEATURE_LABEL = "word unigram + bigram・内容語5品詞・SVD 30"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# 4. LightGBM
# 5. ネステッド・クロスバリデーション
# ==================================================

def main() -> None:
    result = run_experiment(
        TRAIN_PATH,
        PARTS_OF_SPEECH,
        ngram_range=NGRAM_RANGE,
    )

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
