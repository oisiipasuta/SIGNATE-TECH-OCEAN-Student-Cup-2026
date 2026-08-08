"""
実験ID: exp05
実験名: 品詞アブレーション（名詞＋動詞＋形容詞・形状詞）
著者: Codex

目的・仮説:
- 名詞・動詞に形容詞と形状詞を加え、DXへの積極性、慎重さ、重要性や
  課題の大きさを表現する。購入予測で副詞に次ぐ追加価値があるか検証する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。名詞・動詞・形容詞・形状詞を抽出する。
- TF-IDFはmin_df=1、max_features=None。SVDは30次元、random_state=42。
- 各fold内でTF-IDF、SVD、標準化をfitし、他の品詞実験と条件を揃える。

モデル・評価:
- LogisticRegression(solver="saga", max_iter=5000)。
- L2（C=0.001～10）、L1（C=0.01～10）、Elastic Net
  （C=0.1～10、l1_ratio=0.25/0.50/0.75）をinner CVで比較する。
- outer 5-fold、inner 4-fold、random_state=42。正則化と閾値はinner OOF F1
  のみで同時選択し、最終thresholdはouter fold閾値の平均とする。

出力:
- experiments/exp_dx_outlook/results/exp05/feature_importance.png
- experiments/exp_dx_outlook/results/exp05/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-09実行）:
- 入力特徴量数: 1、SVD後特徴量数: 各fold 30
- TF-IDF語彙数: 4155, 4129, 4109, 4075, 4145
- 除外した全欠損特徴量: なし
- fold選択正則化: L1/0.1, EN(0.50)/0.1, EN(0.75)/0.1, EN(0.50)/0.1, L1/0.1
- fold threshold: 0.320, 0.270, 0.270, 0.290, 0.300
- fold F1: 0.6234, 0.5321, 0.5974, 0.5000, 0.5610
- fold F1 mean ± std: 0.5628 ± 0.0442
- nested OOF F1: 0.5578、最終threshold: 0.2900
"""

from __future__ import annotations

import sys
from pathlib import Path

from sklearn.metrics import f1_score

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.exp_dx_outlook._pos_ablation_common import (
    configure_japanese_font,
    make_f1_figure,
    make_feature_importance_figure,
    print_summary,
    run_experiment,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp05"
EXPERIMENT_NAME = "品詞アブレーション（名詞＋動詞＋形容詞・形状詞）"
PARTS_OF_SPEECH = ("名詞", "動詞", "形容詞", "形状詞")
PARTS_LABEL = "名詞＋動詞＋形容詞・形状詞"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み
# ==================================================

def main() -> None:
    result = run_experiment(TRAIN_PATH, PARTS_OF_SPEECH)

    # ==================================================
    # 3. 特徴量前処理
    # 4. 正則化付きロジスティック回帰
    # 5. ネステッド・クロスバリデーション
    # ==================================================
    nested_oof_f1 = f1_score(result.y, result.oof_predictions)
    feature_importance = result.feature_importance

    # ==================================================
    # 6. 実験結果・図の出力
    # ==================================================
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(
        feature_importance, EXPERIMENT_ID, PARTS_LABEL
    )
    f1_figure = make_f1_figure(
        result.fold_scores, nested_oof_f1, EXPERIMENT_ID, PARTS_LABEL
    )
    feature_figure.savefig(
        RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    print_summary(
        result,
        EXPERIMENT_ID,
        EXPERIMENT_NAME,
        PARTS_OF_SPEECH,
        font_name,
    )


if __name__ == "__main__":
    main()
