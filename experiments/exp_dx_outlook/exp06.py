"""
実験ID: exp06
実験名: 品詞アブレーション（名詞＋動詞＋形容詞・形状詞＋副詞）
著者: Codex

目的・仮説:
- exp05の品詞に副詞を加え、「積極的に」「段階的に」「早急に」など行動の
  強さ・速度感を表現する。exp05との差から副詞の追加価値を判断する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。名詞・動詞・形容詞・形状詞・副詞を抽出する。
- TF-IDFはmin_df=1、max_features=None。SVDは30次元、random_state=42。
- 各fold内でTF-IDF、SVD、標準化をfitし、exp03～exp05と条件を揃える。

モデル・評価:
- LogisticRegression(solver="saga", max_iter=5000)。
- L2（C=0.001～10）、L1（C=0.01～10）、Elastic Net
  （C=0.1～10、l1_ratio=0.25/0.50/0.75）をinner CVで比較する。
- outer 5-fold、inner 4-fold、random_state=42。正則化と閾値はinner OOF F1
  のみで同時選択し、outer validationは選択に使用しない。

出力:
- experiments/exp_dx_outlook/results/exp06/feature_importance.png
- experiments/exp_dx_outlook/results/exp06/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-09実行）:
- 入力特徴量数: 1、SVD後特徴量数: 各fold 30
- TF-IDF語彙数: 4239, 4210, 4193, 4157, 4233
- 除外した全欠損特徴量: なし
- fold選択正則化: L1/0.1, EN(0.50)/0.1, L1/0.1, L1/0.1, L1/0.1
- fold threshold: 0.330, 0.280, 0.285, 0.295, 0.305
- fold F1: 0.6400, 0.5321, 0.6133, 0.5400, 0.5750
- fold F1 mean ± std: 0.5801 ± 0.0415
- nested OOF F1: 0.5740、最終threshold: 0.2990
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

EXPERIMENT_ID = "exp06"
EXPERIMENT_NAME = "品詞アブレーション（名詞＋動詞＋形容詞・形状詞＋副詞）"
PARTS_OF_SPEECH = ("名詞", "動詞", "形容詞", "形状詞", "副詞")
PARTS_LABEL = "名詞＋動詞＋形容詞・形状詞＋副詞"
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
