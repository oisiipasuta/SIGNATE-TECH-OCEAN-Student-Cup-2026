"""
実験ID: exp05
実験名: LightGBM品詞アブレーション（名詞＋動詞＋形容詞・形状詞）
著者: Codex

目的・仮説:
- 名詞・動詞に形容詞・形状詞を加え、DXへの積極性、重要性、課題の大きさを表現する。
- exp04との差から形容詞・形状詞の追加価値を検証する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。名詞・動詞・形容詞・形状詞を抽出する。
- TF-IDFはmin_df=1、max_features=None。TruncatedSVDは30次元、random_state=42。
- TF-IDFとSVDは各CV foldの学習部分だけでfitする。

モデル・評価:
- exp_baseと同じLightGBM: n_estimators=500, learning_rate=0.03,
  max_depth=3, num_leaves=7, random_state=42。
- outer StratifiedKFold=5、inner=4。閾値はouter学習部分のinner OOF F1だけで選ぶ。
- 最終thresholdはouter foldで選ばれた閾値の平均。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp05/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp05/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-09実行）:
- 入力特徴量数: 1、SVD後特徴量数: 各fold 30
- TF-IDF語彙数: 4155, 4129, 4109, 4075, 4145
- 除外した全欠損特徴量: なし
- fold threshold: 0.185, 0.145, 0.115, 0.265, 0.255
- fold F1: 0.5625, 0.4921, 0.5094, 0.5455, 0.5169
- fold F1 mean ± std: 0.5253 ± 0.0254
- nested OOF F1: 0.5233、最終threshold: 0.1930
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

EXPERIMENT_ID = "exp05"
EXPERIMENT_NAME = "LightGBM品詞アブレーション（名詞＋動詞＋形容詞・形状詞）"
PARTS_OF_SPEECH = ("名詞", "動詞", "形容詞", "形状詞")
PARTS_LABEL = "名詞＋動詞＋形容詞・形状詞"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み
# ==================================================

def main() -> None:
    result = run_experiment(TRAIN_PATH, PARTS_OF_SPEECH)

    # ==================================================
    # 3. 特徴量前処理
    # 4. LightGBM
    # 5. ネステッド・クロスバリデーション
    # ==================================================
    # 上記は4実験で条件を共有するため _common.py で実行する。

    # ==================================================
    # 6. 実験結果・図の出力
    # ==================================================
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(
        result.feature_importance, EXPERIMENT_ID, PARTS_LABEL
    )
    f1_figure = make_f1_figure(
        result.fold_scores, result.nested_oof_f1, EXPERIMENT_ID, PARTS_LABEL
    )
    feature_figure.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    print_summary(result, EXPERIMENT_ID, EXPERIMENT_NAME, PARTS_OF_SPEECH, font_name)


if __name__ == "__main__":
    main()
