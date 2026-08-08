"""
実験ID: exp09
実験名: SVD全30成分の再現ベースライン
著者: Codex

目的・仮説:
- exp07・exp08と同じ実装経路でSVD全30成分を評価し、比較用ベースラインを再現する。
- exp04と同じ結果になることを確認し、列選択実験の差だけを公平に比較する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。MeCab（UniDic）で名詞・動詞を抽出する。
- 各foldの学習部分だけでTF-IDFと30次元TruncatedSVDをfitする。
- 変換後の30列をすべてLightGBMへ入力する。

モデル・評価:
- exp_baseと同じLightGBM: n_estimators=500, learning_rate=0.03,
  max_depth=3, num_leaves=7, random_state=42。
- outer StratifiedKFold=5、inner=4。閾値はouter学習部分のinner OOF F1だけで選ぶ。
- 最終thresholdはouter foldで選ばれた閾値の平均。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp09/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp09/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-09実行）:
- 入力特徴量数: 1、SVD生成数: 各fold 30、選択後特徴量数: 各fold 30
- TF-IDF語彙数: 3912, 3887, 3866, 3835, 3909
- 除外した全欠損特徴量: なし
- fold threshold: 0.235, 0.235, 0.175, 0.455, 0.225
- fold F1: 0.4889, 0.6078, 0.5814, 0.4691, 0.5227
- fold F1 mean ± std: 0.5340 ± 0.0530
- nested OOF F1: 0.5369、最終threshold: 0.2650
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.dx_outlook import get_dx_outlook_feature_columns
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

EXPERIMENT_ID = "exp09"
EXPERIMENT_NAME = "SVD全30成分の再現ベースライン"
PARTS_OF_SPEECH = ("名詞", "動詞")
FEATURE_COLUMNS = tuple(get_dx_outlook_feature_columns(30))
FEATURE_LABEL = "名詞＋動詞・SVD全30成分"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み
# ==================================================

def main() -> None:
    result = run_experiment(TRAIN_PATH, PARTS_OF_SPEECH, FEATURE_COLUMNS)

    # ==================================================
    # 3. 特徴量前処理・固定列選択
    # 4. LightGBM
    # 5. ネステッド・クロスバリデーション
    # ==================================================
    # 上記は比較条件を共有するため _common.py で実行する。

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
    print_summary(result, EXPERIMENT_ID, EXPERIMENT_NAME, PARTS_OF_SPEECH, font_name)


if __name__ == "__main__":
    main()
