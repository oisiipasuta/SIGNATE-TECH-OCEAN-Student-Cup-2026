"""
実験ID: exp03
実験名: DX推進室完全一致アブレーション
著者: Codex

目的・仮説:
- exp02へ競技データで頻出するDX推進室完全一致フラグだけを追加し、汎用的な
  意味特徴を超える追加効果と、データ固有表現への依存を分離して確認する。

特徴量・前処理:
- all_features_v3の23列、汎用組織図15列、完全一致フラグ1列を使用する。
- 特徴量生成、業界統合、TF-IDF/SVD、補完、One-Hot Encodingは各fold内でfit。
- 「必ず含む」など生成注記らしい表現は使用しない。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。
- exp02およびall_features_v3既存値nested OOF F1=0.7744との差を報告する。

出力:
- experiments/exp_tree_of_corp/results/exp03/feature_importance.png
- experiments/exp_tree_of_corp/results/exp03/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成入力特徴量数39、One-Hot後52/51/53/53/52列。全行欠損による除外なし。
- fold閾値: 0.340/0.315/0.360/0.360/0.285、最終閾値=0.3320。
- fold F1: 0.7647/0.8767/0.7838/0.7297/0.8169、平均=0.7944、
  標準偏差=0.0499、nested OOF F1=0.7944。
- all_features_v3比+0.0200だがexp02比-0.0045。完全一致語の追加は採用しない。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.exp_tree_of_corp._common import (  # noqa: E402
    ExperimentSpec,
    configure_japanese_font,
    load_data,
    make_f1_figure,
    make_feature_importance_figure,
    print_association_profile,
    print_summary,
    run_nested_cv,
)


# ==================================================
# 1. 実験設定
# ==================================================

SPEC = ExperimentSpec("exp03", "DX推進室完全一致アブレーション", 3)
RESULT_DIR = BASE_DIR / "experiments" / "exp_tree_of_corp" / "results" / SPEC.experiment_id


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# 4. LightGBM・閾値選択
# 5. ネステッド・クロスバリデーション
# 6. 実験結果・図の出力
# ==================================================

def main() -> None:
    train, test, y = load_data()
    print_association_profile(train, test, y)
    result = run_nested_cv(train, y, SPEC)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(result.feature_importance, SPEC)
    f1_figure = make_f1_figure(result, SPEC)
    feature_figure.savefig(
        RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    print_summary(train, result, SPEC, font_name)


if __name__ == "__main__":
    main()
