"""
実験ID: exp07
実験名: 選択組織図6特徴追加
著者: Codex

目的・仮説:
- all_features_v3へ、組織規模・構造・デジタル体制を表す指定6特徴だけを追加する。
- デジタル組織数と組織ノード数を同時に与え、規模調整済みの
  デジタル組織比率（デジタル組織数÷組織ノード数）の追加効果を検証する。

特徴量・前処理:
- all_features_v3の23列へ、DX変革組織有無、平均分岐数、第一階層組織数、
  デジタル組織数、組織ノード数、デジタル組織比率の6列だけを追加する。
- DX推進室完全一致フラグや、その他の組織図特徴は使用しない。
- 特徴量生成、業界統合、TF-IDF/SVD、補完、One-Hot Encodingは各fold内でfit。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。
- all_features_v3既存値nested OOF F1=0.7744および汎用15特徴を加えた
  exp02=0.7989との差を報告する。

出力:
- experiments/exp_tree_of_corp/results/exp07/feature_importance.png
- experiments/exp_tree_of_corp/results/exp07/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成入力特徴量数29、One-Hot後42/41/43/43/42列。全行欠損による除外なし。
- fold閾値: 0.345/0.155/0.175/0.340/0.270、最終閾値=0.2570。
- fold F1: 0.7536/0.8235/0.7619/0.7200/0.8333、平均=0.7785、
  標準偏差=0.0432、nested OOF F1=0.7792。
- 追加6特徴内のouter平均split importanceは、デジタル組織比率71.2、
  DX変革組織有無68.4、平均分岐数67.2、第一階層組織数41.0、
  組織ノード数28.0、デジタル組織数12.4。重要度は方向や因果を示さない。
- all_features_v3の0.7744に対して+0.0048。指定6特徴の組み合わせには小幅な
  追加効果があったが、汎用15特徴を加えたexp02の0.7989には届かなかった。
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

SPEC = ExperimentSpec("exp07", "v3＋選択組織図6特徴", 7)
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
