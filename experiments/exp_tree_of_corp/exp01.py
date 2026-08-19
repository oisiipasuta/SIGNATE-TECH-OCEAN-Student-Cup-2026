"""
実験ID: exp01
実験名: 汎用組織図特徴単独
著者: Codex

目的・仮説:
- 組織図の構造、機能多様性、デジタル推進体制だけで購入フラグを予測できるかを
  測り、既存特徴と混ぜる前の単独信号を確認する。

特徴量・前処理:
- calc_features.tree_of_corpの汎用15特徴量のみを使用する。
- 横型・自由記述図の階層特徴は欠損とし、学習foldの中央値で補完する。
- DX推進室完全一致フラグは使用しない。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。

出力:
- experiments/exp_tree_of_corp/results/exp01/feature_importance.png
- experiments/exp_tree_of_corp/results/exp01/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成入力特徴量数15、変換後特徴量数15。全行欠損による除外なし。
- fold閾値: 0.125/0.130/0.140/0.165/0.135、最終閾値=0.1390。
- fold F1: 0.4028/0.4444/0.3429/0.4068/0.4348、平均=0.4063、
  標準偏差=0.0355、nested OOF F1=0.4059。
- 組織図単独にも信号はあるが、単独分類器としての性能は限定的だった。
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

SPEC = ExperimentSpec("exp01", "汎用組織図特徴単独", 1)
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
