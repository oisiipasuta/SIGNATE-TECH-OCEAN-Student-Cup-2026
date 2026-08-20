"""
実験ID: exp02
実験名: 上場種別・事業特徴の追加
著者: Codex

仮説:
- 現行20特徴量へ上場種別とBtoB/BtoC等の事業特徴を追加すると、企業属性に
  よる購入率の差を業界とは独立に捉えられる。

特徴量・前処理:
- exp01の20列に、上場種別と特徴を追加する。
- 特徴量生成、補完、One-Hot Encodingは各inner/outer学習foldだけでfitする。
- 業界・上場種別・特徴の水準は集約せず、変換後重要度へ個別に表示する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp02/feature_importance.png
- experiments/exp_ai/results/exp02/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで22。
- One-Hot後特徴量数: 40, 40, 42, 41, 41。全欠損除外: なし。
- fold threshold: 0.350, 0.225, 0.320, 0.245, 0.255。
- fold F1: 0.7164, 0.8421, 0.7123, 0.6977, 0.7568。
- fold F1 mean ± std: 0.7451 ± 0.0523。
- nested OOF F1: 0.7447、最終threshold: 0.2790。
- exp01比-0.0187であり、企業属性2列の単純追加は不採用候補。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score  # 静的検証用。F1計算本体は共通処理で実行する。


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.exp_ai._common import (  # noqa: E402
    MODEL_PARAMS,
    ExperimentSpec,
    configure_japanese_font,
    load_data,
    make_f1_figure,
    make_feature_importance_figure,
    print_summary,
    run_nested_cv,
)


# ==================================================
# 1. 実験設定
# ==================================================

SPEC = ExperimentSpec("exp02", "上場種別・事業特徴の追加", 2)
RESULT_DIR = BASE_DIR / "experiments" / "exp_ai" / "results" / SPEC.experiment_id


# ==================================================
# 2. データ読み込み
# ==================================================

def main() -> None:
    X, y = load_data()

    # ==================================================
    # 3. 特徴量前処理
    # 4. LightGBM・閾値選択
    # 5. ネステッド・クロスバリデーション
    # ==================================================
    result = run_nested_cv(X, y, SPEC)

    # ==================================================
    # 6. 実験結果・図の出力
    # ==================================================
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_importance_figure = make_feature_importance_figure(
        result.feature_importance, SPEC
    )
    f1_figure = make_f1_figure(result, SPEC)
    feature_importance_figure.savefig(
        RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_importance_figure)
    plt.close(f1_figure)
    print_summary(X, result, SPEC, font_name)


if __name__ == "__main__":
    main()
