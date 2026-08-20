"""
実験ID: exp06
実験名: 企業概要・組織図テキストの統合
著者: Codex

仮説:
- exp05へ企業概要と組織図の文字特徴量を追加すると、既存の財務・アンケート・
  DX展望とは異なる事業内容と組織能力のシグナルを利用できる。

特徴量・前処理:
- exp05の特徴量へ企業概要・組織図を追加する。
- 各列を文字2～5-gram TF-IDFへ変換し、それぞれ10次元へSVDする。
- 会社名は事前スクリーニング性能が低いため使用しない。
- TF-IDF、SVD、補完、One-Hot Encodingは各inner/outer学習fold内でfitする。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp06/feature_importance.png
- experiments/exp_ai/results/exp06/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで70。
- One-Hot後特徴量数: 88, 88, 90, 89, 89。全欠損除外: なし。
- fold threshold: 0.350, 0.200, 0.200, 0.135, 0.370。
- fold F1: 0.7692, 0.8000, 0.7632, 0.7174, 0.7273。
- fold F1 mean ± std: 0.7554 ± 0.0299。
- nested OOF F1: 0.7540、最終threshold: 0.2510。
- exp01比-0.0094。企業概要・組織図SVDの累積追加は改善しなかった。
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

SPEC = ExperimentSpec("exp06", "企業概要・組織図テキストの統合", 6)
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
