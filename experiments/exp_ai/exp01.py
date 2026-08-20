"""
実験ID: exp01
実験名: 現行 all_features_v2 ベースライン再現
著者: Codex

仮説:
- all_features_v1の19列とDX展望SVD第2成分を使う現行構成を再現し、以後の
  累積特徴量アブレーションの比較基準にする。

特徴量・前処理:
- AllFeaturesV2Transformerを各inner/outer学習fold内でfitする。
- 数値列は学習foldの中央値で補完する。
- 業界は学習foldだけで補完・One-Hot Encodingし、未知カテゴリは無視する。
- 学習foldで全欠損の列だけを除外する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp01/feature_importance.png
- experiments/exp_ai/results/exp01/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで20。
- One-Hot後特徴量数: 33, 32, 34, 34, 33。全欠損除外: なし。
- fold threshold: 0.355, 0.195, 0.255, 0.285, 0.325。
- fold F1: 0.7164, 0.8205, 0.7632, 0.7342, 0.7778。
- fold F1 mean ± std: 0.7624 ± 0.0361。
- nested OOF F1: 0.7634、最終threshold: 0.2830。
- 既存exp_mix/exp02の結果を完全再現した。
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

SPEC = ExperimentSpec("exp01", "現行 all_features_v2 ベースライン再現", 1)
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
