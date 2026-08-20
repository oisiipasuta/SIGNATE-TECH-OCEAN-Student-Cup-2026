"""
実験ID: exp10
実験名: DX展望SVD 20成分
著者: Codex

仮説:
- 名詞・動詞TF-IDFのSVD累積寄与率を先に確認し、情報保持と変数数のバランスが
  良い成分数なら、v2のSVD第2成分だけよりDX展望を有効に利用できる。

成分数の事前診断:
- outer学習foldだけで30成分をfitした平均累積寄与率は、10成分9.3193%、
  20成分13.6649%、30成分17.3769%（各foldの範囲17.3193～17.4636%）。
- 20成分で先頭30成分が説明する分散の78.64%を保持するため、20成分を採用した。

特徴量・前処理:
- AllFeaturesV1Transformerの19列 + DX展望SVD 20列。
- TF-IDF、SVD、補完、One-Hot Encodingは各inner/outer学習foldだけでfitする。
- SVDは名詞・動詞、単語unigram、random_state=42。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp10/feature_importance.png
- experiments/exp_ai/results/exp10/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで39。
- One-Hot後特徴量数: 52, 51, 53, 53, 52。全欠損除外: なし。
- fold threshold: 0.220, 0.220, 0.320, 0.260, 0.215。
- fold F1: 0.7632, 0.7792, 0.7027, 0.7160, 0.7250。
- fold F1 mean ± std: 0.7372 ± 0.0291。
- nested OOF F1: 0.7371、最終threshold: 0.2470。
- exp01比: -0.0263。20成分への拡張は改善しなかった。
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


SPEC = ExperimentSpec("exp10", "DX展望SVD 20成分", 10)
RESULT_DIR = BASE_DIR / "experiments" / "exp_ai" / "results" / SPEC.experiment_id


def main() -> None:
    X, y = load_data()
    result = run_nested_cv(X, y, SPEC)

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
