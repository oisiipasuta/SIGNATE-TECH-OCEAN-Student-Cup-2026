"""
実験ID: exp12
実験名: DX展望SVD 第1成分のみ
著者: Codex

仮説:
- DX展望SVDの第1成分だけを使い、現行第2成分だけのexp01と比較することで、
  文書全体の共通傾向を表す第1成分に単独の予測力があるか確認する。

特徴量・前処理:
- AllFeaturesV1Transformerの19列 + DX展望SVD第1成分の1列。
- 各学習foldで30成分SVDをfitし、baselineと同じ軸推定条件で第1成分だけを選ぶ。
- TF-IDF、SVD、補完、One-Hot Encodingは各inner/outer学習foldだけでfitする。
- SVDは名詞・動詞、単語unigram、random_state=42。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp12/feature_importance.png
- experiments/exp_ai/results/exp12/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで20。
- One-Hot後特徴量数: 33, 32, 34, 34, 33。全欠損除外: なし。
- fold threshold: 0.315, 0.240, 0.295, 0.415, 0.330。
- fold F1: 0.6216, 0.7342, 0.6753, 0.6364, 0.6857。
- fold F1 mean ± std: 0.6706 ± 0.0397。
- nested OOF F1: 0.6721、最終threshold: 0.3190。
- exp01（第2成分のみ）比: -0.0913。第1成分単独は大幅に悪化した。
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


SPEC = ExperimentSpec("exp12", "DX展望SVD 第1成分のみ", 12)
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
