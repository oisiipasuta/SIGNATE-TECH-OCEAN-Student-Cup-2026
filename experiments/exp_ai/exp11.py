"""
実験ID: exp11
実験名: DX展望SVD 5成分
著者: Codex

仮説:
- SVD 20成分はノイズとなった可能性がある。第2～5成分はexp10で実際に利用されて
  いたため、5成分までに絞ればDX展望の主要情報を残しながら過学習を抑えられる。

成分数の事前診断:
- outer学習foldでの平均累積寄与率は3成分4.8577%、5成分6.5372%。
- 5成分は3成分より累積説明分散が1.6795ポイント多く、exp10でも第3～5成分の
  split importanceが0ではなかったため、3ではなく5成分を採用する。

特徴量・前処理:
- AllFeaturesV1Transformerの19列 + DX展望SVD 5列。
- TF-IDF、SVD、補完、One-Hot Encodingは各inner/outer学習foldだけでfitする。
- SVDは名詞・動詞、単語unigram、random_state=42。
- 学習foldで全欠損の列だけを除外する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp11/feature_importance.png
- experiments/exp_ai/results/exp11/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで24。
- One-Hot後特徴量数: 37, 36, 38, 38, 37。全欠損除外: なし。
- fold threshold: 0.320, 0.200, 0.330, 0.160, 0.310。
- fold F1: 0.7273, 0.7654, 0.7500, 0.6957, 0.7500。
- fold F1 mean ± std: 0.7377 ± 0.0243。
- nested OOF F1: 0.7363、最終threshold: 0.2640。
- exp01比: -0.0271、exp10（20成分）比: -0.0008。5成分に絞っても
  改善せず、DX展望の複数SVD成分追加は見送る。
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


SPEC = ExperimentSpec("exp11", "DX展望SVD 5成分", 11)
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
