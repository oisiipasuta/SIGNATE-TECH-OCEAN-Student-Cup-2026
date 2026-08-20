"""
実験ID: exp09
実験名: v2未使用アンケートだけの追加
著者: Codex

仮説:
- v2の最終20列に情報が入っていないアンケート3・6・10だけを独立追加し、
  既存回答から作れる差分特徴量を重ねないことで変数増加を3列に抑える。

特徴量・前処理:
- AllFeaturesV2Transformerの20列 + アンケート3・6・10の3列。
- 1～5以外は欠損とし、中央値補完は各学習foldだけでfitする。
- 学習foldで全欠損の列だけを除外する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp09/feature_importance.png
- experiments/exp_ai/results/exp09/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで23。
- One-Hot後特徴量数: 36, 35, 37, 37, 36。全欠損除外: なし。
- fold threshold: 0.415, 0.255, 0.205, 0.445, 0.310。
- fold F1: 0.7077, 0.8493, 0.7561, 0.7647, 0.7887。
- fold F1 mean ± std: 0.7733 ± 0.0462。
- nested OOF F1: 0.7744、最終threshold: 0.3260。
- exp01比: +0.0110。今回の独立追加4実験では最高だった。
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


SPEC = ExperimentSpec("exp09", "v2未使用アンケートだけの追加", 9)
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
