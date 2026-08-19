"""
実験ID: exp08
実験名: v2未使用財務項目だけの追加
著者: Codex

仮説:
- v2が使う元項目を再追加せず、未使用の財務項目だけから情報を補えば、少ない
  変数増加で収益性・財務構成・投資行動の追加シグナルを得られる。

特徴量・前処理:
- AllFeaturesV2Transformerの20列 + 9列。
- 追加列: log_資本金、流動資産比率、負債比率、経常利益率、純利益率、
  減価償却費対売上、運転資本変動対売上、投資CF対売上、
  有形固定資産変動対売上。
- 固定資産・純資産は流動資産・負債とほぼ補完関係なので重複追加しない。
- ゼロ除算と無限値は欠損とし、補完は各学習foldだけでfitする。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp08/feature_importance.png
- experiments/exp_ai/results/exp08/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで29。
- One-Hot後特徴量数: 42, 41, 43, 43, 42。全欠損除外: なし。
- fold threshold: 0.380, 0.195, 0.205, 0.180, 0.170。
- fold F1: 0.7812, 0.7838, 0.6988, 0.7045, 0.7529。
- fold F1 mean ± std: 0.7443 ± 0.0365。
- nested OOF F1: 0.7411、最終threshold: 0.2260。
- exp01比: -0.0223。未使用財務項目に限定しても改善しなかった。
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


SPEC = ExperimentSpec("exp08", "v2未使用財務項目だけの追加", 8)
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
