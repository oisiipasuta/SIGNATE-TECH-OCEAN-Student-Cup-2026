"""
実験ID: exp03
実験名: 財務比率・欠損特徴量の追加
著者: Codex

仮説:
- exp02へ収益性、安全性、借入依存度、キャッシュフロー品質を表す比率と
  財務欠損フラグを追加すると、企業規模だけでは表現できない購買余力を捉えられる。

特徴量・前処理:
- exp02へ経常利益率、純利益率、ROA、ROE、流動資産比率、負債比率、
  借入依存度、営業CF対営業利益、投資CF対売上、減価償却費対売上を追加する。
- 営業利益・経常利益の欠損フラグを追加する。
- ゼロ除算と無限値は欠損へ変換し、補完はfoldの学習部分だけで行う。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp03/feature_importance.png
- experiments/exp_ai/results/exp03/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで34。
- One-Hot後特徴量数: 52, 52, 54, 53, 53。全欠損除外: なし。
- fold threshold: 0.365, 0.200, 0.170, 0.415, 0.280。
- fold F1: 0.8000, 0.8101, 0.7160, 0.7324, 0.7123。
- fold F1 mean ± std: 0.7542 ± 0.0422。
- nested OOF F1: 0.7534、最終threshold: 0.2860。
- exp01比-0.0100であり、財務比率ブロックの累積追加では改善しなかった。
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

SPEC = ExperimentSpec("exp03", "財務比率・欠損特徴量の追加", 3)
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
