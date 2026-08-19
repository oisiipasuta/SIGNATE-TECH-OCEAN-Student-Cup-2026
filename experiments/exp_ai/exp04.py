"""
実験ID: exp04
実験名: アンケート関係特徴量の追加
著者: Codex

仮説:
- exp03へ導入状況、外部連携、戦略とのギャップを追加すると、単一回答よりも
  DXの意欲と実行状況の不一致を表現できる。

特徴量・前処理:
- 未使用のアンケート3・6・10を追加する。
- 戦略導入ギャップ、戦略抵抗ギャップ、外部支援必要度、
  ツール未導入フラグを追加する。
- 満足度不足と成果不足は既存のDX全体不満度・DX成果不足度を継続利用する。
- 全変換は各inner/outer学習fold内でfitする。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp04/feature_importance.png
- experiments/exp_ai/results/exp04/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで41。
- One-Hot後特徴量数: 59, 59, 61, 60, 60。全欠損除外: なし。
- fold threshold: 0.425, 0.355, 0.170, 0.435, 0.305。
- fold F1: 0.8308, 0.7826, 0.7059, 0.7536, 0.7606。
- fold F1 mean ± std: 0.7667 ± 0.0407。
- nested OOF F1: 0.7632、最終threshold: 0.3380。
- exp01比-0.0002で同等。追加ブロック中では最も有望だが、単純累積では更新しない。
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

SPEC = ExperimentSpec("exp04", "アンケート関係特徴量の追加", 4)
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
