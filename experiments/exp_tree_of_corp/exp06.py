"""
実験ID: exp06
実験名: DX推進室完全一致のみ追加
著者: Codex

目的・仮説:
- all_features_v3へDX推進室完全一致フラグ1列だけを追加し、競技データ固有の
  頻出表現が単独で持つ追加効果を確認する。

特徴量・前処理:
- all_features_v3の23列へDX推進室完全一致フラグ1列だけを追加する。
- デジタル組織数、比率、その他の構造・機能特徴は使用しない。
- 「必ず含む」など生成注記らしい表現は使用しない。
- 特徴量生成、業界統合、TF-IDF/SVD、補完、One-Hot Encodingは各fold内でfit。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。
- all_features_v3既存値nested OOF F1=0.7744との差を報告する。

出力:
- experiments/exp_tree_of_corp/results/exp06/feature_importance.png
- experiments/exp_tree_of_corp/results/exp06/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成入力特徴量数24、One-Hot後37/36/38/38/37列。全行欠損による除外なし。
- fold閾値: 0.245/0.340/0.375/0.380/0.260、最終閾値=0.3200。
- fold F1: 0.7887/0.8000/0.8056/0.7368/0.8333、平均=0.7929、
  標準偏差=0.0316、nested OOF F1=0.7922。
- all_features_v3の0.7744に対して+0.0178。完全一致1列には追加効果があったが、
  汎用15特徴を加えたexp02の0.7989には届かなかった。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.exp_tree_of_corp._common import (  # noqa: E402
    ExperimentSpec,
    configure_japanese_font,
    load_data,
    make_f1_figure,
    make_feature_importance_figure,
    print_association_profile,
    print_summary,
    run_nested_cv,
)


# ==================================================
# 1. 実験設定
# ==================================================

SPEC = ExperimentSpec("exp06", "DX推進室完全一致のみ追加", 6)
RESULT_DIR = BASE_DIR / "experiments" / "exp_tree_of_corp" / "results" / SPEC.experiment_id


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# 4. LightGBM・閾値選択
# 5. ネステッド・クロスバリデーション
# 6. 実験結果・図の出力
# ==================================================

def main() -> None:
    train, test, y = load_data()
    print_association_profile(train, test, y)
    result = run_nested_cv(train, y, SPEC)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(result.feature_importance, SPEC)
    f1_figure = make_f1_figure(result, SPEC)
    feature_figure.savefig(
        RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    print_summary(train, result, SPEC, font_name)


if __name__ == "__main__":
    main()
