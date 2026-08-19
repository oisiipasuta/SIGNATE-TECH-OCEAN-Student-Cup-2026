"""
実験ID: exp05
実験名: デジタル組織数のみ追加
著者: Codex

目的・仮説:
- 組織図の全15特徴を追加したexp02の改善が、最も相関の強かった
  デジタル組織数1列だけで再現できるかを確認する。

特徴量・前処理:
- all_features_v3の23列へデジタル組織数1列だけを追加する。
- DX推進室完全一致、比率、その他の構造・機能特徴は使用しない。
- 特徴量生成、業界統合、TF-IDF/SVD、補完、One-Hot Encodingは各fold内でfit。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。
- all_features_v3既存値nested OOF F1=0.7744との差を報告する。

出力:
- experiments/exp_tree_of_corp/results/exp05/feature_importance.png
- experiments/exp_tree_of_corp/results/exp05/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成入力特徴量数24、One-Hot後37/36/38/38/37列。全行欠損による除外なし。
- fold閾値: 0.375/0.215/0.235/0.425/0.425、最終閾値=0.3350。
- fold F1: 0.7353/0.8267/0.7895/0.7606/0.7385、平均=0.7701、
  標準偏差=0.0343、nested OOF F1=0.7718。
- all_features_v3の0.7744に対して-0.0026。デジタル組織数1列では改善しなかった。
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

SPEC = ExperimentSpec("exp05", "デジタル組織数のみ追加", 5)
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
