"""
実験ID: exp05
実験名: DX展望SVD第1～10成分への拡張
著者: Codex

仮説:
- exp04のDX展望SVD第2成分だけを使う構成から第1～10成分へ広げることで、
  単一軸では失われる将来計画・課題・投資意向の複数テーマを利用できる。

特徴量・前処理:
- all_features_v1の19列とexp02～exp04の追加特徴量を使用する。
- DX展望は名詞・動詞TF-IDFを30成分へSVDし、第1～10成分を使用する。
- TF-IDF、SVD、補完、One-Hot Encodingは各inner/outer学習fold内でfitする。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 分類閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_ai/results/exp05/feature_importance.png
- experiments/exp_ai/results/exp05/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 生成特徴量数: 全outer foldで50。
- One-Hot後特徴量数: 68, 68, 70, 69, 69。全欠損除外: なし。
- fold threshold: 0.280, 0.295, 0.210, 0.220, 0.255。
- fold F1: 0.8378, 0.7647, 0.7692, 0.7059, 0.7297。
- fold F1 mean ± std: 0.7615 ± 0.0447。
- nested OOF F1: 0.7599、最終threshold: 0.2520。
- exp01比-0.0035。SVD第1～10成分の一括追加はベースラインを更新しない。
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

SPEC = ExperimentSpec("exp05", "DX展望SVD第1～10成分への拡張", 5)
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
