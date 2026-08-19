"""
実験ID: exp22
実験名: DX展望・固定辞書v1（7カテゴリ×2集計）LightGBM
著者: Codex

目的・仮説:
- 購入フラグを見ずに固定したDX展望辞書v1が、購入フラグの分類に有効か検証する。
- 各カテゴリの単純な有無ではなく、一致表現の種類数と総出現回数を使うことで、
  教育需要、外部教育、投資姿勢、課題の強さを表現できると仮定する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。
- calc_features.calculate_dx_outlook_dictionary_featuresを使用する。
- EDU、EXTERNAL、EXPAND、CAUTIOUS、MAINTAIN、SUPPRESS、NEEDの各カテゴリで、
  matched_expressions（異なる一致表現数）とtotal_occurrences（総出現回数）を作る。
- 入力1列から計14数値特徴量へ変換する。辞書は固定v1で、CV中に変更・fitしない。
- 同一カテゴリ内の入れ子表現は別表現として数える。標準化や欠損補完は行わない。

モデル・評価:
- 固定LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- stratified outer 5-fold、inner 4-fold、random_state=42。
- 各outer foldの閾値は、そのouter学習部分に対するinner OOF F1だけで選ぶ。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp22/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp22/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1、変換後特徴量数: 各fold 14
- 除外した全欠損特徴量: なし
- fold threshold: 0.290, 0.260, 0.170, 0.225, 0.180
- fold F1: 0.5610, 0.4792, 0.5306, 0.4800, 0.4848
- fold F1 mean ± std: 0.5071 ± 0.0331
- nested OOF F1: 0.5053、最終threshold: 0.2250
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features import (
    DX_OUTLOOK_DICTIONARY_FEATURE_COLUMNS,
    DX_OUTLOOK_DICTIONARY_VERSION,
    calculate_dx_outlook_dictionary_features,
)
from experiments.exp_dx_outlook_lightgbm._common import (
    INNER_N_SPLITS,
    MODEL_PARAMS,
    OUTER_N_SPLITS,
    RANDOM_STATE,
    build_model,
    configure_japanese_font,
    load_data,
    make_f1_figure,
    make_feature_importance_figure,
    select_threshold,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp22"
EXPERIMENT_NAME = "DX展望・固定辞書v1（7カテゴリ×2集計）LightGBM"
FEATURE_LABEL = "固定辞書v1・7カテゴリ×一致種類数/総出現回数（計14）"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


@dataclass
class ExperimentResult:
    y: pd.Series
    oof_predictions: np.ndarray
    fold_scores: list[float]
    fold_thresholds: list[float]
    inner_scores: list[float]
    transformed_counts: list[int]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_threshold: float
    excluded_all_missing: list[str]


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# ==================================================

def transform_dictionary_features(X: pd.DataFrame) -> pd.DataFrame:
    """購入フラグを受け取らず、固定済み辞書v1だけで14特徴量を作る。"""
    transformed = calculate_dx_outlook_dictionary_features(X)
    if transformed.isna().any().any():
        raise ValueError("辞書特徴量に欠損値が生成されました。")
    if transformed.columns.tolist() != DX_OUTLOOK_DICTIONARY_FEATURE_COLUMNS:
        raise ValueError("辞書特徴量の列順が固定仕様と一致しません。")
    return transformed


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[float, float]:
    """outer学習部分のinner OOF予測だけで閾値を選ぶ。"""
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    inner_oof_probabilities = np.zeros(len(X), dtype=float)

    for train_index, valid_index in inner_cv.split(X, y):
        transformed_train = transform_dictionary_features(X.iloc[train_index])
        transformed_valid = transform_dictionary_features(X.iloc[valid_index])
        model = build_model()
        model.fit(transformed_train, y.iloc[train_index])
        inner_oof_probabilities[valid_index] = model.predict_proba(transformed_valid)[:, 1]

    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 4. LightGBM
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_experiment() -> ExperimentResult:
    X, y, excluded = load_data(TRAIN_PATH)
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    transformed_counts: list[int] = []
    importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        threshold, inner_f1 = calculate_inner_threshold(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            seed=RANDOM_STATE + fold + 1,
        )
        transformed_train = transform_dictionary_features(X_train)
        transformed_valid = transform_dictionary_features(X_valid)
        model = build_model()
        model.fit(transformed_train, y_train)
        valid_probabilities = model.predict_proba(transformed_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        transformed_counts.append(transformed_train.shape[1])
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": transformed_train.columns,
                    f"fold_{fold}": model.feature_importances_,
                }
            ).set_index("feature")
        )
        print(
            f"Fold {fold}: inner F1={inner_f1:.4f}, threshold={threshold:.3f}, "
            f"outer F1={fold_f1:.4f}, features={transformed_train.shape[1]}"
        )

    feature_importance = (
        pd.concat(importance_frames, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .reset_index()
    )
    return ExperimentResult(
        y=y,
        oof_predictions=oof_predictions,
        fold_scores=fold_scores,
        fold_thresholds=fold_thresholds,
        inner_scores=inner_scores,
        transformed_counts=transformed_counts,
        feature_importance=feature_importance,
        nested_oof_f1=float(f1_score(y, oof_predictions)),
        final_threshold=float(np.mean(fold_thresholds)),
        excluded_all_missing=excluded,
    )


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def print_summary(result: ExperimentResult, font_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Dictionary version: {DX_OUTLOOK_DICTIONARY_VERSION}")
    print("Raw feature count: 1")
    print(f"Transformed feature counts: {result.transformed_counts}")
    print(f"Excluded all-missing features: {result.excluded_all_missing}")
    print(f"Model params: {MODEL_PARAMS}")
    print(f"Fold thresholds: {[round(value, 3) for value in result.fold_thresholds]}")
    print(f"Fold F1: {[round(value, 4) for value in result.fold_scores]}")
    print(
        f"Fold F1 mean ± std: {np.mean(result.fold_scores):.4f} ± "
        f"{np.std(result.fold_scores):.4f}"
    )
    print(f"Nested OOF F1: {result.nested_oof_f1:.4f}")
    print(f"Final threshold: {result.final_threshold:.4f}")
    print(f"Plot font: {font_name}")
    print("=" * 88)


def main() -> None:
    result = run_experiment()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(
        result.feature_importance,
        EXPERIMENT_ID,
        FEATURE_LABEL,
    )
    f1_figure = make_f1_figure(
        result.fold_scores,
        result.nested_oof_f1,
        EXPERIMENT_ID,
        FEATURE_LABEL,
    )
    feature_figure.savefig(
        RESULT_DIR / "feature_importance.png",
        dpi=160,
        bbox_inches="tight",
    )
    f1_figure.savefig(
        RESULT_DIR / "f1_scores.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(feature_figure)
    plt.close(f1_figure)
    print_summary(result, font_name)


if __name__ == "__main__":
    main()
