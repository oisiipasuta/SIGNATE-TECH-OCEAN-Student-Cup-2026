"""
実験ID: exp19
実験名: 名詞unigram SVD 30 ＋ bigram SVD 30（結合60次元）
著者: Codex

目的・仮説:
- unigramとbigramを同じSVD空間へ混ぜず、別々に30次元化して結合することで、
  単語情報と組合せ情報の両方を保持できるか検証する。

特徴量・前処理:
- 「今後のDX展望」からMeCab（UniDic）で名詞だけを抽出する。
- unigram TF-IDF (1,1) → SVD 30 と bigram TF-IDF (2,2) → SVD 30を
  各学習fold内で独立にfitし、unigram_svd_*とbigram_svd_*の計60列を結合する。
- inner CVでも各inner学習fold内だけで両変換器をfitし、前処理リークを防ぐ。

モデル・評価:
- 固定LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer 5-fold、inner 4-fold。閾値はouter学習部分のinner OOF F1だけで選ぶ。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp19/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp19/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1
- unigram語彙数: 2941, 2915, 2886, 2871, 2930
- bigram語彙数: 44630, 44264, 44562, 44139, 44402
- 結合後特徴量数: 各fold 60、除外した全欠損特徴量: なし
- fold threshold: 0.200, 0.155, 0.160, 0.155, 0.160
- fold F1: 0.6122, 0.5203, 0.5652, 0.5043, 0.5783
- fold F1 mean ± std: 0.5561 ± 0.0392
- nested OOF F1: 0.5519、最終threshold: 0.1660
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

from experiments.exp_dx_outlook_lightgbm._common import (
    INNER_N_SPLITS,
    MODEL_PARAMS,
    OUTER_N_SPLITS,
    RANDOM_STATE,
    build_model,
    build_text_transformer,
    configure_japanese_font,
    load_data,
    make_f1_figure,
    make_feature_importance_figure,
    select_threshold,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp19"
EXPERIMENT_NAME = "名詞unigram SVD 30 ＋ bigram SVD 30（結合60次元）"
PARTS_OF_SPEECH = ("名詞",)
CHANNELS = (("unigram", (1, 1)), ("bigram", (2, 2)))
FEATURE_LABEL = "名詞・unigram 30＋bigram 30（計60）"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


@dataclass
class CombinedResult:
    y: pd.Series
    oof_predictions: np.ndarray
    fold_scores: list[float]
    fold_thresholds: list[float]
    inner_scores: list[float]
    unigram_vocabulary_counts: list[int]
    bigram_vocabulary_counts: list[int]
    transformed_counts: list[int]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_threshold: float
    excluded_all_missing: list[str]


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# ==================================================

def fit_transform_channels(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    train_frames: list[pd.DataFrame] = []
    valid_frames: list[pd.DataFrame] = []
    vocabulary_counts: dict[str, int] = {}

    for channel, ngram_range in CHANNELS:
        transformer = build_text_transformer(PARTS_OF_SPEECH, ngram_range)
        transformed_train = transformer.fit_transform(X_train).copy()
        transformed_valid = transformer.transform(X_valid).copy()
        columns = [f"{channel}_svd_{index + 1:02d}" for index in range(30)]
        transformed_train.columns = columns
        transformed_valid.columns = columns
        train_frames.append(transformed_train)
        valid_frames.append(transformed_valid)
        vocabulary_counts[channel] = len(transformer.vectorizer_.get_feature_names_out())

    return (
        pd.concat(train_frames, axis=1),
        pd.concat(valid_frames, axis=1),
        vocabulary_counts,
    )


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[float, float]:
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    inner_oof_probabilities = np.zeros(len(X), dtype=float)

    for train_index, valid_index in inner_cv.split(X, y):
        transformed_train, transformed_valid, _ = fit_transform_channels(
            X.iloc[train_index], X.iloc[valid_index]
        )
        model = build_model()
        model.fit(transformed_train, y.iloc[train_index])
        inner_oof_probabilities[valid_index] = model.predict_proba(transformed_valid)[:, 1]

    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 4. LightGBM
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_combined_experiment() -> CombinedResult:
    X, y, excluded = load_data(TRAIN_PATH)
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    unigram_counts: list[int] = []
    bigram_counts: list[int] = []
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
        transformed_train, transformed_valid, vocabulary = fit_transform_channels(
            X_train, X_valid
        )
        model = build_model()
        model.fit(transformed_train, y_train)
        valid_probabilities = model.predict_proba(transformed_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        unigram_counts.append(vocabulary["unigram"])
        bigram_counts.append(vocabulary["bigram"])
        transformed_counts.append(transformed_train.shape[1])
        importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_train.columns, f"fold_{fold}": model.feature_importances_}
            ).set_index("feature")
        )
        print(
            f"Fold {fold}: inner F1={inner_f1:.4f}, threshold={threshold:.3f}, "
            f"outer F1={fold_f1:.4f}, unigram vocab={vocabulary['unigram']}, "
            f"bigram vocab={vocabulary['bigram']}"
        )

    feature_importance = (
        pd.concat(importance_frames, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .reset_index()
    )
    return CombinedResult(
        y=y,
        oof_predictions=oof_predictions,
        fold_scores=fold_scores,
        fold_thresholds=fold_thresholds,
        inner_scores=inner_scores,
        unigram_vocabulary_counts=unigram_counts,
        bigram_vocabulary_counts=bigram_counts,
        transformed_counts=transformed_counts,
        feature_importance=feature_importance,
        nested_oof_f1=float(f1_score(y, oof_predictions)),
        final_threshold=float(np.mean(fold_thresholds)),
        excluded_all_missing=excluded,
    )


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def print_summary(result: CombinedResult, font_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Parts of speech: {list(PARTS_OF_SPEECH)}")
    print(f"Channels: {CHANNELS}")
    print("Raw feature count: 1")
    print(f"Unigram vocabulary counts: {result.unigram_vocabulary_counts}")
    print(f"Bigram vocabulary counts: {result.bigram_vocabulary_counts}")
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
    result = run_combined_experiment()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(
        result.feature_importance, EXPERIMENT_ID, FEATURE_LABEL
    )
    f1_figure = make_f1_figure(
        result.fold_scores, result.nested_oof_f1, EXPERIMENT_ID, FEATURE_LABEL
    )
    feature_figure.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    print_summary(result, font_name)


if __name__ == "__main__":
    main()
