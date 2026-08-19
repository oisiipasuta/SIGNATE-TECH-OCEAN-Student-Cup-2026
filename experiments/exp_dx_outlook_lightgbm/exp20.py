"""
実験ID: exp20
実験名: min_df=2 名詞unigram SVD 30 ＋ bigram SVD 30（結合60次元）
著者: Codex

目的・仮説:
- exp19で1文書だけに出現する名詞がunigramの約30%、bigramの約67%を占めた。
- 各TF-IDFへmin_df=2を設定し、単発語を除外することでSVD空間のノイズを減らし、
  exp19よりF1が改善するかを検証する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。MeCab（UniDic）で名詞だけを抽出する。
- calc_features.dx_outlook.DXOutlookMultiNgramTfidfSVDを利用する。
- unigram TF-IDF (1,1) → SVD 30 と bigram TF-IDF (2,2) → SVD 30を
  独立にfitし、計60列を結合する。各TF-IDFはmin_df=2、max_features=None。
- inner/outer CVの各学習fold内だけでTF-IDFとSVDをfitし、検証foldにはtransformのみ行う。

モデル・評価:
- 固定LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- stratified outer 5-fold、inner 4-fold、random_state=42。
- 各outer foldの閾値は、そのouter学習部分に対するinner OOF F1だけで選ぶ。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp20/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp20/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1
- unigram語彙数: 2038, 2012, 2023, 1993, 2025
- bigram語彙数: 13876, 13832, 13864, 13835, 13769
- 変換後特徴量数: 各fold 60、除外した全欠損特徴量: なし
- fold threshold: 0.170, 0.240, 0.285, 0.180, 0.175
- fold F1: 0.5421, 0.4902, 0.5352, 0.5660, 0.5055
- fold F1 mean ± std: 0.5278 ± 0.0270
- nested OOF F1: 0.5283、最終threshold: 0.2100
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

from calc_features.dx_outlook import DXOutlookMultiNgramTfidfSVD
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

EXPERIMENT_ID = "exp20"
EXPERIMENT_NAME = "min_df=2 名詞unigram SVD 30 ＋ bigram SVD 30（結合60次元）"
PARTS_OF_SPEECH = ("名詞",)
CHANNELS = (("unigram", (1, 1)), ("bigram", (2, 2)))
N_COMPONENTS = 30
MIN_DF = 2
FEATURE_LABEL = "名詞・min_df=2・unigram 30＋bigram 30（計60）"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


@dataclass
class ExperimentResult:
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

def build_feature_transformer() -> DXOutlookMultiNgramTfidfSVD:
    """学習fold専用のmin_df対応TF-IDF・SVD変換器を作る。"""
    return DXOutlookMultiNgramTfidfSVD(
        n_components=N_COMPONENTS,
        target_parts_of_speech=PARTS_OF_SPEECH,
        channels=CHANNELS,
        min_df=MIN_DF,
        max_features=None,
        random_state=RANDOM_STATE,
    )


def fit_transform_fold(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """学習部分だけでfitし、学習・検証特徴量と語彙数を返す。"""
    transformer = build_feature_transformer()
    transformed_train = transformer.fit_transform(X_train)
    transformed_valid = transformer.transform(X_valid)
    return transformed_train, transformed_valid, transformer.vocabulary_counts_.copy()


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[float, float]:
    """outer学習部分のinner OOF予測だけで閾値を選ぶ。"""
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    inner_oof_probabilities = np.zeros(len(X), dtype=float)

    for train_index, valid_index in inner_cv.split(X, y):
        transformed_train, transformed_valid, _ = fit_transform_fold(
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
        transformed_train, transformed_valid, vocabulary = fit_transform_fold(
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
                {
                    "feature": transformed_train.columns,
                    f"fold_{fold}": model.feature_importances_,
                }
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
    return ExperimentResult(
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

def print_summary(result: ExperimentResult, font_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Parts of speech: {list(PARTS_OF_SPEECH)}")
    print(f"Channels: {CHANNELS}, min_df={MIN_DF}")
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
