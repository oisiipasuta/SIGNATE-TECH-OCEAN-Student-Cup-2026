"""
実験ID: exp21
実験名: min_df=2・品詞別unigram＋bigram比較
著者: Codex

目的・仮説:
- exp20の名詞特徴量と同じTF-IDF・SVD条件を、名詞、動詞、形容詞、副詞へ
  個別に適用し、どの品詞の情報が購入フラグの分類に有効か比較する。
- UniDicでは形容表現が形容詞と形状詞に分かれるため、形容詞条件には形状詞も含める。

特徴量・前処理:
- 入力は「今後のDX展望」1列。
- calc_features.dx_outlook.DXOutlookMultiNgramTfidfSVDを利用する。
- 比較条件は名詞のみ、動詞のみ、形容詞＋形状詞、副詞のみの4種類。
- 各条件でunigram TF-IDF → SVD 30とbigram TF-IDF → SVD 30を独立にfitし、
  60列へ結合する。min_df=2、max_features=None、random_state=42。
- inner/outer CVの各学習fold内だけでTF-IDFとSVDをfitする。

モデル・評価:
- 固定LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- 全条件で同じstratified outer 5-fold、inner 4-foldを使う。
- 各outer foldの閾値は、そのouter学習部分に対するinner OOF F1だけで選ぶ。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp21/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp21/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 各条件1、変換後特徴量数: 各条件・各fold 60
- 除外した全欠損特徴量: なし
- 名詞: unigram語彙 2038, 2012, 2023, 1993, 2025、
  bigram語彙 13876, 13832, 13864, 13835, 13769、
  threshold 0.170, 0.240, 0.285, 0.180, 0.175、
  fold F1 0.5421, 0.4902, 0.5352, 0.5660, 0.5055、
  mean 0.5278 ± 0.0270、nested OOF F1 0.5283、最終threshold 0.2100。
- 動詞: unigram語彙 690, 688, 682, 691, 687、
  bigram語彙 2867, 2837, 2840, 2845, 2836、
  threshold 0.195, 0.205, 0.140, 0.110, 0.245、
  fold F1 0.5977, 0.4909, 0.5000, 0.4580, 0.5263、
  mean 0.5146 ± 0.0469、nested OOF F1 0.5081、最終threshold 0.1790。
- 形容詞＋形状詞: unigram語彙 176, 180, 176, 177, 175、
  bigram語彙 898, 927, 936, 906, 917、
  threshold 0.140, 0.290, 0.080, 0.185, 0.175、
  fold F1 0.5273, 0.5000, 0.4839, 0.4952, 0.4368、
  mean 0.4886 ± 0.0296、nested OOF F1 0.4903、最終threshold 0.1740。
- 副詞: unigram語彙 65, 65, 65, 61, 63、bigram語彙 311, 297, 292, 297, 300、
  threshold 0.080, 0.050, 0.055, 0.075, 0.055、
  fold F1 0.3788, 0.4161, 0.4521, 0.3780, 0.4507、
  mean 0.4151 ± 0.0327、nested OOF F1 0.4167、最終threshold 0.0630。
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
    select_threshold,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp21"
EXPERIMENT_NAME = "min_df=2・品詞別unigram＋bigram比較"
CHANNELS = (("unigram", (1, 1)), ("bigram", (2, 2)))
N_COMPONENTS = 30
MIN_DF = 2
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


@dataclass(frozen=True)
class Variant:
    key: str
    label: str
    parts_of_speech: tuple[str, ...]


VARIANTS = (
    Variant("noun", "名詞", ("名詞",)),
    Variant("verb", "動詞", ("動詞",)),
    Variant("adjective", "形容詞＋形状詞", ("形容詞", "形状詞")),
    Variant("adverb", "副詞", ("副詞",)),
)


@dataclass
class VariantResult:
    variant: Variant
    fold_scores: list[float]
    fold_thresholds: list[float]
    inner_scores: list[float]
    unigram_vocabulary_counts: list[int]
    bigram_vocabulary_counts: list[int]
    transformed_counts: list[int]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_threshold: float


@dataclass
class ComparisonResult:
    variants: list[VariantResult]
    excluded_all_missing: list[str]


# ==================================================
# 2. データ読み込み
# 3. 特徴量前処理
# ==================================================

def build_feature_transformer(variant: Variant) -> DXOutlookMultiNgramTfidfSVD:
    """品詞条件ごとの学習fold専用変換器を作る。"""
    return DXOutlookMultiNgramTfidfSVD(
        n_components=N_COMPONENTS,
        target_parts_of_speech=variant.parts_of_speech,
        channels=CHANNELS,
        min_df=MIN_DF,
        max_features=None,
        random_state=RANDOM_STATE,
    )


def fit_transform_fold(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    variant: Variant,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    transformer = build_feature_transformer(variant)
    transformed_train = transformer.fit_transform(X_train)
    transformed_valid = transformer.transform(X_valid)
    return transformed_train, transformed_valid, transformer.vocabulary_counts_.copy()


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    variant: Variant,
    seed: int,
) -> tuple[float, float]:
    """outer学習部分のinner OOF予測だけで品詞条件ごとの閾値を選ぶ。"""
    inner_cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    inner_oof_probabilities = np.zeros(len(X), dtype=float)

    for train_index, valid_index in inner_cv.split(X, y):
        transformed_train, transformed_valid, _ = fit_transform_fold(
            X.iloc[train_index],
            X.iloc[valid_index],
            variant,
        )
        model = build_model()
        model.fit(transformed_train, y.iloc[train_index])
        inner_oof_probabilities[valid_index] = model.predict_proba(transformed_valid)[:, 1]

    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 4. LightGBM
# 5. ネステッド・クロスバリデーション
# ==================================================

def evaluate_variant(X: pd.DataFrame, y: pd.Series, variant: Variant) -> VariantResult:
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
            variant,
            seed=RANDOM_STATE + fold + 1,
        )
        transformed_train, transformed_valid, vocabulary = fit_transform_fold(
            X_train,
            X_valid,
            variant,
        )
        model = build_model()
        model.fit(transformed_train, y_train)
        probabilities = model.predict_proba(transformed_valid)[:, 1]
        predictions = (probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, predictions))

        oof_predictions[valid_index] = predictions
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
            f"{variant.label} fold {fold}: inner F1={inner_f1:.4f}, "
            f"threshold={threshold:.3f}, outer F1={fold_f1:.4f}, "
            f"unigram vocab={vocabulary['unigram']}, bigram vocab={vocabulary['bigram']}"
        )

    feature_importance = (
        pd.concat(importance_frames, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .reset_index()
    )
    return VariantResult(
        variant=variant,
        fold_scores=fold_scores,
        fold_thresholds=fold_thresholds,
        inner_scores=inner_scores,
        unigram_vocabulary_counts=unigram_counts,
        bigram_vocabulary_counts=bigram_counts,
        transformed_counts=transformed_counts,
        feature_importance=feature_importance,
        nested_oof_f1=float(f1_score(y, oof_predictions)),
        final_threshold=float(np.mean(fold_thresholds)),
    )


def run_experiment() -> ComparisonResult:
    X, y, excluded = load_data(TRAIN_PATH)
    results = [evaluate_variant(X, y, variant) for variant in VARIANTS]
    return ComparisonResult(variants=results, excluded_all_missing=excluded)


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def make_feature_importance_figure(result: ComparisonResult) -> plt.Figure:
    """4条件それぞれの全60特徴量を、outer-fold平均split重要度順に描く。"""
    figure, axes = plt.subplots(1, len(result.variants), figsize=(36, 25))
    for axis, variant_result in zip(axes, result.variants):
        importance = variant_result.feature_importance.sort_values(
            "importance",
            ascending=True,
        )
        bars = axis.barh(
            importance["feature"],
            importance["importance"],
            color="#2c6bed",
        )
        axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=6)
        axis.set_title(variant_result.variant.label, fontsize=14)
        axis.set_xlabel("平均 split importance")
        axis.tick_params(axis="y", labelsize=7)
        axis.grid(axis="x", alpha=0.25)
        maximum = max(float(importance["importance"].max()), 1.0)
        axis.set_xlim(0, maximum * 1.18)
    figure.suptitle(
        f"{EXPERIMENT_ID} LightGBM特徴量重要度：品詞別 min_df=2 比較",
        fontsize=20,
        y=0.995,
    )
    figure.text(
        0.5,
        0.978,
        "split importance・outer 5-fold平均（利用頻度であり方向・因果を表さない）",
        ha="center",
        color="#5f6b7a",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    return figure


def make_f1_figure(result: ComparisonResult) -> plt.Figure:
    """各品詞条件のouter fold F1とnested OOF F1を比較する。"""
    figure, axes = plt.subplots(2, 2, figsize=(18, 12), sharey=True)
    for axis, variant_result in zip(axes.flat, result.variants):
        folds = np.arange(len(variant_result.fold_scores))
        mean_score = float(np.mean(variant_result.fold_scores))
        std_score = float(np.std(variant_result.fold_scores))
        bars = axis.bar(folds, variant_result.fold_scores, color="#e6793c", width=0.65)
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=10)
        axis.axhline(mean_score, color="#234d68", linestyle="--", linewidth=2)
        axis.set_title(variant_result.variant.label, fontsize=15)
        axis.set_xlabel("Outer validation fold（0始まり）")
        axis.set_ylabel("F1")
        axis.set_xticks(folds)
        axis.set_ylim(0, 1)
        axis.grid(axis="y", alpha=0.25)
        axis.text(
            0.98,
            0.96,
            f"Nested OOF F1: {variant_result.nested_oof_f1:.4f}\n"
            f"Fold mean: {mean_score:.4f}\n"
            f"Fold std: {std_score:.4f}\n"
            f"Final threshold: {variant_result.final_threshold:.4f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#222222", "boxstyle": "round"},
        )
    figure.suptitle(
        f"{EXPERIMENT_ID} 品詞別 nested CV F1：min_df=2・unigram 30＋bigram 30",
        fontsize=20,
        y=0.995,
    )
    figure.text(
        0.5,
        0.952,
        "各barは学習に未使用のouter foldを、inner CVだけで選んだ閾値で評価",
        ha="center",
        color="#5f6b7a",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.915))
    return figure


def print_summary(result: ComparisonResult, font_name: str) -> None:
    print("\n" + "=" * 100)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Raw feature count per variant: 1, min_df={MIN_DF}, channels={CHANNELS}")
    print(f"Excluded all-missing features: {result.excluded_all_missing}")
    print(f"Model params: {MODEL_PARAMS}")
    for variant_result in result.variants:
        print(f"\n[{variant_result.variant.label}]")
        print(f"Parts of speech: {list(variant_result.variant.parts_of_speech)}")
        print(f"Unigram vocabulary counts: {variant_result.unigram_vocabulary_counts}")
        print(f"Bigram vocabulary counts: {variant_result.bigram_vocabulary_counts}")
        print(f"Transformed feature counts: {variant_result.transformed_counts}")
        print(f"Fold thresholds: {[round(value, 3) for value in variant_result.fold_thresholds]}")
        print(f"Fold F1: {[round(value, 4) for value in variant_result.fold_scores]}")
        print(
            f"Fold F1 mean ± std: {np.mean(variant_result.fold_scores):.4f} ± "
            f"{np.std(variant_result.fold_scores):.4f}"
        )
        print(f"Nested OOF F1: {variant_result.nested_oof_f1:.4f}")
        print(f"Final threshold: {variant_result.final_threshold:.4f}")
    print(f"Plot font: {font_name}")
    print("=" * 100)


def main() -> None:
    result = run_experiment()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(result)
    f1_figure = make_f1_figure(result)
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
