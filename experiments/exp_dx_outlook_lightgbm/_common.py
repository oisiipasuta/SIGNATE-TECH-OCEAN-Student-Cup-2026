"""DX展望の品詞アブレーションをLightGBMで比較する共通処理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from calc_features.dx_outlook import DXOutlookTfidfSVD


TARGET_COLUMN = "購入フラグ"
TEXT_COLUMN = "今後のDX展望"
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)
TFIDF_SVD_PARAMS: dict[str, Any] = {
    "n_components": 30,
    "text_column": TEXT_COLUMN,
    "min_df": 1,
    "max_features": None,
    "random_state": RANDOM_STATE,
}
MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "split",
}


@dataclass
class ExperimentResult:
    """1実験分のメモリ内評価結果。"""

    y: pd.Series
    oof_predictions: np.ndarray
    fold_scores: list[float]
    fold_thresholds: list[float]
    inner_scores: list[float]
    vocabulary_counts: list[int]
    transformed_counts: list[int]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_threshold: float
    excluded_all_missing: list[str]


def load_data(train_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """対象テキストと目的変数を読み、全欠損なら明示的に停止する。"""
    train = pd.read_csv(train_path)
    required = [TEXT_COLUMN, TARGET_COLUMN]
    missing = [column for column in required if column not in train.columns]
    if missing:
        raise KeyError(f"学習データに必要な列がありません: {missing}")
    excluded = [TEXT_COLUMN] if train[TEXT_COLUMN].isna().all() else []
    if excluded:
        raise ValueError(f"対象テキスト列が全欠損です: {excluded}")
    return train[[TEXT_COLUMN]].copy(), train[TARGET_COLUMN].astype(int), excluded


def build_text_transformer(
    parts_of_speech: tuple[str, ...],
    ngram_range: tuple[int, int] = (1, 1),
) -> DXOutlookTfidfSVD:
    """各fold専用のMeCab + TF-IDF + SVD変換器を作る。"""
    return DXOutlookTfidfSVD(
        **TFIDF_SVD_PARAMS,
        target_parts_of_speech=parts_of_speech,
        ngram_range=ngram_range,
    )


def build_model() -> LGBMClassifier:
    """exp_baseと同じ固定パラメータのLightGBMを作る。"""
    return LGBMClassifier(**MODEL_PARAMS)


def select_transformed_features(
    transformed: pd.DataFrame,
    feature_columns: tuple[str, ...] | None,
) -> pd.DataFrame:
    """SVDを30次元までfitした後、実験で固定した列だけを選ぶ。"""
    if feature_columns is None:
        return transformed
    missing = [column for column in feature_columns if column not in transformed.columns]
    if missing:
        raise KeyError(f"SVD変換後に必要な特徴量がありません: {missing}")
    return transformed.loc[:, list(feature_columns)]


def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """F1最大、同点なら0.5に近い閾値を選ぶ。"""
    scores = np.array(
        [f1_score(y_true, probabilities >= threshold) for threshold in THRESHOLD_CANDIDATES]
    )
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        tied,
        key=lambda index: (abs(THRESHOLD_CANDIDATES[index] - 0.5), index),
    )
    return float(THRESHOLD_CANDIDATES[best_index]), best_score


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    parts_of_speech: tuple[str, ...],
    seed: int,
    feature_columns: tuple[str, ...] | None,
    ngram_range: tuple[int, int],
) -> tuple[float, float]:
    """外側学習データ内のinner OOF予測だけで閾値を選ぶ。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    inner_oof_probabilities = np.zeros(len(X), dtype=float)

    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        X_train = X.iloc[inner_train_index]
        X_valid = X.iloc[inner_valid_index]
        y_train = y.iloc[inner_train_index]

        transformer = build_text_transformer(parts_of_speech, ngram_range)
        transformed_train = select_transformed_features(
            transformer.fit_transform(X_train), feature_columns
        )
        transformed_valid = select_transformed_features(
            transformer.transform(X_valid), feature_columns
        )
        model = build_model()
        model.fit(transformed_train, y_train)
        inner_oof_probabilities[inner_valid_index] = model.predict_proba(
            transformed_valid
        )[:, 1]

    return select_threshold(y, inner_oof_probabilities)


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    parts_of_speech: tuple[str, ...],
    feature_columns: tuple[str, ...] | None,
    ngram_range: tuple[int, int],
) -> ExperimentResult:
    """固定outer分割で評価し、未見foldだけからOOF指標を作る。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    vocabulary_counts: list[int] = []
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
            parts_of_speech,
            seed=RANDOM_STATE + fold + 1,
            feature_columns=feature_columns,
            ngram_range=ngram_range,
        )
        transformer = build_text_transformer(parts_of_speech, ngram_range)
        transformed_train = select_transformed_features(
            transformer.fit_transform(X_train), feature_columns
        )
        transformed_valid = select_transformed_features(
            transformer.transform(X_valid), feature_columns
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
        vocabulary_counts.append(len(transformer.vectorizer_.get_feature_names_out()))
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
            f"outer F1={fold_f1:.4f}, vocabulary={vocabulary_counts[-1]}"
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
        vocabulary_counts=vocabulary_counts,
        transformed_counts=transformed_counts,
        feature_importance=feature_importance,
        nested_oof_f1=float(f1_score(y, oof_predictions)),
        final_threshold=float(np.mean(fold_thresholds)),
        excluded_all_missing=[],
    )


def run_experiment(
    train_path: Path,
    parts_of_speech: tuple[str, ...],
    feature_columns: tuple[str, ...] | None = None,
    ngram_range: tuple[int, int] = (1, 1),
) -> ExperimentResult:
    X, y, excluded = load_data(train_path)
    result = run_nested_cv(X, y, parts_of_speech, feature_columns, ngram_range)
    result.excluded_all_missing = excluded
    return result


def configure_japanese_font() -> str:
    candidates = (
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    )
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return str(plt.rcParams["font.family"])


def make_feature_importance_figure(
    feature_importance: pd.DataFrame,
    experiment_id: str,
    parts_label: str,
) -> plt.Figure:
    """全30個のSVD特徴量をfold平均split重要度の降順で示す。"""
    plot_data = feature_importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(4.5, 0.38 * len(plot_data) + 2.8)))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color="#2563EB")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title(f"{experiment_id} LightGBM特徴量重要度：{parts_label}", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "split importance・外側5-fold平均（利用頻度であり方向・因果を表さない）",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    ax.set_xlabel("平均 split importance")
    ax.set_ylabel(f"SVD特徴量（全{len(plot_data)}列）")
    ax.grid(axis="x", alpha=0.25)
    ax.margins(x=0.18)
    fig.tight_layout()
    return fig


def make_f1_figure(
    fold_scores: list[float],
    nested_oof_f1: float,
    experiment_id: str,
    parts_label: str,
) -> plt.Figure:
    """outer validation fold別F1をゼロ基準で示す。"""
    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(fold_scores))
    fold_std = float(np.std(fold_scores))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(folds, fold_scores, color="#E07A3F", width=0.65)
    ax.bar_label(bars, labels=[f"{score:.4f}" for score in fold_scores], padding=3)
    ax.axhline(mean_score, color="#244A64", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(folds)
    ax.set_xlabel("Outer validation fold（0始まり）")
    ax.set_ylabel("F1")
    ax.set_title(f"{experiment_id} LightGBM fold別F1：{parts_label}", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "各barは学習に未使用のouter foldを、inner CVだけで選んだ閾値で評価",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    ax.text(
        0.98,
        0.97,
        f"Nested OOF F1: {nested_oof_f1:.4f}\n"
        f"Fold mean: {mean_score:.4f}\nFold std: {fold_std:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    return fig


def print_summary(
    result: ExperimentResult,
    experiment_id: str,
    experiment_name: str,
    parts_of_speech: tuple[str, ...],
    font_name: str,
) -> None:
    print("\n" + "=" * 88)
    print(f"Experiment: {experiment_id} {experiment_name}")
    print(f"Parts of speech: {list(parts_of_speech)}")
    print("Raw feature count: 1")
    print(f"TF-IDF vocabulary counts by outer fold: {result.vocabulary_counts}")
    print(f"Transformed feature counts by outer fold: {result.transformed_counts}")
    print(f"Excluded all-missing features: {result.excluded_all_missing}")
    print(f"Model params: {MODEL_PARAMS}")
    print(f"Fold thresholds: {[round(v, 3) for v in result.fold_thresholds]}")
    print(f"Fold F1: {[round(v, 4) for v in result.fold_scores]}")
    print(
        f"Fold F1 mean ± std: {np.mean(result.fold_scores):.4f} ± "
        f"{np.std(result.fold_scores):.4f}"
    )
    print(f"Nested OOF F1: {result.nested_oof_f1:.4f}")
    print(f"Final threshold: {result.final_threshold:.4f}")
    print(f"Plot font: {font_name}")
    print("=" * 88)
