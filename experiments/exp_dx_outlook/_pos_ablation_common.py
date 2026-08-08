"""品詞アブレーション実験で共用する、リークのないネストCV実装。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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


@dataclass(frozen=True)
class RegularizationConfig:
    """ロジスティック回帰の正則化候補。"""

    label: str
    penalty: str
    c: float
    l1_ratio: float | None = None


REGULARIZATION_CONFIGS = (
    *(
        RegularizationConfig(f"L2 / C={c:g}", "l2", c)
        for c in (0.001, 0.01, 0.1, 1.0, 10.0)
    ),
    *(
        RegularizationConfig(f"L1 / C={c:g}", "l1", c)
        for c in (0.01, 0.1, 1.0, 10.0)
    ),
    *(
        RegularizationConfig(
            f"ElasticNet({ratio:.2f}) / C={c:g}",
            "elasticnet",
            c,
            ratio,
        )
        for c in (0.1, 1.0, 10.0)
        for ratio in (0.25, 0.50, 0.75)
    ),
)


@dataclass
class ExperimentResult:
    """1実験分のメモリ内評価結果。"""

    y: pd.Series
    oof_predictions: np.ndarray
    fold_scores: list[float]
    fold_thresholds: list[float]
    selected_configs: list[RegularizationConfig]
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


def build_text_transformer(parts_of_speech: tuple[str, ...]) -> DXOutlookTfidfSVD:
    """各fold専用のMeCab + TF-IDF + SVD変換器を作る。"""
    return DXOutlookTfidfSVD(
        **TFIDF_SVD_PARAMS,
        target_parts_of_speech=parts_of_speech,
    )


def build_model(config: RegularizationConfig) -> Pipeline:
    """標準化と正則化付きロジスティック回帰をfold内でfitする。"""
    model_kwargs: dict[str, Any] = {
        "C": config.c,
        # scikit-learn 1.8以降はpenaltyではなくl1_ratioで正則化を指定する。
        "l1_ratio": (
            config.l1_ratio
            if config.penalty == "elasticnet"
            else (1.0 if config.penalty == "l1" else 0.0)
        ),
        "solver": "saga",
        "max_iter": 5000,
        "tol": 1e-4,
        "random_state": RANDOM_STATE,
    }
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(**model_kwargs)),
        ]
    )


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


def select_regularization_and_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    parts_of_speech: tuple[str, ...],
    seed: int,
) -> tuple[RegularizationConfig, float, float]:
    """外側学習データ内だけで、正則化候補と閾値を同時選択する。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    candidate_probabilities = np.zeros(
        (len(REGULARIZATION_CONFIGS), len(X)), dtype=float
    )

    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        X_train = X.iloc[inner_train_index]
        X_valid = X.iloc[inner_valid_index]
        y_train = y.iloc[inner_train_index]

        transformer = build_text_transformer(parts_of_speech)
        transformed_train = transformer.fit_transform(X_train)
        transformed_valid = transformer.transform(X_valid)
        for config_index, config in enumerate(REGULARIZATION_CONFIGS):
            model = build_model(config)
            model.fit(transformed_train, y_train)
            candidate_probabilities[config_index, inner_valid_index] = (
                model.predict_proba(transformed_valid)[:, 1]
            )

    candidates: list[tuple[float, int, RegularizationConfig, float]] = []
    for config_index, config in enumerate(REGULARIZATION_CONFIGS):
        threshold, score = select_threshold(y, candidate_probabilities[config_index])
        candidates.append((score, -config_index, config, threshold))
    best = max(candidates, key=lambda candidate: candidate[:2])
    return best[2], best[3], best[0]


def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    parts_of_speech: tuple[str, ...],
) -> ExperimentResult:
    """固定outer分割で評価し、各foldの未見データだけからOOF指標を作る。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    selected_configs: list[RegularizationConfig] = []
    inner_scores: list[float] = []
    vocabulary_counts: list[int] = []
    transformed_counts: list[int] = []
    importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        config, threshold, inner_f1 = select_regularization_and_threshold(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            parts_of_speech,
            seed=RANDOM_STATE + fold + 1,
        )
        transformer = build_text_transformer(parts_of_speech)
        transformed_train = transformer.fit_transform(X_train)
        transformed_valid = transformer.transform(X_valid)
        model = build_model(config)
        model.fit(transformed_train, y_train)
        valid_probabilities = model.predict_proba(transformed_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        selected_configs.append(config)
        inner_scores.append(inner_f1)
        vocabulary_counts.append(len(transformer.vectorizer_.get_feature_names_out()))
        transformed_counts.append(transformed_train.shape[1])

        coefficients = np.abs(model.named_steps["model"].coef_[0])
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": transformer.get_feature_names_out(),
                    f"fold_{fold}": coefficients,
                }
            ).set_index("feature")
        )
        print(
            f"Fold {fold}: inner F1={inner_f1:.4f}, {config.label}, "
            f"threshold={threshold:.3f}, outer F1={fold_f1:.4f}, "
            f"vocabulary={vocabulary_counts[-1]}"
        )

    importance_by_fold = pd.concat(importance_frames, axis=1).fillna(0.0)
    feature_importance = (
        importance_by_fold.mean(axis=1)
        .rename("mean_abs_coefficient")
        .sort_values(ascending=False)
        .reset_index()
    )
    nested_oof_f1 = float(f1_score(y, oof_predictions))
    return ExperimentResult(
        y=y,
        oof_predictions=oof_predictions,
        fold_scores=fold_scores,
        fold_thresholds=fold_thresholds,
        selected_configs=selected_configs,
        inner_scores=inner_scores,
        vocabulary_counts=vocabulary_counts,
        transformed_counts=transformed_counts,
        feature_importance=feature_importance,
        nested_oof_f1=nested_oof_f1,
        final_threshold=float(np.mean(fold_thresholds)),
        excluded_all_missing=[],
    )


def run_experiment(train_path: Path, parts_of_speech: tuple[str, ...]) -> ExperimentResult:
    X, y, excluded = load_data(train_path)
    result = run_nested_cv(X, y, parts_of_speech)
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
    """全SVD特徴量を平均絶対係数の降順で示す。"""
    plot_data = feature_importance.sort_values("mean_abs_coefficient", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(8.0, 0.38 * len(plot_data) + 2.8)))
    bars = ax.barh(
        plot_data["feature"],
        plot_data["mean_abs_coefficient"],
        color="#2563EB",
    )
    ax.bar_label(bars, fmt="%.5f", padding=3, fontsize=8)
    ax.set_title(f"{experiment_id} 特徴量重要度：{parts_label}", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "標準化後の平均絶対係数・外側5-fold平均（方向・因果を表さない）",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    ax.set_xlabel("平均絶対係数")
    ax.set_ylabel("SVD特徴量（全30列）")
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
    ax.set_title(f"{experiment_id} fold別F1：{parts_label}", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "各barは学習に未使用のouter foldを、inner CV選択の正則化・閾値で評価",
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
    print(f"Selected regularization: {[c.label for c in result.selected_configs]}")
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
