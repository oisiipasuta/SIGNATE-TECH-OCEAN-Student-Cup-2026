"""
実験ID: exp01
実験名: MeCabを使わない文字N-gram TF-IDFベースライン
著者: Codex

目的・仮説:
- 「今後のDX展望」を単語分割せず、文字2〜5-gramのTF-IDFに変換する。
- 日本語の部分文字列だけで、MeCabを使う後続実験と比較できる強い基準を作る。
- 高次元疎行列に対し、ロジスティック回帰のL2・L1と正則化強度を比較する。

特徴量・前処理:
- 入力: train/testで共通する文字列列のうち平均文字数が最大の自由記述列
  （現データでは「今後のDX展望」）。全欠損列だけを除外する。
- TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2,
  max_features=5000, sublinear_tf=True)。欠損文は空文字に置換する。
- TF-IDFは各CV学習foldだけでfitする。MeCabや単語特徴量は使わない。

モデル・正則化候補:
- LogisticRegression(solver="liblinear", max_iter=2000)。
- L2: C=0.001, 0.01, 0.1, 1, 10
- L1: C=0.01, 0.1, 1, 10
- 各outer fold内のinner OOF F1だけで、正則化候補と分類閾値を同時に選ぶ。

評価設計:
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 閾値候補は0.05〜0.95を0.005刻み。outer validationは選択に使わない。
- 最終thresholdはouter foldで選ばれた閾値の平均。
- 重要度は各outerモデルの絶対係数を特徴名で揃えて平均した値であり、因果や方向を表さない。

出力:
- experiments/exp_dx_outlook/results/exp01/feature_importance.png
- experiments/exp_dx_outlook/results/exp01/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-09）:
- 入力特徴量数: 1（今後のDX展望）
- TF-IDF後特徴量数: 各outer fold 5000、fold間union 5446
- 除外した全欠損特徴量: なし
- fold選択正則化: L2/C=0.1, L2/C=1, L2/C=0.1, L2/C=1, L2/C=0.1
- fold threshold: 0.290, 0.350, 0.290, 0.325, 0.295
- fold F1: 0.6944, 0.5432, 0.5797, 0.4944, 0.5556
- fold F1 mean ± std: 0.5735 ± 0.0666
- nested OOF F1: 0.5692
- 最終threshold: 0.3100
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp01"
EXPERIMENT_NAME = "MeCabを使わない文字N-gram TF-IDFベースライン"
BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook" / "results" / EXPERIMENT_ID

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)

TFIDF_PARAMS = {
    "analyzer": "char",
    "ngram_range": (2, 5),
    "min_df": 2,
    "max_features": 5000,
    "sublinear_tf": True,
    "dtype": np.float64,
}


@dataclass(frozen=True)
class RegularizationConfig:
    label: str
    c: float
    l1_ratio: float


REGULARIZATION_CONFIGS = [
    *[
        RegularizationConfig(f"L2 / C={c:g}", c, 0.0)
        for c in (0.001, 0.01, 0.1, 1.0, 10.0)
    ],
    *[
        RegularizationConfig(f"L1 / C={c:g}", c, 1.0)
        for c in (0.01, 0.1, 1.0, 10.0)
    ],
]


# ==================================================
# 2. データ読み込み・列解決
# ==================================================

def resolve_target_column(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [column for column in train.columns if column not in test.columns]
    if len(train_only) != 1:
        raise ValueError(f"trainだけに存在する目的変数列が1列ではありません: {train_only}")
    return train_only[0]


def resolve_text_column(train: pd.DataFrame, test: pd.DataFrame) -> str:
    shared = [column for column in train.columns if column in test.columns]
    text_columns = [
        column
        for column in shared
        if pd.api.types.is_string_dtype(train[column].dtype)
    ]
    if not text_columns:
        raise ValueError("train/testに共通する文字列列がありません。")
    mean_lengths = {
        column: train[column].fillna("").astype(str).str.len().mean()
        for column in text_columns
    }
    return max(mean_lengths, key=mean_lengths.get)


def load_data() -> tuple[pd.Series, pd.Series, pd.Series, str, str, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    target_column = resolve_target_column(train, test)
    text_column = resolve_text_column(train, test)
    excluded_all_missing = (
        [text_column] if train[text_column].isna().all() else []
    )
    if excluded_all_missing:
        raise ValueError(f"対象テキスト列が全欠損です: {excluded_all_missing}")
    X = train[text_column].fillna("").astype(str)
    X_test = test[text_column].fillna("").astype(str)
    y = train[target_column].astype(int)
    return X, X_test, y, target_column, text_column, excluded_all_missing


# ==================================================
# 3. 特徴量前処理
# ==================================================

def build_pipeline(config: RegularizationConfig) -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**TFIDF_PARAMS)),
            ("model", build_model(config)),
        ]
    )


def build_model(config: RegularizationConfig) -> LogisticRegression:
    return LogisticRegression(
        C=config.c,
        l1_ratio=config.l1_ratio,
        solver="liblinear",
        max_iter=2000,
        tol=1e-3,
        random_state=RANDOM_STATE,
    )


# ==================================================
# 4. モデル・正則化選択
# ==================================================

def select_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    scores = np.array(
        [f1_score(y_true, probabilities >= threshold) for threshold in THRESHOLD_CANDIDATES]
    )
    best_score = float(scores.max())
    best_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        best_indices,
        key=lambda index: (abs(THRESHOLD_CANDIDATES[index] - 0.5), index),
    )
    return float(THRESHOLD_CANDIDATES[best_index]), best_score


def select_regularization_and_threshold(
    X: pd.Series,
    y: pd.Series,
    seed: int,
) -> tuple[RegularizationConfig, float, float]:
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probabilities_by_config = np.zeros(
        (len(REGULARIZATION_CONFIGS), len(X)), dtype=float
    )
    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        X_inner_train = vectorizer.fit_transform(X.iloc[inner_train_index])
        X_inner_valid = vectorizer.transform(X.iloc[inner_valid_index])
        for config_index, config in enumerate(REGULARIZATION_CONFIGS):
            model = build_model(config)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X_inner_train, y.iloc[inner_train_index])
            probabilities_by_config[config_index, inner_valid_index] = (
                model.predict_proba(X_inner_valid)[:, 1]
            )

    best: tuple[float, int, RegularizationConfig, float] | None = None
    for config_index, config in enumerate(REGULARIZATION_CONFIGS):
        inner_probabilities = probabilities_by_config[config_index]
        threshold, score = select_threshold(y, inner_probabilities)
        candidate = (score, -config_index, config, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("正則化候補を選択できませんでした。")
    return best[2], best[3], best[0]


# ==================================================
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_nested_cv(
    X: pd.Series,
    y: pd.Series,
) -> tuple[
    np.ndarray,
    list[float],
    list[float],
    list[RegularizationConfig],
    list[int],
    pd.DataFrame,
]:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_probabilities = np.zeros(len(X), dtype=float)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    fold_configs: list[RegularizationConfig] = []
    transformed_counts: list[int] = []
    importance_frames: list[pd.DataFrame] = []

    for fold, (outer_train_index, outer_valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[outer_train_index].reset_index(drop=True)
        y_train = y.iloc[outer_train_index].reset_index(drop=True)
        X_valid = X.iloc[outer_valid_index]
        y_valid = y.iloc[outer_valid_index]

        config, threshold, inner_f1 = select_regularization_and_threshold(
            X_train, y_train, RANDOM_STATE + fold + 1
        )
        pipeline = build_pipeline(config)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)
        oof_probabilities[outer_valid_index] = valid_probabilities
        fold_scores.append(float(fold_f1))
        fold_thresholds.append(threshold)
        fold_configs.append(config)

        vectorizer = pipeline.named_steps["tfidf"]
        model = pipeline.named_steps["model"]
        feature_names = vectorizer.get_feature_names_out()
        transformed_counts.append(len(feature_names))
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": feature_names,
                    f"fold_{fold}": np.abs(model.coef_[0]),
                }
            ).set_index("feature")
        )
        print(
            f"Fold {fold}: inner F1={inner_f1:.4f}, {config.label}, "
            f"threshold={threshold:.3f}, outer F1={fold_f1:.4f}, "
            f"features={len(feature_names)}",
            flush=True,
        )

    feature_importance = pd.concat(importance_frames, axis=1).fillna(0.0)
    feature_importance["mean_abs_coefficient"] = feature_importance.mean(axis=1)
    feature_importance = feature_importance.sort_values(
        "mean_abs_coefficient", ascending=False
    )
    return (
        oof_probabilities,
        fold_scores,
        fold_thresholds,
        fold_configs,
        transformed_counts,
        feature_importance,
    )


# ==================================================
# 6. 実験結果・可視化
# ==================================================

def configure_japanese_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/NotoSansJP-VF.ttf"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/meiryo.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    values = feature_importance["mean_abs_coefficient"].sort_values(ascending=True)
    count = len(values)
    figure_height = min(480.0, max(12.0, count * 0.060))
    fig, ax = plt.subplots(figsize=(18, figure_height))
    positions = np.arange(count)
    ax.barh(positions, values.to_numpy(), color="#2878B5")
    ax.set_yticks(positions)
    ax.set_yticklabels(values.index, fontsize=3.2)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("平均絶対係数（重要度）")
    ax.set_title(
        "文字N-gramの完全な重要度順位\n"
        "各outer foldのロジスティック回帰 | 絶対係数を特徴名で整列して平均"
    )
    offset = max(float(values.max()) * 0.004, 1e-8)
    for position, value in zip(positions, values.to_numpy()):
        ax.text(value + offset, position, f"{value:.6f}", va="center", fontsize=3.0)
    ax.set_xlim(0, float(values.max()) * 1.16 if values.max() > 0 else 1.0)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(
    fold_scores: list[float],
    nested_oof_f1: float,
    fold_std: float,
) -> None:
    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(fold_scores))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(folds, fold_scores, color="#E07A3F", width=0.65)
    ax.axhline(mean_score, color="#244A64", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(folds)
    ax.set_xlabel("Outer validation fold（0始まり）")
    ax.set_ylabel("F1")
    ax.set_title(
        "文字N-gram Logistic Regression: outer-fold F1\n"
        "各barは学習に未使用のouter foldを、inner CV選択の正則化・閾値で評価"
    )
    for bar, score in zip(bars, fold_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score - 0.025,
            f"{score:.4f}",
            ha="center",
            va="top",
            color="white",
            fontweight="bold",
        )
    ax.text(
        0.98,
        0.97,
        f"Nested OOF F1: {nested_oof_f1:.4f}\nFold std: {fold_std:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.text(
        OUTER_N_SPLITS - 0.55,
        mean_score + 0.015,
        f"fold mean = {mean_score:.4f}",
        ha="right",
        color="#244A64",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=160)
    plt.close(fig)


def main() -> None:
    configure_japanese_font()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    X, _X_test, y, target_column, text_column, excluded_all_missing = load_data()
    (
        oof_probabilities,
        fold_scores,
        fold_thresholds,
        fold_configs,
        transformed_counts,
        feature_importance,
    ) = run_nested_cv(X, y)

    oof_predictions = np.zeros(len(y), dtype=int)
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    for fold, (_, valid_index) in enumerate(outer_cv.split(X, y)):
        oof_predictions[valid_index] = (
            oof_probabilities[valid_index] >= fold_thresholds[fold]
        ).astype(int)

    nested_oof_f1 = float(f1_score(y, oof_predictions))
    fold_mean = float(np.mean(fold_scores))
    fold_std = float(np.std(fold_scores))
    final_threshold = float(np.mean(fold_thresholds))
    plot_feature_importance(feature_importance)
    plot_f1_scores(fold_scores, nested_oof_f1, fold_std)

    print()
    print("=" * 72)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Target column: {target_column}")
    print(f"Text column: {text_column}")
    print("Raw feature count: 1")
    print(f"TF-IDF feature counts by fold: {transformed_counts}")
    print(f"Union importance feature count: {len(feature_importance)}")
    print(f"Excluded all-missing features: {excluded_all_missing}")
    print(f"Selected regularization: {[config.label for config in fold_configs]}")
    print(f"Fold thresholds: {[round(value, 3) for value in fold_thresholds]}")
    print(f"Fold F1: {[round(value, 4) for value in fold_scores]}")
    print(f"Fold F1 mean ± std: {fold_mean:.4f} ± {fold_std:.4f}")
    print(f"Nested OOF F1: {nested_oof_f1:.4f}")
    print(f"Final threshold: {final_threshold:.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
