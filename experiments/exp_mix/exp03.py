"""
実験ID: exp03
実験名: all_features_v3 混合特徴量
著者: Codex

仮説:
- exp_ai/exp09で改善した未使用アンケート3・6・10をall_features_v2へ加えた
  all_features_v3を使うと、exp_mix/exp02より高いF1を再現できる。

特徴量・前処理:
- calc_features.all_features_v3.AllFeaturesV3Transformerを各fold内でfitする。
- all_features_v2の20列 + アンケート3・6・10の計23列。
- 数値列は学習foldの中央値で補完する。
- 業界は学習foldだけで補完・One-Hot Encodingし、未知カテゴリは無視する。
- 学習foldで全欠損の列だけを除外する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。

出力:
- experiments/exp_mix/results/exp03/feature_importance.png
- experiments/exp_mix/results/exp03/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-19）:
- 入力列数42、生成特徴量数23（全outer foldで共通）。
- One-Hot後特徴量数: 36, 35, 37, 37, 36。全欠損除外: なし。
- fold threshold: 0.415, 0.255, 0.205, 0.445, 0.310。
- fold F1: 0.7077, 0.8493, 0.7561, 0.7647, 0.7887。
- fold F1 mean ± std: 0.7733 ± 0.0462。
- nested OOF F1: 0.7744、最終threshold: 0.3260。
- exp_ai/exp09の特徴量、fold閾値、fold F1、nested OOF F1を完全再現した。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import font_manager
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.all_features_v1 import INDUSTRY_COLUMN  # noqa: E402
from calc_features.all_features_v3 import AllFeaturesV3Transformer  # noqa: E402


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp03"
EXPERIMENT_NAME = "all_features_v3 混合特徴量"
TARGET_COLUMN = "購入フラグ"
MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "split",
}
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_mix" / "results" / EXPERIMENT_ID


@dataclass
class FittedFoldModel:
    feature_builder: AllFeaturesV3Transformer
    kept_columns: list[str]
    preprocessor: ColumnTransformer
    model: LGBMClassifier
    transformed_feature_names: list[str]


@dataclass
class ExperimentResult:
    fold_scores: list[float]
    fold_thresholds: list[float]
    inner_scores: list[float]
    generated_feature_counts: list[int]
    transformed_feature_counts: list[int]
    excluded_all_missing: list[str]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_threshold: float


# ==================================================
# 2. データ読み込み
# ==================================================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    X = train.drop(columns=[TARGET_COLUMN])
    y = train[TARGET_COLUMN].astype(int)
    return X, y


# ==================================================
# 3. 特徴量前処理
# ==================================================

def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column == INDUSTRY_COLUMN]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def fit_fold_model(X: pd.DataFrame, y: pd.Series) -> FittedFoldModel:
    feature_builder = AllFeaturesV3Transformer(random_state=RANDOM_STATE)
    generated = feature_builder.fit_transform(X)
    excluded = [column for column in generated if generated[column].isna().all()]
    kept_columns = [column for column in generated if column not in excluded]
    generated = generated.loc[:, kept_columns]

    preprocessor = build_preprocessor(kept_columns)
    values = preprocessor.fit_transform(generated)
    names = preprocessor.get_feature_names_out().tolist()
    transformed = pd.DataFrame(np.asarray(values), index=generated.index, columns=names)
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(transformed, y)
    return FittedFoldModel(feature_builder, kept_columns, preprocessor, model, names)


def predict_probabilities(fitted: FittedFoldModel, X: pd.DataFrame) -> np.ndarray:
    generated = fitted.feature_builder.transform(X).loc[:, fitted.kept_columns]
    values = fitted.preprocessor.transform(generated)
    transformed = pd.DataFrame(
        np.asarray(values), index=generated.index, columns=fitted.transformed_feature_names
    )
    return fitted.model.predict_proba(transformed)[:, 1]


# ==================================================
# 4. LightGBM・閾値選択
# ==================================================

def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    scores = np.array(
        [
            f1_score(y_true, (probabilities >= threshold).astype(int))
            for threshold in THRESHOLD_CANDIDATES
        ]
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
    seed: int,
) -> tuple[float, float]:
    cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probabilities = np.zeros(len(X), dtype=float)
    for train_index, valid_index in cv.split(X, y):
        fitted = fit_fold_model(X.iloc[train_index], y.iloc[train_index])
        probabilities[valid_index] = predict_probabilities(fitted, X.iloc[valid_index])
    return select_threshold(y, probabilities)


# ==================================================
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_nested_cv(X: pd.DataFrame, y: pd.Series) -> ExperimentResult:
    cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    generated_counts: list[int] = []
    transformed_counts: list[int] = []
    excluded_features: set[str] = set()
    importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        threshold, inner_f1 = calculate_inner_threshold(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            RANDOM_STATE + fold + 1,
        )
        fitted = fit_fold_model(X_train, y_train)
        probabilities = predict_probabilities(fitted, X_valid)
        predictions = (probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, predictions))

        all_generated = fitted.feature_builder.get_feature_names_out()
        excluded = set(all_generated) - set(fitted.kept_columns)
        oof_predictions[valid_index] = predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        generated_counts.append(len(all_generated))
        transformed_counts.append(len(fitted.transformed_feature_names))
        excluded_features.update(excluded)
        importance_frames.append(
            pd.DataFrame(
                {
                    "feature": fitted.transformed_feature_names,
                    f"fold_{fold}": fitted.model.feature_importances_,
                }
            ).set_index("feature")
        )
        print(
            f"exp03 fold {fold}: inner F1={inner_f1:.4f}, threshold={threshold:.3f}, "
            f"outer F1={fold_f1:.4f}, generated={len(all_generated)}, "
            f"transformed={len(fitted.transformed_feature_names)}",
            flush=True,
        )

    importance = (
        pd.concat(importance_frames, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .reset_index()
    )
    return ExperimentResult(
        fold_scores,
        fold_thresholds,
        inner_scores,
        generated_counts,
        transformed_counts,
        sorted(excluded_features),
        importance,
        float(f1_score(y, oof_predictions)),
        float(np.mean(fold_thresholds)),
    )


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def configure_japanese_font() -> str:
    for path in (
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/meiryo.ttc"),
        Path("C:/Windows/Fonts/msgothic.ttc"),
    ):
        if path.exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return str(plt.rcParams["font.family"])


def plot_feature_importance(data: pd.DataFrame) -> None:
    plot_data = data.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(7.0, 0.42 * len(plot_data) + 3.0)))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color="#2563EB")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("exp03 all_features_v3 特徴量重要度", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "split importance・outer 5-fold平均（利用頻度であり、方向や因果を示さない）",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    ax.set_xlabel("平均 split importance")
    ax.set_ylabel(f"変換後特徴量（全{len(plot_data)}列）")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(result: ExperimentResult) -> None:
    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(result.fold_scores))
    fold_std = float(np.std(result.fold_scores))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(folds, result.fold_scores, color="#E07A3F", width=0.65)
    ax.bar_label(bars, labels=[f"{score:.4f}" for score in result.fold_scores], padding=3)
    ax.axhline(mean_score, color="#244A64", linestyle="--", linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(folds)
    ax.set_xlabel("Outer validation fold（0始まり）")
    ax.set_ylabel("F1")
    ax.set_title("exp03 all_features_v3 fold別F1", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "各barは学習未使用のouter foldを、inner CVだけで選んだ閾値で評価",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    ax.text(
        0.98,
        0.97,
        f"Nested OOF F1: {result.nested_oof_f1:.4f}\n"
        f"Fold mean: {mean_score:.4f}\nFold std: {fold_std:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    X, y = load_data()
    result = run_nested_cv(X, y)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    plot_feature_importance(result.feature_importance)
    plot_f1_scores(result)

    print("\n" + "=" * 88)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Input columns: {X.shape[1]}")
    print(f"Generated feature counts: {result.generated_feature_counts}")
    print(f"Transformed feature counts: {result.transformed_feature_counts}")
    print(f"Excluded all-missing features: {result.excluded_all_missing}")
    print(f"Fold thresholds: {[round(value, 3) for value in result.fold_thresholds]}")
    print(f"Inner F1: {[round(value, 4) for value in result.inner_scores]}")
    print(f"Fold F1: {[round(value, 4) for value in result.fold_scores]}")
    print(
        f"Fold F1 mean ± std: {np.mean(result.fold_scores):.4f} ± "
        f"{np.std(result.fold_scores):.4f}"
    )
    print(f"Nested OOF F1: {result.nested_oof_f1:.4f}")
    print(f"Final threshold: {result.final_threshold:.4f}")
    print(f"Plot font: {font_name}")
    print("=" * 88)


if __name__ == "__main__":
    main()
