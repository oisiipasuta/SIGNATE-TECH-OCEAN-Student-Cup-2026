"""exp_tree_of_corp系列で共有する特徴量生成、評価、描画処理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.all_features_v1 import INDUSTRY_COLUMN  # noqa: E402
from calc_features.all_features_v3 import AllFeaturesV3Transformer  # noqa: E402
from calc_features.tree_of_corp import (  # noqa: E402
    TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS,
    NormalizedTreeOfCorpTransformer,
    TreeOfCorpTransformer,
    calculate_tree_of_corp_features,
    calculate_tree_of_corp_normalized_features,
)


TARGET_COLUMN = "購入フラグ"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
BASELINE_NESTED_OOF_F1 = 0.7744

EXP07_TREE_FEATURE_COLUMNS = (
    "DX変革組織有無",
    "平均分岐数",
    "第一階層組織数",
    "デジタル組織数",
    "組織ノード数",
    "デジタル組織比率",
)

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)

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

LIGHTGBM_NAME_TRANSLATION = str.maketrans(
    {
        "[": "［",
        "]": "］",
        "{": "｛",
        "}": "｝",
        '"': "”",
        ":": "：",
        ",": "・",
    }
)


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    experiment_name: str
    variant: int


@dataclass
class FittedFoldModel:
    feature_builder: "TreeExperimentFeatureBuilder"
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


class TreeExperimentFeatureBuilder(BaseEstimator, TransformerMixin):
    """組織図単独またはall_features_v3との混合特徴をfold単位で生成する。"""

    def __init__(self, variant: int, *, random_state: int = RANDOM_STATE) -> None:
        self.variant = variant
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TreeExperimentFeatureBuilder":
        del y
        if self.variant not in {1, 2, 3, 4, 5, 6, 7}:
            raise ValueError(f"variantは1～7である必要があります: {self.variant}")

        if self.variant == 4:
            self.tree_transformer_ = NormalizedTreeOfCorpTransformer()
        else:
            self.tree_transformer_ = TreeOfCorpTransformer(
                include_artifact=self.variant in {3, 6}
            )
        self.tree_transformer_.fit(X)
        if self.variant == 7:
            self.normalized_tree_transformer_ = NormalizedTreeOfCorpTransformer()
            self.normalized_tree_transformer_.fit(X)
        if self.variant >= 2:
            self.base_transformer_ = AllFeaturesV3Transformer(
                random_state=self.random_state
            )
            self.base_transformer_.fit(X)

        generated = self._transform_parts(X)
        if generated.columns.duplicated().any():
            duplicated = generated.columns[generated.columns.duplicated()].tolist()
            raise ValueError(f"生成特徴量名が重複しています: {duplicated}")
        self.feature_names_out_ = generated.columns.tolist()
        return self

    def _transform_parts(self, X: pd.DataFrame) -> pd.DataFrame:
        tree_features = self.tree_transformer_.transform(X)
        if self.variant == 1:
            return tree_features
        if self.variant == 5:
            tree_features = tree_features.loc[:, ["デジタル組織数"]]
        elif self.variant == 6:
            tree_features = tree_features.loc[:, list(TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS)]
        elif self.variant == 7:
            normalized_features = self.normalized_tree_transformer_.transform(X)
            tree_features = pd.concat([tree_features, normalized_features], axis=1)
            tree_features = tree_features.loc[:, list(EXP07_TREE_FEATURE_COLUMNS)]
        return pd.concat([self.base_transformer_.transform(X), tree_features], axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["tree_transformer_", "feature_names_out_"],
        )
        generated = self._transform_parts(X)
        return generated.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


# ==================================================
# 2. データ読み込み・実データ関連分析
# ==================================================

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"train.csvに{TARGET_COLUMN}列がありません")
    return train.drop(columns=[TARGET_COLUMN]), test, train[TARGET_COLUMN].astype(int)


def _standardized_mean_difference(train: pd.Series, test: pd.Series) -> float:
    pooled_std = float(np.sqrt((train.var(ddof=1) + test.var(ddof=1)) / 2))
    if not np.isfinite(pooled_std) or pooled_std == 0:
        return 0.0
    return float((test.mean() - train.mean()) / pooled_std)


def print_association_profile(
    train: pd.DataFrame,
    test: pd.DataFrame,
    y: pd.Series,
) -> None:
    """組織図特徴の相関、購入率、欠損、train/test差を標準出力へ表示する。"""
    train_features = calculate_tree_of_corp_features(train, include_artifact=True)
    test_features = calculate_tree_of_corp_features(test, include_artifact=True)
    train_features = pd.concat(
        [train_features, calculate_tree_of_corp_normalized_features(train)], axis=1
    )
    test_features = pd.concat(
        [test_features, calculate_tree_of_corp_normalized_features(test)], axis=1
    )
    records: list[dict[str, object]] = []
    for column in train_features:
        train_values = train_features[column]
        test_values = test_features[column]
        unique_values = set(train_values.dropna().unique().tolist())
        rate_without = float(y[train_values.eq(0)].mean()) if 0 in unique_values else np.nan
        rate_with = float(y[train_values.eq(1)].mean()) if unique_values <= {0.0, 1.0} else np.nan
        records.append(
            {
                "feature": column,
                "pearson": train_values.corr(y, method="pearson"),
                "spearman": train_values.corr(y, method="spearman"),
                "purchase_rate_0": rate_without,
                "purchase_rate_1": rate_with,
                "train_missing": train_values.isna().mean(),
                "test_missing": test_values.isna().mean(),
                "test_smd": _standardized_mean_difference(train_values, test_values),
            }
        )
    profile = pd.DataFrame(records).sort_values(
        "pearson", key=lambda values: values.abs(), ascending=False
    )
    print("\n組織図特徴の実データ関連分析（相関は因果を意味しない）")
    print(profile.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    correlations = train_features.corr().abs()
    high_pairs: list[tuple[str, str, float]] = []
    for index, first in enumerate(correlations.columns):
        for second in correlations.columns[index + 1 :]:
            value = float(correlations.loc[first, second])
            if value >= 0.85:
                high_pairs.append((first, second, value))
    print("\n絶対相関0.85以上の特徴ペア:")
    if high_pairs:
        for first, second, value in sorted(high_pairs, key=lambda row: -row[2]):
            print(f"- {first} / {second}: {value:.4f}")
    else:
        print("- なし")


# ==================================================
# 3. 特徴量前処理
# ==================================================

def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column == INDUSTRY_COLUMN]
    numeric = [column for column in columns if column not in categorical]
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric:
        transformers.append(("numeric", SimpleImputer(strategy="median"), numeric))
    if categorical:
        transformers.append(
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
            )
        )
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def _sanitize_feature_names(names: list[str]) -> list[str]:
    sanitized = [name.translate(LIGHTGBM_NAME_TRANSLATION) for name in names]
    if len(set(sanitized)) != len(sanitized):
        raise ValueError("LightGBM用の安全化後に特徴量名が重複しました")
    return sanitized


# ==================================================
# 4. LightGBM・inner CV閾値選択
# ==================================================

def fit_fold_model(
    X: pd.DataFrame,
    y: pd.Series,
    variant: int,
) -> FittedFoldModel:
    feature_builder = TreeExperimentFeatureBuilder(variant, random_state=RANDOM_STATE)
    generated = feature_builder.fit_transform(X)
    excluded = [column for column in generated if generated[column].isna().all()]
    kept_columns = [column for column in generated if column not in excluded]
    generated = generated.loc[:, kept_columns]

    preprocessor = build_preprocessor(kept_columns)
    values = preprocessor.fit_transform(generated)
    transformed_names = _sanitize_feature_names(
        preprocessor.get_feature_names_out().tolist()
    )
    transformed = pd.DataFrame(
        np.asarray(values),
        index=generated.index,
        columns=transformed_names,
    )
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(transformed, y)
    return FittedFoldModel(
        feature_builder,
        kept_columns,
        preprocessor,
        model,
        transformed_names,
    )


def predict_probabilities(fitted: FittedFoldModel, X: pd.DataFrame) -> np.ndarray:
    generated = fitted.feature_builder.transform(X).loc[:, fitted.kept_columns]
    values = fitted.preprocessor.transform(generated)
    transformed = pd.DataFrame(
        np.asarray(values),
        index=generated.index,
        columns=fitted.transformed_feature_names,
    )
    return fitted.model.predict_proba(transformed)[:, 1]


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
    tied_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        tied_indices,
        key=lambda index: (abs(THRESHOLD_CANDIDATES[index] - 0.5), index),
    )
    return float(THRESHOLD_CANDIDATES[best_index]), best_score


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    variant: int,
    seed: int,
) -> tuple[float, float]:
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    inner_probabilities = np.zeros(len(X), dtype=float)
    for train_index, valid_index in inner_cv.split(X, y):
        fitted = fit_fold_model(
            X.iloc[train_index],
            y.iloc[train_index],
            variant,
        )
        inner_probabilities[valid_index] = predict_probabilities(
            fitted,
            X.iloc[valid_index],
        )
    return select_threshold(y, inner_probabilities)


# ==================================================
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    spec: ExperimentSpec,
) -> ExperimentResult:
    outer_cv = StratifiedKFold(
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

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        threshold, inner_f1 = calculate_inner_threshold(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            variant=spec.variant,
            seed=RANDOM_STATE + fold + 1,
        )
        fitted = fit_fold_model(X_train, y_train, spec.variant)
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
            f"{spec.experiment_id} fold {fold}: inner F1={inner_f1:.4f}, "
            f"threshold={threshold:.3f}, outer F1={fold_f1:.4f}, "
            f"generated={len(all_generated)}, "
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


def make_feature_importance_figure(
    importance: pd.DataFrame,
    spec: ExperimentSpec,
) -> plt.Figure:
    plot_data = importance.sort_values("importance", ascending=True)
    figure, axis = plt.subplots(
        figsize=(12, max(7.0, 0.42 * len(plot_data) + 3.0))
    )
    bars = axis.barh(plot_data["feature"], plot_data["importance"], color="#2563EB")
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.set_title(
        f"{spec.experiment_id} {spec.experiment_name} 特徴量重要度",
        fontsize=16,
        pad=32,
    )
    axis.text(
        0.5,
        1.01,
        "split importance・outer 5-fold平均（利用頻度であり、方向や因果を示さない）",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    axis.set_xlabel("平均 split importance")
    axis.set_ylabel(f"変換後特徴量（全{len(plot_data)}列）")
    axis.grid(axis="x", alpha=0.25)
    axis.set_axisbelow(True)
    axis.margins(x=0.16)
    figure.tight_layout()
    return figure


def make_f1_figure(result: ExperimentResult, spec: ExperimentSpec) -> plt.Figure:
    folds = np.arange(OUTER_N_SPLITS)
    fold_mean = float(np.mean(result.fold_scores))
    fold_std = float(np.std(result.fold_scores))
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(folds, result.fold_scores, color="#E07A3F", width=0.65)
    axis.bar_label(
        bars,
        labels=[f"{score:.4f}" for score in result.fold_scores],
        padding=3,
    )
    axis.axhline(
        fold_mean,
        color="#244A64",
        linestyle="--",
        linewidth=1.5,
        label=f"fold平均={fold_mean:.4f}",
    )
    axis.set_ylim(0, 1)
    axis.set_xticks(folds)
    axis.set_xlabel("Outer validation fold（0始まり）")
    axis.set_ylabel("F1")
    axis.set_title(
        f"{spec.experiment_id} {spec.experiment_name} fold別F1",
        fontsize=16,
        pad=32,
    )
    axis.text(
        0.5,
        1.01,
        "各barは学習未使用のouter foldを、inner CVだけで選んだ閾値で評価",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    axis.text(
        0.98,
        0.97,
        f"Nested OOF F1: {result.nested_oof_f1:.4f}\n"
        f"Fold std: {fold_std:.4f}\n"
        f"Final threshold: {result.final_threshold:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


def save_result_figures(
    result: ExperimentResult,
    spec: ExperimentSpec,
    result_dir: Path,
) -> str:
    result_dir.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(result.feature_importance, spec)
    f1_figure = make_f1_figure(result, spec)
    feature_figure.savefig(
        result_dir / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(result_dir / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    return font_name


def print_summary(
    X: pd.DataFrame,
    result: ExperimentResult,
    spec: ExperimentSpec,
    font_name: str,
) -> None:
    print("\n" + "=" * 96)
    print(f"Experiment: {spec.experiment_id} {spec.experiment_name}")
    print(f"Raw input columns: {X.shape[1]}")
    print(f"Generated feature counts: {result.generated_feature_counts}")
    print(f"Transformed feature counts: {result.transformed_feature_counts}")
    print(f"Excluded all-missing features: {result.excluded_all_missing}")
    print(f"Model params: {MODEL_PARAMS}")
    print(f"Fold thresholds: {[round(value, 3) for value in result.fold_thresholds]}")
    print(f"Inner F1: {[round(value, 4) for value in result.inner_scores]}")
    print(f"Fold F1: {[round(value, 4) for value in result.fold_scores]}")
    print(
        f"Fold F1 mean ± std: {np.mean(result.fold_scores):.4f} ± "
        f"{np.std(result.fold_scores):.4f}"
    )
    print(f"Nested OOF F1: {result.nested_oof_f1:.4f}")
    print(f"Final threshold: {result.final_threshold:.4f}")
    if spec.variant >= 2:
        difference = result.nested_oof_f1 - BASELINE_NESTED_OOF_F1
        print(
            f"Difference vs all_features_v3 baseline "
            f"({BASELINE_NESTED_OOF_F1:.4f}): {difference:+.4f}"
        )
    print(f"Plot font: {font_name}")
    print("=" * 96)


__all__ = [
    "BASE_DIR",
    "BASELINE_NESTED_OOF_F1",
    "EXP07_TREE_FEATURE_COLUMNS",
    "ExperimentResult",
    "ExperimentSpec",
    "TreeExperimentFeatureBuilder",
    "configure_japanese_font",
    "load_data",
    "make_f1_figure",
    "make_feature_importance_figure",
    "print_association_profile",
    "print_summary",
    "run_nested_cv",
    "save_result_figures",
    "select_threshold",
]
