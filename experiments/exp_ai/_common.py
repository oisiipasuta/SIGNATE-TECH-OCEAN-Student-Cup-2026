"""exp_ai 系列で共有するリーク防止の特徴量生成・評価・描画処理。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import font_manager
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_is_fitted


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.all_features_v1 import (  # noqa: E402
    INDUSTRY_COLUMN,
    AllFeaturesV1Transformer,
)
from calc_features.all_features_v2 import AllFeaturesV2Transformer  # noqa: E402
from calc_features.dx_outlook import (  # noqa: E402
    DXOutlookTfidfSVD,
)


TARGET_COLUMN = "購入フラグ"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"

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

CATEGORICAL_FEATURES = (INDUSTRY_COLUMN, "上場種別", "特徴")
DX_COMPONENT_COUNT = 10
EXP10_DX_COMPONENT_COUNT = 20
EXP11_DX_COMPONENT_COUNT = 5
TEXT_SVD_COMPONENT_COUNT = 10
LISTING_MIN_FREQUENCY = 0.05
OTHER_LISTING_LABEL = "その他"
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
    feature_builder: "AIStudyFeatureBuilder"
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


class CharTfidfSVD(BaseEstimator, TransformerMixin):
    """単一テキスト列を文字TF-IDFから低次元のSVD特徴量へ変換する。"""

    def __init__(
        self,
        column: str,
        *,
        n_components: int = TEXT_SVD_COMPONENT_COUNT,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.column = column
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CharTfidfSVD":
        del y
        texts = X[self.column].fillna("").astype(str)
        self.vectorizer_ = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            max_features=20_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        matrix = self.vectorizer_.fit_transform(texts)
        max_components = min(matrix.shape[0] - 1, matrix.shape[1] - 1)
        if max_components < 1:
            raise ValueError(
                f"{self.column} のTF-IDF行列がSVDに必要な大きさを満たしません: "
                f"{matrix.shape}"
            )
        self.actual_components_ = min(self.n_components, max_components)
        self.svd_ = TruncatedSVD(
            n_components=self.actual_components_,
            n_iter=5,
            random_state=self.random_state,
        )
        self.svd_.fit(matrix)
        self.feature_names_out_ = [
            f"{self.column}_char_svd_{index + 1:02d}"
            for index in range(self.actual_components_)
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["vectorizer_", "svd_", "feature_names_out_"],
        )
        texts = X[self.column].fillna("").astype(str)
        matrix = self.vectorizer_.transform(texts)
        values = self.svd_.transform(matrix)
        return pd.DataFrame(
            values,
            index=X.index,
            columns=self.feature_names_out_,
        )

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    values = numerator.astype(float).div(denominator.astype(float).replace(0, np.nan))
    return values.replace([np.inf, -np.inf], np.nan)


def _financial_features(X: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=X.index)
    features["経常利益率"] = _safe_divide(X["経常利益"], X["売上"])
    features["純利益率"] = _safe_divide(X["当期純利益"], X["売上"])
    features["ROA"] = _safe_divide(X["当期純利益"], X["総資産"])
    features["ROE"] = _safe_divide(X["当期純利益"], X["自己資本"])
    features["流動資産比率"] = _safe_divide(X["流動資産"], X["総資産"])
    features["負債比率"] = _safe_divide(X["負債"], X["総資産"])
    features["借入依存度"] = _safe_divide(
        X["短期借入金"] + X["長期借入金"], X["負債"]
    )
    features["営業CF対営業利益"] = _safe_divide(X["営業CF"], X["営業利益"])
    features["投資CF対売上"] = _safe_divide(X["投資CF"], X["売上"])
    features["減価償却費対売上"] = _safe_divide(X["減価償却費"], X["売上"])
    features["営業利益_欠損"] = X["営業利益"].isna().astype(int)
    features["経常利益_欠損"] = X["経常利益"].isna().astype(int)
    return features


def _unused_financial_features(X: pd.DataFrame) -> pd.DataFrame:
    """v2の生成に使われていない財務列から、重複を抑えた9特徴量を作る。"""
    features = pd.DataFrame(index=X.index)
    capital = pd.to_numeric(X["資本金"], errors="coerce")
    features["log_資本金"] = np.log1p(capital.where(capital.ge(0)))
    features["流動資産比率"] = _safe_divide(X["流動資産"], X["総資産"])
    features["負債比率"] = _safe_divide(X["負債"], X["総資産"])
    features["経常利益率"] = _safe_divide(X["経常利益"], X["売上"])
    features["純利益率"] = _safe_divide(X["当期純利益"], X["売上"])
    features["減価償却費対売上"] = _safe_divide(X["減価償却費"], X["売上"])
    features["運転資本変動対売上"] = _safe_divide(X["運転資本変動"], X["売上"])
    features["投資CF対売上"] = _safe_divide(X["投資CF"], X["売上"])
    features["有形固定資産変動対売上"] = _safe_divide(
        X["有形固定資産変動"], X["売上"]
    )
    return features


def _survey_features(X: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame(index=X.index)
    features["アンケート３"] = X["アンケート３"]
    features["アンケート６"] = X["アンケート６"]
    features["アンケート１０"] = X["アンケート１０"]
    features["戦略導入ギャップ"] = X["アンケート１"] - X["アンケート３"]
    features["戦略抵抗ギャップ"] = X["アンケート１"] - X["アンケート４"]
    features["外部支援必要度"] = 6 - X["アンケート１０"]
    features["ツール未導入フラグ"] = (X["アンケート６"] == 2).astype(int)
    return features


def _unused_survey_features(X: pd.DataFrame) -> pd.DataFrame:
    """v2の出力に含まれないアンケート3・6・10だけを追加する。"""
    features = pd.DataFrame(index=X.index)
    for column in ("アンケート３", "アンケート６", "アンケート１０"):
        values = pd.to_numeric(X[column], errors="coerce")
        features[column] = values.where(values.between(1, 5))
    return features


class AIStudyFeatureBuilder(BaseEstimator, TransformerMixin):
    """既存6実験と、v2への独立追加実験をfold単位で学習・生成する。"""

    def __init__(self, variant: int, *, random_state: int = RANDOM_STATE) -> None:
        self.variant = variant
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AIStudyFeatureBuilder":
        del y
        if self.variant not in range(1, 14):
            raise ValueError(f"variantは1～13である必要があります: {self.variant}")

        if self.variant in {1, 2, 3, 4, 7, 8, 9}:
            self.base_transformer_ = AllFeaturesV2Transformer(
                random_state=self.random_state
            )
            self.base_transformer_.fit(X)
        else:
            self.base_transformer_ = AllFeaturesV1Transformer()
            self.base_transformer_.fit(X)
            if self.variant == 10:
                fitted_component_count = EXP10_DX_COMPONENT_COUNT
            elif self.variant == 11:
                fitted_component_count = EXP11_DX_COMPONENT_COUNT
            else:
                fitted_component_count = 30
            self.dx_transformer_ = DXOutlookTfidfSVD(
                n_components=fitted_component_count,
                target_parts_of_speech=("名詞", "動詞"),
                random_state=self.random_state,
            )
            self.dx_transformer_.fit(X)

        if self.variant == 7:
            frequencies = X["上場種別"].value_counts(normalize=True, dropna=True)
            self.retained_listing_types_ = frozenset(
                frequencies.index[frequencies >= LISTING_MIN_FREQUENCY].tolist()
            )

        self.text_transformers_: list[CharTfidfSVD] = []
        if self.variant == 6:
            for column in ("企業概要", "組織図"):
                transformer = CharTfidfSVD(
                    column,
                    n_components=TEXT_SVD_COMPONENT_COUNT,
                    random_state=self.random_state,
                )
                transformer.fit(X)
                self.text_transformers_.append(transformer)

        generated = self._transform_parts(X)
        if generated.columns.duplicated().any():
            duplicated = generated.columns[generated.columns.duplicated()].tolist()
            raise ValueError(f"生成特徴量名が重複しています: {duplicated}")
        self.feature_names_out_ = generated.columns.tolist()
        return self

    def _transform_parts(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.variant in {1, 2, 3, 4, 7, 8, 9}:
            parts = [self.base_transformer_.transform(X)]
        else:
            base = self.base_transformer_.transform(X)
            dx = self.dx_transformer_.transform(X)
            if self.variant == 12:
                dx_columns = [self.dx_transformer_.get_feature_names_out()[0]]
            elif self.variant == 13:
                dx_columns = [self.dx_transformer_.get_feature_names_out()[2]]
            elif self.variant == 10:
                component_count = EXP10_DX_COMPONENT_COUNT
                dx_columns = self.dx_transformer_.get_feature_names_out()[:component_count]
            elif self.variant == 11:
                component_count = EXP11_DX_COMPONENT_COUNT
                dx_columns = self.dx_transformer_.get_feature_names_out()[:component_count]
            else:
                component_count = DX_COMPONENT_COUNT
                dx_columns = self.dx_transformer_.get_feature_names_out()[:component_count]
            parts = [base, dx.loc[:, dx_columns]]

        if 2 <= self.variant <= 6:
            parts.append(X.loc[:, ["上場種別", "特徴"]].copy())
        if 3 <= self.variant <= 6:
            parts.append(_financial_features(X))
        if 4 <= self.variant <= 6:
            parts.append(_survey_features(X))
        if self.variant == 6:
            parts.extend(transformer.transform(X) for transformer in self.text_transformers_)
        if self.variant == 7:
            listing = X["上場種別"].astype("string")
            grouped = listing.where(
                listing.isna() | listing.isin(self.retained_listing_types_),
                OTHER_LISTING_LABEL,
            )
            parts.append(grouped.to_frame("上場種別"))
        if self.variant == 8:
            parts.append(_unused_financial_features(X))
        if self.variant == 9:
            parts.append(_unused_survey_features(X))

        return pd.concat(parts, axis=1)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["base_transformer_", "feature_names_out_"])
        generated = self._transform_parts(X)
        return generated.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def load_data(train_path: Path = TRAIN_PATH) -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(train_path)
    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"train.csvに{TARGET_COLUMN}列がありません")
    X = train.drop(columns=[TARGET_COLUMN])
    y = train[TARGET_COLUMN].astype(int)
    return X, y


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column in CATEGORICAL_FEATURES]
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
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        verbose_feature_names_out=False,
    )


def _sanitize_feature_names(names: list[str]) -> list[str]:
    """LightGBMが拒否するJSON記号だけを可読な全角記号へ置換する。"""
    sanitized = [name.translate(LIGHTGBM_NAME_TRANSLATION) for name in names]
    if len(set(sanitized)) != len(sanitized):
        duplicates = sorted(
            {name for name in sanitized if sanitized.count(name) > 1}
        )
        raise ValueError(f"安全化後の特徴量名が重複しています: {duplicates}")
    return sanitized


def fit_fold_model(
    X: pd.DataFrame,
    y: pd.Series,
    variant: int,
) -> FittedFoldModel:
    feature_builder = AIStudyFeatureBuilder(variant, random_state=RANDOM_STATE)
    generated = feature_builder.fit_transform(X)
    excluded = [column for column in generated if generated[column].isna().all()]
    kept_columns = [column for column in generated if column not in excluded]
    generated = generated.loc[:, kept_columns]

    preprocessor = build_preprocessor(kept_columns)
    transformed_values = preprocessor.fit_transform(generated)
    transformed_names = _sanitize_feature_names(
        preprocessor.get_feature_names_out().tolist()
    )
    transformed = pd.DataFrame(
        np.asarray(transformed_values),
        index=generated.index,
        columns=transformed_names,
    )
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(transformed, y)
    return FittedFoldModel(
        feature_builder=feature_builder,
        kept_columns=kept_columns,
        preprocessor=preprocessor,
        model=model,
        transformed_feature_names=transformed_names,
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
    tied = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        tied,
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
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        threshold, inner_f1 = calculate_inner_threshold(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            variant=spec.variant,
            seed=RANDOM_STATE + fold + 1,
        )
        fitted = fit_fold_model(X_train, y_train, spec.variant)
        valid_probabilities = predict_probabilities(fitted, X_valid)
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        all_generated = fitted.feature_builder.get_feature_names_out()
        excluded = set(all_generated) - set(fitted.kept_columns)
        oof_predictions[valid_index] = valid_predictions
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

    feature_importance = (
        pd.concat(importance_frames, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .reset_index()
    )
    return ExperimentResult(
        fold_scores=fold_scores,
        fold_thresholds=fold_thresholds,
        inner_scores=inner_scores,
        generated_feature_counts=generated_counts,
        transformed_feature_counts=transformed_counts,
        excluded_all_missing=sorted(excluded_features),
        feature_importance=feature_importance,
        nested_oof_f1=float(f1_score(y, oof_predictions)),
        final_threshold=float(np.mean(fold_thresholds)),
    )


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
    spec: ExperimentSpec,
) -> plt.Figure:
    plot_data = feature_importance.sort_values("importance", ascending=True)
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
    axis.margins(x=0.15)
    figure.tight_layout()
    return figure


def make_f1_figure(result: ExperimentResult, spec: ExperimentSpec) -> plt.Figure:
    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(result.fold_scores))
    fold_std = float(np.std(result.fold_scores))
    figure, axis = plt.subplots(figsize=(10, 6))
    bars = axis.bar(folds, result.fold_scores, color="#E07A3F", width=0.65)
    axis.bar_label(
        bars,
        labels=[f"{score:.4f}" for score in result.fold_scores],
        padding=3,
    )
    axis.axhline(mean_score, color="#244A64", linestyle="--", linewidth=1.5)
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
        f"Fold mean: {mean_score:.4f}\nFold std: {fold_std:.4f}",
        transform=axis.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    axis.grid(axis="y", alpha=0.2)
    axis.set_axisbelow(True)
    figure.tight_layout()
    return figure


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
    print(f"Plot font: {font_name}")
    print("=" * 96)


__all__ = [
    "BASE_DIR",
    "MODEL_PARAMS",
    "ExperimentResult",
    "ExperimentSpec",
    "configure_japanese_font",
    "load_data",
    "make_f1_figure",
    "make_feature_importance_figure",
    "print_summary",
    "run_nested_cv",
]
