"""
実験ID: exp07
実験名: all_features_v3＋平均分岐数
著者: Codex

目的・仮説:
- exp05（DX変革組織有無＋平均分岐数）の改善が、平均分岐数だけでも再現するか確認する。
- exp06（DX変革組織有無だけ）との比較で、2特徴の組み合わせ効果を切り分ける。

特徴量・前処理:
- all_features_v3の23列へ平均分岐数を1列加え、計24列とする。
- AllFeaturesV3TransformerとTreeOfCorpTransformerを各fold内でfitする。
- 数値列は学習fold中央値で補完し、業界は学習fold最頻値補完後にOne-Hot化する。
- 学習foldで全欠損の列だけを除外する。TF-IDF/SVDを含む全変換はfold内で学習する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4、seed=42。
- 各outer foldの閾値は、そのouter学習部分のinner OOF F1だけから選ぶ。

出力:
- experiments/exp_mix/results/exp07/feature_importance.png
- experiments/exp_mix/results/exp07/f1_scores.png
- 実験スクリプト自体はCSV、JSON、予測値、submissionを出力しない。

実行結果:
- 入力42列から24特徴を生成。One-Hot後37/36/38/38/37列、全欠損除外なし。
- fold閾値: 0.305/0.250/0.210/0.145/0.300、最終閾値=0.2420。
- inner F1: 0.7902/0.7308/0.7432/0.7605/0.7550。
- fold F1: 0.7143/0.8378/0.7317/0.7143/0.7838、平均=0.7564、
  標準偏差=0.0480、nested OOF F1=0.7538。
- all_features_v3単独の0.7744を下回り、平均分岐数単独では改善を再現しなかった。
- exp05の0.8063は、DX変革組織有無との組み合わせ、またはCV上の相互作用に
  依存する可能性がある。特徴量重要度は因果を示さない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
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
    TREE_OF_CORP_FEATURE_COLUMNS,
    TreeOfCorpTransformer,
)
from experiments.exp_tree_of_corp._common import (  # noqa: E402
    ExperimentResult,
    ExperimentSpec,
    configure_japanese_font,
    make_f1_figure,
    make_feature_importance_figure,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp07"
EXPERIMENT_NAME = "all_features_v3＋平均分岐数"
TARGET_COLUMN = "購入フラグ"
SPEC = ExperimentSpec(EXPERIMENT_ID, EXPERIMENT_NAME, 7)
SELECTED_TREE_FEATURES = ("平均分岐数",)

MODEL_PARAMS: dict[str, Any] = {
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
LIGHTGBM_NAME_TRANSLATION = str.maketrans(
    {"[": "［", "]": "］", "{": "｛", "}": "｝", '"': "”", ":": "：", ",": "・"}
)


class AverageBranchingFeatureBuilder(BaseEstimator, TransformerMixin):
    """all_features_v3と平均分岐数をfold単位で生成する。"""

    def __init__(self, *, random_state: int = RANDOM_STATE) -> None:
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AverageBranchingFeatureBuilder":
        del y
        if not set(SELECTED_TREE_FEATURES).issubset(TREE_OF_CORP_FEATURE_COLUMNS):
            raise ValueError("選択した組織図特徴が汎用特徴スキーマにありません")
        self.base_transformer_ = AllFeaturesV3Transformer(
            random_state=self.random_state
        )
        self.tree_transformer_ = TreeOfCorpTransformer(include_artifact=False)
        base_features = self.base_transformer_.fit_transform(X)
        tree_features = self.tree_transformer_.fit_transform(X).loc[
            :, list(SELECTED_TREE_FEATURES)
        ]
        duplicated = set(base_features.columns).intersection(tree_features.columns)
        if duplicated:
            raise ValueError(f"all_features_v3と平均分岐数が重複しています: {duplicated}")
        self.feature_names_out_ = [
            *base_features.columns.tolist(),
            *tree_features.columns.tolist(),
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["base_transformer_", "tree_transformer_", "feature_names_out_"],
        )
        combined = pd.concat(
            [
                self.base_transformer_.transform(X),
                self.tree_transformer_.transform(X).loc[
                    :, list(SELECTED_TREE_FEATURES)
                ],
            ],
            axis=1,
        )
        if combined.columns.duplicated().any():
            duplicated = combined.columns[combined.columns.duplicated()].tolist()
            raise ValueError(f"exp07の特徴量名が重複しています: {duplicated}")
        return combined.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


@dataclass
class FittedFoldModel:
    feature_builder: AverageBranchingFeatureBuilder
    kept_columns: list[str]
    preprocessor: ColumnTransformer
    model: LGBMClassifier
    transformed_feature_names: list[str]


# ==================================================
# 2. データ読み込み
# ==================================================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"train.csvに{TARGET_COLUMN}列がありません")
    y = train[TARGET_COLUMN].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN}は0/1である必要があります")
    return train.drop(columns=[TARGET_COLUMN]), y


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


def _sanitize_feature_names(names: list[str]) -> list[str]:
    sanitized = [name.translate(LIGHTGBM_NAME_TRANSLATION) for name in names]
    if len(set(sanitized)) != len(sanitized):
        raise ValueError("LightGBM用の安全化後に特徴量名が重複しました")
    return sanitized


def fit_fold_model(X: pd.DataFrame, y: pd.Series) -> FittedFoldModel:
    feature_builder = AverageBranchingFeatureBuilder(random_state=RANDOM_STATE)
    generated = feature_builder.fit_transform(X)
    excluded = [column for column in generated if generated[column].isna().all()]
    kept_columns = [column for column in generated if column not in excluded]
    generated = generated.loc[:, kept_columns]
    preprocessor = build_preprocessor(kept_columns)
    values = preprocessor.fit_transform(generated)
    names = _sanitize_feature_names(preprocessor.get_feature_names_out().tolist())
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
# 4. LightGBM・inner CV閾値選択
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
    *,
    seed: int,
) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
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
            seed=RANDOM_STATE + fold + 1,
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
            f"exp07 fold {fold}: inner F1={inner_f1:.4f}, threshold={threshold:.3f}, "
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

def main() -> None:
    X, y = load_data()
    result = run_nested_cv(X, y)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(result.feature_importance, SPEC)
    f1_figure = make_f1_figure(result, SPEC)
    feature_figure.savefig(
        RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)

    print("\n" + "=" * 96)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Input columns: {X.shape[1]}")
    print(f"Selected tree features: {list(SELECTED_TREE_FEATURES)}")
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


if __name__ == "__main__":
    main()
