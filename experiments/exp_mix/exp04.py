"""
実験ID: exp04
実験名: all_features_v4（v3＋汎用組織図特徴）
著者: Codex

目的・仮説:
- exp_tree_of_corp/exp02で有効だったall_features_v3＋汎用組織図15特徴を、
  calc_features.all_features_v4として正式化し、exp_mix系列で再現評価する。

特徴量・前処理:
- AllFeaturesV4Transformerを各fold内でfitし、v3の23列と汎用組織図15列、
  計38列を生成する。DX推進室完全一致フラグは使用しない。
- 数値列は学習foldの中央値で補完する。
- 業界は学習foldだけで補完・One-Hot Encodingし、未知カテゴリは無視する。
- 学習foldで全欠損の列だけを除外する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4。
- 各outer foldの閾値はouter学習部分のinner OOF F1だけから選ぶ。
- exp_tree_of_corp/exp02のnested OOF F1=0.7989を再現できるか確認する。

出力:
- experiments/exp_mix/results/exp04/feature_importance.png
- experiments/exp_mix/results/exp04/f1_scores.png
- 実験スクリプト自体はCSV、JSON、予測値、submissionを出力しない。

実行結果（2026-08-19）:
- 入力42列から38特徴を生成。One-Hot後51/50/52/52/51列、全欠損除外なし。
- fold閾値: 0.345/0.185/0.225/0.330/0.300、最終閾値=0.2770。
- fold F1: 0.7647/0.8642/0.7632/0.7692/0.8286、平均=0.7980、
  標準偏差=0.0412、nested OOF F1=0.7989。
- exp_tree_of_corp/exp02の特徴量数、変換後列数、fold閾値、fold F1、
  nested OOF F1を完全再現した。
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
from calc_features.all_features_v4 import AllFeaturesV4Transformer  # noqa: E402
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

EXPERIMENT_ID = "exp04"
EXPERIMENT_NAME = "all_features_v4（v3＋汎用組織図特徴）"
TARGET_COLUMN = "購入フラグ"
SPEC = ExperimentSpec(EXPERIMENT_ID, EXPERIMENT_NAME, 4)

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


@dataclass
class FittedFoldModel:
    feature_builder: AllFeaturesV4Transformer
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
    return train.drop(columns=[TARGET_COLUMN]), train[TARGET_COLUMN].astype(int)


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
    feature_builder = AllFeaturesV4Transformer(random_state=RANDOM_STATE)
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
            f"exp04 fold {fold}: inner F1={inner_f1:.4f}, threshold={threshold:.3f}, "
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
