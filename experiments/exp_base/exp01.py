"""
実験ID: exp01
実験名: calc_features（dx_outlook除外）+ LightGBM ベースライン
著者: oisiipasuta

実験概要:
- calc_features のうち dx_outlook.py を除く5モジュールから特徴量を生成する。
- LightGBMで購入フラグを学習し、購入確率を予測する。
- 外側5-fold・内側4-foldのネストCVを用い、内側OOF F1が最大となる閾値を
  外側foldごとに決定する。

使用特徴量:
- 実行能力、推進意欲、導入障壁・充足済み度、必要性、購買タイミング
- 全行欠損の仮実装列は学習対象から自動除外する。

前処理:
- 数値特徴量: 中央値補完
- カテゴリ特徴量: 最頻値補完 + One-Hot Encoding
- 前処理は各CV foldの学習データだけでfitする。

モデル:
- LightGBM (LGBMClassifier)
- n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7

結果（2026-08-08実行後に更新）:
- 使用特徴量数: 22（生成30列のうち、全行欠損の仮実装8列を除外）
- fold threshold: 0.275, 0.235, 0.285, 0.375, 0.340
- fold F1: 0.6098, 0.6835, 0.6410, 0.7200, 0.6269
- fold F1 mean ± std: 0.6562 ± 0.0402
- nested OOF F1: 0.6562
- 最終threshold（外側fold閾値の平均）: 0.3020
- One-Hot後の特徴量重要度上位50件とF1スコアの図を
  experiments/exp_base/results/exp01/へ保存する（業界カテゴリは集約しない）。



"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# `python experiments/exp_base/exp01.py` で実行してもリポジトリ直下をimportできるようにする。
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.adoption_barriers import calculate_adoption_barrier_features
from calc_features.excute_capacity import calculate_execution_features
from calc_features.motivation import calculate_motivation_features
from calc_features.necessity import calculate_necessity_features
from calc_features.purchase_timing import calculate_purchase_timing_features


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp01"
EXPERIMENT_NAME = "calc_features（dx_outlook除外）+ LightGBM ベースライン"
AUTHOR = "oisiipasuta"

ID_COLUMN = "企業ID"
TARGET_COLUMN = "購入フラグ"
MODEL_NAME = "LightGBM"

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """dx_outlook.pyを除くcalc_featuresの全生成関数を適用する。"""
    feature_frames = [
        calculate_execution_features(df),
        calculate_motivation_features(df),
        calculate_adoption_barrier_features(df),
        calculate_necessity_features(df),
        calculate_purchase_timing_features(df),
    ]
    return pd.concat(feature_frames, axis=1)


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """train/testに特徴量を生成し、学習可能な型へ揃える。"""
    train_features = calculate_features(train)
    test_features = calculate_features(test)

    # 既存モジュールの仮実装列は全件欠損を返すため、学習対象から除外する。
    all_missing_columns = [
        column
        for column in train_features.columns
        if train_features[column].isna().all()
    ]
    train_features = train_features.drop(columns=all_missing_columns)
    test_features = test_features.drop(columns=all_missing_columns)

    categorical_features = train_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_features = [
        column
        for column in train_features.columns
        if column not in categorical_features
    ]

    # pandasのnullable dtypeをsklearnへ安全に渡す。
    train_features[numeric_features] = train_features[numeric_features].astype(float)
    test_features[numeric_features] = test_features[numeric_features].astype(float)
    for column in categorical_features:
        train_features[column] = train_features[column].astype("object")
        test_features[column] = test_features[column].astype("object")

    return (
        train_features,
        test_features,
        numeric_features,
        categorical_features,
        all_missing_columns,
    )


# ==================================================
# 3. 特徴量前処理
# ==================================================

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """foldごとに新しい前処理 + LightGBMパイプラインを作る。"""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy='median'))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy='most_frequent')),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**MODEL_PARAMS)),
        ]
    )


# ==================================================
# 4. 閾値選択
# ==================================================

def select_threshold(y_true: pd.Series | np.ndarray, probabilities: np.ndarray) -> tuple[float, float]:
    """候補のうちF1最大の閾値を返す。同点なら0.5に近い方を選ぶ。"""
    scores = np.array(
        [
            f1_score(y_true, (probabilities >= threshold).astype(int))
            for threshold in THRESHOLD_CANDIDATES
        ]
    )
    best_score = scores.max()
    best_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = best_indices[
        np.argmin(np.abs(THRESHOLD_CANDIDATES[best_indices] - 0.5))
    ]
    return float(THRESHOLD_CANDIDATES[best_index]), float(best_score)


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> tuple[float, float]:
    """外側学習データ内だけでinner OOF確率を作り、閾値を選ぶ。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    inner_oof_probabilities = np.zeros(len(X), dtype=float)

    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
        inner_oof_probabilities[inner_valid_index] = pipeline.predict_proba(
            X.iloc[inner_valid_index]
        )[:, 1]

    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 5. ネストクロスバリデーション
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    X_test: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, object]:
    """ネストCV評価と外側foldモデルによるtest確率予測を行う。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_probabilities = np.zeros(len(X), dtype=float)
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_numbers = np.zeros(len(X), dtype=int)
    applied_thresholds = np.zeros(len(X), dtype=float)
    test_fold_probabilities = np.zeros((len(X_test), OUTER_N_SPLITS), dtype=float)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    feature_importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        threshold, inner_f1 = calculate_inner_threshold(
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            random_state=RANDOM_STATE + fold,
        )
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)

        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)

        oof_probabilities[valid_index] = valid_probabilities
        oof_predictions[valid_index] = valid_predictions
        fold_numbers[valid_index] = fold
        applied_thresholds[valid_index] = threshold
        test_fold_probabilities[:, fold - 1] = pipeline.predict_proba(X_test)[:, 1]
        fold_scores.append(float(fold_f1))
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)

        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        feature_importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_names, "importance": importances, "fold": fold}
            )
        )
        print(
            f"Outer fold {fold}: threshold={threshold:.3f}, "
            f"inner OOF F1={inner_f1:.4f}, valid F1={fold_f1:.4f}"
        )

    final_threshold = float(np.mean(fold_thresholds))
    test_probabilities = test_fold_probabilities.mean(axis=1)
    test_predictions = (test_probabilities >= final_threshold).astype(int)
    feature_importance = (
        pd.concat(feature_importance_frames, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )

    return {
        "oof_probabilities": oof_probabilities,
        "oof_predictions": oof_predictions,
        "fold_numbers": fold_numbers,
        "applied_thresholds": applied_thresholds,
        "test_probabilities": test_probabilities,
        "test_predictions": test_predictions,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "final_threshold": final_threshold,
        "feature_importance": feature_importance,
    }


# ==================================================
# 6. 実験結果・ファイル出力
# ==================================================

def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X, X_test, numeric_features, categorical_features, excluded = prepare_features(
        train, test
    )
    y = train[TARGET_COLUMN].astype(int)

    results = run_nested_cv(
        X,
        y,
        X_test,
        numeric_features,
        categorical_features,
    )
    fold_scores = results["fold_scores"]
    nested_oof_f1 = f1_score(y, results["oof_predictions"])

    print()
    print("=" * 70)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Features: {len(X.columns)} (excluded all-missing: {len(excluded)})")
    print(f"Fold thresholds: {results['fold_thresholds']}")
    print(f"Final threshold: {results['final_threshold']:.4f}")
    print(f"Fold F1 mean: {np.mean(fold_scores):.4f}")
    print(f"Fold F1 std: {np.std(fold_scores):.4f}")
    print(f"Nested OOF F1: {nested_oof_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
