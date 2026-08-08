"""全学習データで混合モデルを学習し、SIGNATE提出用CSVを生成する。

使い方:
    1. 必要に応じて、このファイル内の THRESHOLD を編集する。
    2. ``python submission.py`` を実行する。

``data/sample_submit.csv`` の2列目を予測値に置き換えた
``submission.csv`` をリポジトリ直下に出力する。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from calc_features.all_features_v1 import INDUSTRY_COLUMN
from calc_features.all_features_v2 import AllFeaturesV2Transformer


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "train.csv"
DEFAULT_TEST_PATH = BASE_DIR / "data" / "test.csv"
DEFAULT_SAMPLE_SUBMIT_PATH = BASE_DIR / "data" / "sample_submit.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "submission.csv"

TARGET_COLUMN = "購入フラグ"
ID_COLUMN = "企業ID"

# exp_mix/exp01のnested CVで選ばれた混合モデル用の閾値。
# 提出時に変更する場合は、ここを編集してから実行する。
THRESHOLD = 0.279

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}


def calculate_features(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, AllFeaturesV2Transformer, ColumnTransformer]:
    """混合特徴量を作り、モデルへ入力できるtrain/test行列を返す。

    使用するのは暫定数値・カテゴリ特徴量19列と、「今後のDX展望」の
    SVD第2主成分1列。特徴量生成器、欠損値補完、One-Hot Encodingは
    学習データだけでfitする。
    """
    feature_builder = AllFeaturesV2Transformer(random_state=42)
    generated_train = feature_builder.fit_transform(train_data)
    generated_test = feature_builder.transform(test_data)

    all_missing_columns = [
        column
        for column in generated_train.columns
        if generated_train[column].isna().all()
    ]
    if all_missing_columns:
        generated_train = generated_train.drop(columns=all_missing_columns)
        generated_test = generated_test.drop(columns=all_missing_columns)

    categorical_columns = [
        column
        for column in generated_train.columns
        if column == INDUSTRY_COLUMN
    ]
    numeric_columns = [
        column
        for column in generated_train.columns
        if column not in categorical_columns
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                ),
                categorical_columns,
            ),
        ],
        verbose_feature_names_out=False,
    )

    train_values = preprocessor.fit_transform(generated_train)
    test_values = preprocessor.transform(generated_test)
    feature_names = preprocessor.get_feature_names_out()
    train_features = pd.DataFrame(
        np.asarray(train_values),
        index=train_data.index,
        columns=feature_names,
    )
    test_features = pd.DataFrame(
        np.asarray(test_values),
        index=test_data.index,
        columns=feature_names,
    )
    return train_features, test_features, feature_builder, preprocessor


def predict_all(
    threshold: float,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    test_path: str | Path = DEFAULT_TEST_PATH,
    sample_submit_path: str | Path = DEFAULT_SAMPLE_SUBMIT_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """全学習データで学習し、thresholdを適用した提出CSVを出力する。"""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold は0以上1以下の有限な数値にしてください。")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submit = pd.read_csv(sample_submit_path, header=None)

    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"train.csv に `{TARGET_COLUMN}` 列がありません。")
    if sample_submit.shape[1] != 2:
        raise ValueError(
            "sample_submit.csv は2列である必要があります"
            f"（実際: {sample_submit.shape[1]}列）。"
        )
    if len(test) != len(sample_submit):
        raise ValueError(
            "test.csv と sample_submit.csv の行数が一致しません: "
            f"{len(test)} != {len(sample_submit)}"
        )
    if ID_COLUMN in test.columns:
        test_ids = test[ID_COLUMN].reset_index(drop=True)
        submit_ids = sample_submit.iloc[:, 0].reset_index(drop=True)
        if not test_ids.equals(submit_ids.astype(test_ids.dtype)):
            raise ValueError("test.csv と sample_submit.csv の企業ID順が一致しません。")

    y_train = train[TARGET_COLUMN].astype(int)
    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError(f"`{TARGET_COLUMN}` には0/1以外の値が含まれています。")

    train_features, test_features, _, _ = calculate_features(train, test)
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(train_features, y_train)

    probabilities = model.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    sample_submit.iloc[:, 1] = predictions

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_submit.to_csv(output_path, index=False, header=False)

    print(f"提出ファイル: {output_path}")
    print(f"入力特徴量数: {train_features.shape[1]}")
    print(f"threshold: {threshold:.4f}")
    print(f"予測件数: {len(predictions)}")
    print(f"購入予測数: {int(predictions.sum())} ({predictions.mean():.1%})")
    return sample_submit


def main() -> None:
    predict_all(threshold=THRESHOLD)


if __name__ == "__main__":
    main()
