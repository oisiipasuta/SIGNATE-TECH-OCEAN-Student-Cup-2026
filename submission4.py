"""exp_mix/exp04（all_features_v4）のSIGNATE提出用CSVを生成する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp_mix.exp04 import (
    TARGET_COLUMN,
    fit_fold_model,
    predict_probabilities,
)


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
SAMPLE_SUBMIT_PATH = BASE_DIR / "data" / "sample_submit.csv"
OUTPUT_PATH = BASE_DIR / "submission4.csv"

# exp_mix/exp04のnested CVで得たouter-fold閾値の平均。
THRESHOLD = 0.2770


def create_submission(
    output_path: str | Path = OUTPUT_PATH,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """全学習データでexp04を再学習し、提出形式の予測を保存する。"""
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("thresholdは0以上1以下の有限値にしてください。")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_SUBMIT_PATH, header=None)

    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"train.csvに目的変数`{TARGET_COLUMN}`がありません。")
    if sample.shape[1] != 2:
        raise ValueError(f"sample_submit.csvは2列である必要があります: {sample.shape[1]}列")
    if len(test) != len(sample):
        raise ValueError(
            "test.csvとsample_submit.csvの行数が一致しません: "
            f"{len(test)} != {len(sample)}"
        )

    y_train = train[TARGET_COLUMN].astype(int)
    if not set(y_train.unique()).issubset({0, 1}):
        raise ValueError(f"`{TARGET_COLUMN}`に0/1以外の値が含まれています。")

    X_train = train.drop(columns=[TARGET_COLUMN])
    fitted = fit_fold_model(X_train, y_train)
    probabilities = predict_probabilities(fitted, test)
    if len(probabilities) != len(sample) or not np.isfinite(probabilities).all():
        raise ValueError("test予測確率の件数または有限性が不正です。")

    predictions = (probabilities >= threshold).astype(int)
    submission = sample.copy()
    submission.iloc[:, 1] = predictions

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(destination, index=False, header=False)

    print(f"提出ファイル: {destination}")
    print(f"生成特徴量数: {len(fitted.feature_builder.get_feature_names_out())}")
    print(f"変換後特徴量数: {len(fitted.transformed_feature_names)}")
    print(f"threshold: {threshold:.4f}")
    print(f"予測件数: {len(predictions)}")
    print(f"購入予測数: {int(predictions.sum())} ({predictions.mean():.1%})")
    return submission


if __name__ == "__main__":
    create_submission()
