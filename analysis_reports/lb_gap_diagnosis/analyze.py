"""exp_mix/exp05のCV-LB差を再現可能に診断する。"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.tree_of_corp import calculate_tree_of_corp_features  # noqa: E402
from experiments.exp_mix.exp05 import (  # noqa: E402
    fit_fold_model,
    predict_probabilities,
)


TARGET_COLUMN = "購入フラグ"
SELECTED_FEATURES = ("DX変革組織有無", "平均分岐数")
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)
EXP08_K_SCORES = (
    0.7744,
    0.7804,
    0.8063,
    0.7901,
    0.7927,
    0.7958,
    0.7907,
    0.7938,
    0.7837,
    0.8010,
    0.7918,
    0.7949,
    0.7947,
    0.8000,
    0.8000,
    0.7989,
)
USER_REPORTED_PUBLIC_LB = 0.77


def main() -> None:
    for k, score in enumerate(EXP08_K_SCORES):
        print(
            "TOP_K_RESULT",
            f"k={k}",
            f"nested_oof_f1={score:.4f}",
            f"user_reported_public_lb={USER_REPORTED_PUBLIC_LB:.2f}",
        )
    train = pd.read_csv(BASE_DIR / "data" / "train.csv")
    test = pd.read_csv(BASE_DIR / "data" / "test.csv")
    y = train[TARGET_COLUMN].astype(int)
    X = train.drop(columns=[TARGET_COLUMN])
    print(
        "SHAPE",
        f"train={len(train)}",
        f"test={len(test)}",
        f"target_rate={y.mean():.6f}",
        f"positives={int(y.sum())}",
    )

    train_tree = calculate_tree_of_corp_features(train)
    test_tree = calculate_tree_of_corp_features(test)
    for column in SELECTED_FEATURES:
        train_values = train_tree[column]
        test_values = test_tree[column]
        pooled_std = float(
            np.sqrt((train_values.var(ddof=1) + test_values.var(ddof=1)) / 2)
        )
        smd = (
            float((test_values.mean() - train_values.mean()) / pooled_std)
            if pooled_std > 0
            else 0.0
        )
        ks = ks_2samp(train_values.dropna(), test_values.dropna())
        print(
            "FEATURE",
            column,
            f"train_mean={train_values.mean():.6f}",
            f"test_mean={test_values.mean():.6f}",
            f"train_missing={train_values.isna().mean():.6f}",
            f"test_missing={test_values.isna().mean():.6f}",
            f"smd={smd:.6f}",
            f"ks={ks.statistic:.6f}",
            f"ks_p={ks.pvalue:.6f}",
            f"corr_y={train_values.corr(y):.6f}",
        )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_probabilities = np.zeros(len(X), dtype=float)
    for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
        fitted = fit_fold_model(X.iloc[train_index], y.iloc[train_index])
        oof_probabilities[valid_index] = predict_probabilities(
            fitted, X.iloc[valid_index]
        )
        print(f"OOF_MODEL fold={fold} complete")

    scores = np.array(
        [
            f1_score(y, (oof_probabilities >= threshold).astype(int))
            for threshold in THRESHOLD_GRID
        ]
    )
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        tied,
        key=lambda index: (abs(THRESHOLD_GRID[index] - 0.5), index),
    )
    best_threshold = float(THRESHOLD_GRID[best_index])
    thresholds = sorted({0.185, 0.230, 0.235, 0.247, 0.270, 0.315, 0.326, best_threshold})
    for threshold in thresholds:
        predictions = (oof_probabilities >= threshold).astype(int)
        print(
            "OOF_THRESHOLD",
            f"threshold={threshold:.3f}",
            f"f1={f1_score(y, predictions):.6f}",
            f"precision={precision_score(y, predictions):.6f}",
            f"recall={recall_score(y, predictions):.6f}",
            f"positive_rate={predictions.mean():.6f}",
        )
    print(
        "GLOBAL_OOF_BEST",
        f"threshold={best_threshold:.3f}",
        f"f1={best_score:.6f}",
    )

    deployment_predictions = (oof_probabilities >= 0.247).astype(int)
    rng = np.random.default_rng(42)
    bootstrap_scores = []
    for _ in range(5000):
        sample_index = rng.integers(0, len(y), size=len(y))
        bootstrap_scores.append(
            f1_score(y.iloc[sample_index], deployment_predictions[sample_index])
        )
    lower, upper = np.quantile(bootstrap_scores, [0.025, 0.975])
    print(
        "OOF_BOOTSTRAP",
        "threshold=0.247",
        f"f1={f1_score(y, deployment_predictions):.6f}",
        f"ci95_lower={lower:.6f}",
        f"ci95_upper={upper:.6f}",
    )

    full_model = fit_fold_model(X, y)
    test_probabilities = predict_probabilities(full_model, test)
    for threshold in (0.200, 0.230, 0.247, 0.270, 0.300, 0.326):
        predictions = (test_probabilities >= threshold).astype(int)
        print(
            "TEST_THRESHOLD",
            f"threshold={threshold:.3f}",
            f"positive_count={int(predictions.sum())}",
            f"positive_rate={predictions.mean():.6f}",
        )

    for filename in (
        "submission3.csv",
        "submission4.csv",
        "submission5.csv",
        "submission6.csv",
    ):
        submission = pd.read_csv(BASE_DIR / filename, header=None)
        print(
            "SUBMISSION",
            filename,
            f"positive_count={int(submission.iloc[:, 1].sum())}",
            f"positive_rate={submission.iloc[:, 1].mean():.6f}",
        )


if __name__ == "__main__":
    main()
