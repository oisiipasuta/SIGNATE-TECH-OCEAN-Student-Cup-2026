"""
実験ID: exp03
実験名: 重複特徴量の除外
著者: oisiipasuta

実験概要:
- Exp01と同じ特徴量生成・LightGBM・ネストCV条件を維持する。
- 同じ情報を表す候補として、従業員規模、現行ツール状態、
  赤字・CF不足フラグを除外し、特徴量削減後のF1をExp1と比較する。
- log_売上とlog_従業員数は互いに異なる企業規模情報なので維持する。

使用特徴量:
- calc_featuresのうちdx_outlook.pyを除く5モジュールから生成する。
- 全行欠損の仮実装列を除外した後、重複候補3列を除外する。

前処理:
- 数値特徴量: 中央値補完
- カテゴリ特徴量: 最頻値補完 + One-Hot Encoding
- 前処理は各CV foldの学習データだけでfitする。

モデル:
- LightGBM (LGBMClassifier)
- n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7

評価設計:
- 外側5-fold・内側4-foldのStratifiedKFold（random_state=42）。
- 各外側foldの閾値は、その外側学習データ内のinner OOF F1だけで決める。
- 閾値候補は0.05から0.95まで0.005刻み。同点なら0.5に近い値を選ぶ。
- 最終thresholdは外側fold閾値の平均とする。

出力:
- experiments/exp_base/results/exp03/feature_importance.png
- experiments/exp_base/results/exp03/f1_scores.png
- CSV、JSON、予測値、提出ファイルは出力しない。

結果（2026-08-08実行）:
- 使用特徴量数: 19（Exp01は22）
- One-Hot後特徴量数: 49（Exp01は55）
- 全行欠損による除外: 人材不足フラグ、予算制約フラグ、組織部門数、
  組織階層数、業務種類数、現場課題数、システム刷新フラグ、導入時期フラグ
- 重複候補として除外: 従業員規模、現行ツール状態、赤字・CF不足フラグ
- fold threshold: 0.310, 0.235, 0.220, 0.155, 0.245
- fold F1: 0.6234, 0.7000, 0.6067, 0.6737, 0.7342
- fold F1 mean ± std: 0.6676 ± 0.0473
- nested OOF F1: 0.6667（Exp01は0.6562、差+0.0105）
- 最終threshold: 0.2330
- 特徴量を削減してF1が同等以上だったため、Exp03を採用候補とする。
"""

from __future__ import annotations

import sys
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

# `python experiments/exp_base/exp03.py` で実行してもリポジトリ直下をimportできるようにする。
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

EXPERIMENT_ID = "exp03"
EXPERIMENT_NAME = "重複特徴量の除外"
AUTHOR = "oisiipasuta"

TARGET_COLUMN = "購入フラグ"
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

DUPLICATE_FEATURES = [
    "従業員規模",
    "現行ツール状態",
    "赤字・CF不足フラグ",
]

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_base" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exp01と同じ5モジュールの特徴量を生成する。"""
    return pd.concat(
        [
            calculate_execution_features(df),
            calculate_motivation_features(df),
            calculate_adoption_barrier_features(df),
            calculate_necessity_features(df),
            calculate_purchase_timing_features(df),
        ],
        axis=1,
    )


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """全欠損列と指定した重複候補を除外し、学習可能な型へ揃える。"""
    train_features = calculate_features(train)
    test_features = calculate_features(test)

    all_missing_columns = [
        column for column in train_features if train_features[column].isna().all()
    ]
    excluded_columns = all_missing_columns + DUPLICATE_FEATURES
    train_features = train_features.drop(columns=excluded_columns)
    test_features = test_features.drop(columns=excluded_columns)

    categorical_features = train_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_features = [
        column for column in train_features if column not in categorical_features
    ]

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
# 3. 特徴量前処理・モデル
# ==================================================

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """foldごとに新しい前処理 + LightGBMパイプラインを作る。"""
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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

def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """F1最大の候補閾値を返す。同点なら0.5に近い方を選ぶ。"""
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
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, object]:
    """ネストCV評価と全変換特徴量のfold平均split重要度を返す。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    feature_importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        threshold, inner_f1 = calculate_inner_threshold(
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            random_state=RANDOM_STATE + fold + 1,
        )
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)

        oof_predictions[valid_index] = valid_predictions
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

    importance_by_fold = pd.concat(feature_importance_frames).pivot_table(
        index="feature",
        columns="fold",
        values="importance",
        fill_value=0.0,
    )
    feature_importance = (
        importance_by_fold.reindex(columns=range(OUTER_N_SPLITS), fill_value=0.0)
        .mean(axis=1)
        .rename("importance")
        .reset_index()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "oof_predictions": oof_predictions,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "final_threshold": float(np.mean(fold_thresholds)),
        "feature_importance": feature_importance,
    }


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def configure_japanese_font() -> str:
    """利用可能な日本語フォントを設定し、採用名を返す。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return str(plt.rcParams["font.family"])


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """全変換特徴量の平均split重要度を降順で描く。"""
    plot_data = feature_importance.sort_values("importance", ascending=True)
    figure_height = max(6.5, 0.36 * len(plot_data) + 2.8)
    fig, ax = plt.subplots(figsize=(12, figure_height))
    bars = ax.barh(
        plot_data["feature"],
        plot_data["importance"],
        color="#2F6B8A",
    )
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("Exp03 特徴量重要度", fontsize=16, fontweight="bold", pad=28)
    ax.text(
        0.5,
        1.01,
        "LightGBM split importance・外側5-fold平均（重要度は因果や方向を示さない）",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4B5563",
    )
    ax.set_xlabel("平均 split importance")
    ax.set_ylabel("One-Hot後の特徴量（業界水準は非集約）")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(
    fold_scores: list[float],
    nested_oof_f1: float,
) -> None:
    """各外側検証foldのF1と集約指標を描く。"""
    fold_mean = float(np.mean(fold_scores))
    fold_std = float(np.std(fold_scores))
    folds = np.arange(len(fold_scores))
    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.bar(folds, fold_scores, color="#2F6B8A", width=0.62)
    ax.bar_label(bars, labels=[f"{score:.4f}" for score in fold_scores], padding=4)
    ax.axhline(
        fold_mean,
        color="#D97706",
        linestyle="--",
        linewidth=1.8,
        label=f"fold平均 = {fold_mean:.4f}",
    )
    ax.text(
        0.98,
        0.96,
        f"nested OOF F1 = {nested_oof_f1:.4f}\nfold標準偏差 = {fold_std:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#9CA3AF"},
    )
    ax.set_title("Exp03 外側検証fold別 F1", fontsize=16, fontweight="bold", pad=30)
    ax.text(
        0.5,
        1.01,
        "各barは学習に未使用の外側foldを評価し、閾値は内側CVだけで選択",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4B5563",
    )
    ax.set_xlabel("外側fold（0始まり）")
    ax.set_ylabel("F1")
    ax.set_xticks(folds)
    ax.set_ylim(0.0, max(1.0, max(fold_scores) + 0.12))
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X, _, numeric_features, categorical_features, all_missing = prepare_features(
        train, test
    )
    y = train[TARGET_COLUMN].astype(int)
    results = run_nested_cv(X, y, numeric_features, categorical_features)
    fold_scores = results["fold_scores"]
    nested_oof_f1 = f1_score(y, results["oof_predictions"])
    feature_importance = results["feature_importance"]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    plot_feature_importance(feature_importance)
    plot_f1_scores(fold_scores, nested_oof_f1)

    print()
    print("=" * 72)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Raw features: {len(X.columns)}")
    print(f"Transformed features: {len(feature_importance)}")
    print(f"Excluded all-missing: {all_missing}")
    print(f"Excluded duplicates: {DUPLICATE_FEATURES}")
    print(f"Fold thresholds: {results['fold_thresholds']}")
    print(f"Final threshold: {results['final_threshold']:.4f}")
    print(f"Fold F1 mean: {np.mean(fold_scores):.4f}")
    print(f"Fold F1 std: {np.std(fold_scores):.4f}")
    print(f"Nested OOF F1: {nested_oof_f1:.4f}")
    print(f"Plot font: {font_name}")
    print("=" * 72)


if __name__ == "__main__":
    main()
