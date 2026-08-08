"""
実験ID: exp02
実験名: クラス重み（scale_pos_weight）の比較
著者: oisiipasuta

目的・仮説:
- 陽性率24.1%の不均衡に対して scale_pos_weight が必要か検証する。
- exp01と同じ特徴量、前処理、LightGBMパラメータ、CV分割を固定し、
  scale_pos_weight = 1.0, 2.0, 3.15 のみを変更する。
- 各重み・各outer foldで、inner 4-fold OOF予測からF1最大の閾値を再最適化する。

特徴量・前処理:
- calc_featuresのうちdx_outlook.pyを除く5モジュールをexp1と同じ順序で使用する。
- 数値特徴量は中央値補完、カテゴリ特徴量は最頻値補完 + One-Hot Encoding。
- 前処理は各CV foldの学習データだけでfitする。

モデル・CV:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 採用条件: baselineより平均F1が0.005以上改善、fold F1標準偏差が増えない、
  nested OOF PR-AUCが悪化しない、のすべてを満たす場合のみ重みありを採用する。
- 差が小さい場合はscale_pos_weight=1.0を採用する。

出力:
- experiments/exp_base/results/exp02/feature_importance.png
- experiments/exp_base/results/exp02/f1_scores.png
- CSV/JSON、予測値、submissionは出力しない。

実行結果（2026-08-08）:
- 陽性率: 24.1240%。入力特徴量22、One-Hot後55。
- 全欠損のため除外: 人材不足フラグ、予算制約フラグ、組織部門数、組織階層数、
  業務種類数、現場課題数、システム刷新フラグ、導入時期フラグ。
- weight=1.0: threshold=[0.275, 0.235, 0.285, 0.375, 0.340]、
  fold F1=[0.6098, 0.6835, 0.6410, 0.7200, 0.6269]、
  mean±std=0.6562±0.0402、nested OOF F1=0.6562、PR-AUC=0.7420。
- weight=2.0: threshold=[0.385, 0.425, 0.375, 0.280, 0.375]、
  fold F1=[0.5600, 0.7143, 0.6400, 0.6667, 0.7692]、
  mean±std=0.6700±0.0705、nested OOF F1=0.6701、PR-AUC=0.7450。
- weight=3.15: threshold=[0.435, 0.435, 0.490, 0.395, 0.380]、
  fold F1=[0.5897, 0.6944, 0.6479, 0.6591, 0.7381]、
  mean±std=0.6659±0.0494、nested OOF F1=0.6667、PR-AUC=0.7454。
- weight=2.0と3.15はいずれもfold間標準偏差がbaselineより増えたため不採用。
- 採用scale_pos_weight=1.0、最終閾値（fold閾値の平均）=0.3020。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

plt.rcParams["font.family"] = "Noto Sans JP"
plt.rcParams["axes.unicode_minus"] = False

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

EXPERIMENT_ID = "exp02"
EXPERIMENT_NAME = "scale_pos_weight comparison"
# 元CSVの列名は環境依存の文字化けを含むため、train/testの列差から厳密に解決する。
TARGET_COLUMN: str | None = None

BASE_MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
}
SCALE_POS_WEIGHTS = (1.0, 2.0, 3.15)
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)
MIN_F1_IMPROVEMENT = 0.005
COMPARISON_TOLERANCE = 1e-12

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_base" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """exp01と同じ5つの特徴量生成関数を、同じ順序で適用する。"""
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
    train_features = calculate_features(train)
    test_features = calculate_features(test)
    all_missing_columns = [
        column for column in train_features.columns if train_features[column].isna().all()
    ]
    train_features = train_features.drop(columns=all_missing_columns)
    test_features = test_features.drop(columns=all_missing_columns)

    categorical_features = train_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_features = [
        column for column in train_features.columns if column not in categorical_features
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
# 3. 前処理・モデル
# ==================================================

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
) -> Pipeline:
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
    model_params = {**BASE_MODEL_PARAMS, "scale_pos_weight": scale_pos_weight}
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**model_params)),
        ]
    )


# ==================================================
# 4. 閾値選択
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
    scale_pos_weight: float,
    random_state: int,
) -> tuple[float, float]:
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    inner_oof_probabilities = np.zeros(len(X), dtype=float)
    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        pipeline = build_pipeline(
            numeric_features, categorical_features, scale_pos_weight
        )
        pipeline.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
        inner_oof_probabilities[inner_valid_index] = pipeline.predict_proba(
            X.iloc[inner_valid_index]
        )[:, 1]
    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 5. ネステッドCV・重み比較
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_pos_weight: float,
) -> dict[str, object]:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_probabilities = np.zeros(len(X), dtype=float)
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    fold_pr_aucs: list[float] = []
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
            scale_pos_weight,
            random_state=RANDOM_STATE + fold,
        )
        pipeline = build_pipeline(
            numeric_features, categorical_features, scale_pos_weight
        )
        pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)
        fold_pr_auc = average_precision_score(y_valid, valid_probabilities)

        oof_probabilities[valid_index] = valid_probabilities
        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(float(fold_f1))
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        fold_pr_aucs.append(float(fold_pr_auc))

        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        feature_importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_names, "importance": importances, "fold": fold}
            )
        )
        print(
            f"weight={scale_pos_weight:g}, outer fold {fold}: "
            f"threshold={threshold:.3f}, inner OOF F1={inner_f1:.4f}, "
            f"valid F1={fold_f1:.4f}, PR-AUC={fold_pr_auc:.4f}"
        )

    importance_long = pd.concat(feature_importance_frames, ignore_index=True)
    feature_importance = (
        importance_long.pivot_table(
            index="feature", columns="fold", values="importance", fill_value=0.0
        )
        .reindex(columns=range(1, OUTER_N_SPLITS + 1), fill_value=0.0)
        .mean(axis=1)
        .rename("importance")
        .reset_index()
        .sort_values(["importance", "feature"], ascending=[False, True])
    )
    return {
        "scale_pos_weight": scale_pos_weight,
        "oof_probabilities": oof_probabilities,
        "oof_predictions": oof_predictions,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "fold_pr_aucs": fold_pr_aucs,
        "final_threshold": float(np.mean(fold_thresholds)),
        "mean_f1": float(np.mean(fold_scores)),
        "std_f1": float(np.std(fold_scores)),
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "nested_oof_pr_auc": float(average_precision_score(y, oof_probabilities)),
        "feature_importance": feature_importance,
    }


def choose_weight(results_by_weight: dict[float, dict[str, object]]) -> float:
    baseline = results_by_weight[1.0]
    eligible: list[float] = []
    for weight in SCALE_POS_WEIGHTS[1:]:
        candidate = results_by_weight[weight]
        improves_f1 = (
            candidate["mean_f1"] - baseline["mean_f1"]
            >= MIN_F1_IMPROVEMENT - COMPARISON_TOLERANCE
        )
        stable = candidate["std_f1"] <= baseline["std_f1"] + COMPARISON_TOLERANCE
        preserves_pr_auc = (
            candidate["nested_oof_pr_auc"]
            >= baseline["nested_oof_pr_auc"] - COMPARISON_TOLERANCE
        )
        if improves_f1 and stable and preserves_pr_auc:
            eligible.append(weight)
    if not eligible:
        return 1.0
    return max(
        eligible,
        key=lambda weight: (
            results_by_weight[weight]["mean_f1"],
            results_by_weight[weight]["nested_oof_pr_auc"],
            -weight,
        ),
    )


# ==================================================
# 6. 結果図・実行
# ==================================================

def plot_feature_importance(
    feature_importance: pd.DataFrame,
    selected_weight: float,
) -> None:
    plot_data = feature_importance.sort_values("importance", ascending=True)
    figure_height = max(6.0, 0.34 * len(plot_data) + 2.0)
    fig, ax = plt.subplots(figsize=(12, figure_height))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color="#3B82F6")
    max_importance = max(float(plot_data["importance"].max()), 1.0)
    ax.set_xlim(0, max_importance * 1.18)
    for bar, value in zip(bars, plot_data["importance"]):
        ax.text(
            bar.get_width() + max_importance * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        f"Feature importance — selected scale_pos_weight={selected_weight:g}",
        y=0.995,
        fontsize=14,
    )
    ax.set_title(
        "LightGBM split importance, averaged over the 5 outer-fold models; all transformed features",
        fontsize=9,
        color="#475569",
        pad=8,
    )
    ax.set_xlabel("Mean split importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(
    results_by_weight: dict[float, dict[str, object]],
    selected_weight: float,
) -> None:
    folds = np.arange(1, OUTER_N_SPLITS + 1)
    width = 0.24
    colors = ("#64748B", "#3B82F6", "#F59E0B")
    fig, ax = plt.subplots(figsize=(12, 7))
    for index, (weight, color) in enumerate(zip(SCALE_POS_WEIGHTS, colors)):
        result = results_by_weight[weight]
        positions = folds + (index - 1) * width
        bars = ax.bar(
            positions,
            result["fold_scores"],
            width=width,
            color=color,
            label=f"weight={weight:g}",
        )
        ax.axhline(
            result["mean_f1"],
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
        )
        for bar, value in zip(bars, result["fold_scores"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.008,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
    summaries = []
    for weight in SCALE_POS_WEIGHTS:
        result = results_by_weight[weight]
        marker = "  SELECTED" if weight == selected_weight else ""
        summaries.append(
            f"w={weight:g}: mean={result['mean_f1']:.4f}, std={result['std_f1']:.4f}, "
            f"nested OOF F1={result['nested_oof_f1']:.4f}, "
            f"PR-AUC={result['nested_oof_pr_auc']:.4f}{marker}"
        )
    ax.text(
        0.01,
        0.98,
        "\n".join(summaries),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fig.suptitle("Outer-fold F1 by scale_pos_weight", y=0.985, fontsize=14)
    ax.set_title(
        "Each bar is an untouched outer fold; threshold from inner-CV OOF predictions; dashed lines are fold means",
        fontsize=9,
        color="#475569",
        pad=8,
    )
    ax.set_xticks(folds, [f"Fold {fold}" for fold in folds])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X, _, numeric_features, categorical_features, excluded = prepare_features(train, test)
    target_candidates = [column for column in train.columns if column not in test.columns]
    if TARGET_COLUMN is not None:
        target_column = TARGET_COLUMN
    elif len(target_candidates) == 1:
        target_column = target_candidates[0]
    else:
        raise ValueError(
            "trainにだけ存在する目的変数列を一意に特定できません: "
            f"{target_candidates}"
        )
    y = train[target_column].astype(int)

    results_by_weight = {
        weight: run_nested_cv(
            X, y, numeric_features, categorical_features, scale_pos_weight=weight
        )
        for weight in SCALE_POS_WEIGHTS
    }
    selected_weight = choose_weight(results_by_weight)
    selected_result = results_by_weight[selected_weight]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(selected_result["feature_importance"], selected_weight)
    plot_f1_scores(results_by_weight, selected_weight)

    transformed_feature_count = len(selected_result["feature_importance"])
    print()
    print("=" * 90)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Positive rate: {y.mean():.4%}")
    print(
        f"Input features: {len(X.columns)}; transformed features: "
        f"{transformed_feature_count}; excluded all-missing: {excluded}"
    )
    for weight in SCALE_POS_WEIGHTS:
        result = results_by_weight[weight]
        print(
            f"weight={weight:g}: thresholds={result['fold_thresholds']}, "
            f"fold F1={result['fold_scores']}, mean={result['mean_f1']:.4f}, "
            f"std={result['std_f1']:.4f}, nested OOF F1={result['nested_oof_f1']:.4f}, "
            f"nested OOF PR-AUC={result['nested_oof_pr_auc']:.4f}"
        )
    print(
        f"Selected scale_pos_weight: {selected_weight:g}; "
        f"final threshold: {selected_result['final_threshold']:.4f}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()
