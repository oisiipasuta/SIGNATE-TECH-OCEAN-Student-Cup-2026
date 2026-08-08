"""
実験ID: exp05
実験名: 重要特徴量のみのモデル
著者: Codex

目的・仮説:
- 742件の小規模データで、特徴量数を絞ると汎化性能が改善するかを確認する。
- 利用可能な全特徴量、学習fold内重要度の上位20、上位10を比較する。
- 「実行能力」「必要性」「意欲」は特徴量群の見出しと解釈し、ユーザー提示の
  18候補を含む、calc_featuresの利用可能な22元特徴量を選択母集団とする。

特徴量・前処理:
- calc_featuresの5モジュールから30列を生成し、全行欠損列だけを除外する。
- 数値列は中央値補完、カテゴリ列は最頻値補完 + One-Hot Encodingを行う。
- 補完、One-Hot Encoding、特徴量選択は各学習fold内だけでfitする。
- 上位Kの選択では、学習foldに全特徴量モデルをfitし、カテゴリ水準のsplit
  importanceを元特徴量単位に合計して順位を決める。

モデル・CV・閾値:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 各outer fold・各モデルでinner OOF F1最大の閾値を決め、未使用のouter
  validation foldだけで評価する。同点時は0.5に最も近い閾値を採用する。
- 最終閾値は採用モデルのouter fold閾値の平均とする。

出力:
- experiments/exp_base/results/exp05/feature_importance.png
- experiments/exp_base/results/exp05/f1_scores.png
- CSV/JSON、予測値、submissionは出力しない。

実行結果（2026-08-08）:
- 生成30特徴量 / 利用可能22元特徴量。変換後特徴量数はall=54, 55, 53, 54, 54、
  top20=52, 53, 51, 52, 52、top10=39, 40, 38, 39, 39（各outer fold）。
  top20の完全な平均重要度図ではfold間のカテゴリ水準の和集合53列を表示する。
- 全行欠損のため除外: 人材不足フラグ、予算制約フラグ、組織部門数、組織階層数、
  業務種類数、現場課題数、システム刷新フラグ、導入時期フラグ。
- all: threshold=0.275, 0.235, 0.285, 0.375, 0.340、fold F1=0.6098,
  0.6835, 0.6410, 0.7200, 0.6269、mean ± std=0.6562 ± 0.0402、
  nested OOF F1=0.6562。
- top20: threshold=0.275, 0.245, 0.285, 0.375, 0.340、fold F1=0.6098,
  0.6923, 0.6410, 0.7200, 0.6269、mean ± std=0.6580 ± 0.0415、
  nested OOF F1=0.6579。
- top10: threshold=0.395, 0.165, 0.230, 0.395, 0.230、fold F1=0.6154,
  0.6444, 0.5814, 0.5806, 0.6133、mean ± std=0.6070 ± 0.0239、
  nested OOF F1=0.6087。
- 採用モデル: top20。最終threshold=0.3040。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.adoption_barriers import calculate_adoption_barrier_features
from calc_features.excute_capacity import calculate_execution_features
from calc_features.motivation import calculate_motivation_features
from calc_features.necessity import calculate_necessity_features
from calc_features.purchase_timing import calculate_purchase_timing_features

JAPANESE_FONT_PATH = Path("C:/Windows/Fonts/NotoSansJP-VF.ttf")
if JAPANESE_FONT_PATH.exists():
    font_manager.fontManager.addfont(JAPANESE_FONT_PATH)
    plt.rcParams["font.family"] = font_manager.FontProperties(
        fname=JAPANESE_FONT_PATH
    ).get_name()


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp05"
EXPERIMENT_NAME = "重要特徴量のみのモデル"
TARGET_COLUMN: str | None = None

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "split",
}
VARIANTS = {"all": None, "top20": 20, "top10": 10}
VARIANT_LABELS = {"all": "All features", "top20": "Top 20", "top10": "Top 10"}
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_base" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存の実装済み特徴量を、元特徴量単位の選択母集団として生成する。"""
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
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str], int]:
    train_features = calculate_features(train)
    test_features = calculate_features(test)
    generated_count = len(train_features.columns)
    all_missing = [
        column for column in train_features.columns if train_features[column].isna().all()
    ]
    train_features = train_features.drop(columns=all_missing)
    test_features = test_features.drop(columns=all_missing)
    categorical = train_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric = [column for column in train_features.columns if column not in categorical]
    train_features[numeric] = train_features[numeric].astype(float)
    test_features[numeric] = test_features[numeric].astype(float)
    for column in categorical:
        train_features[column] = train_features[column].astype("object")
        test_features[column] = test_features[column].astype("object")
    return train_features, test_features, numeric, categorical, all_missing, generated_count


# ==================================================
# 3. 前処理・モデル・fold内特徴量選択
# ==================================================

def build_pipeline(columns: list[str], categorical_features: list[str]) -> Pipeline:
    categorical = [column for column in columns if column in categorical_features]
    numeric = [column for column in columns if column not in categorical]
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            )
        )
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
    preprocessor = ColumnTransformer(
        transformers=transformers,
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**MODEL_PARAMS)),
        ]
    )


def transformed_to_source_features(
    pipeline: Pipeline,
    selected_columns: list[str],
    categorical_features: list[str],
) -> list[str]:
    """ColumnTransformerの出力順に対応する元特徴量名を返す。"""
    categorical = [c for c in selected_columns if c in categorical_features]
    numeric = [c for c in selected_columns if c not in categorical]
    sources = list(numeric)
    if categorical:
        encoder = (
            pipeline.named_steps["preprocessor"]
            .named_transformers_["categorical"]
            .named_steps["onehot"]
        )
        for column, categories in zip(categorical, encoder.categories_):
            sources.extend([column] * len(categories))
    return sources


def select_top_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    categorical_features: list[str],
    top_k: int | None,
) -> list[str]:
    """当該学習foldだけのsplit importanceで上位の元特徴量を選ぶ。"""
    columns = X_train.columns.tolist()
    if top_k is None or top_k >= len(columns):
        return columns
    ranking_pipeline = build_pipeline(columns, categorical_features)
    ranking_pipeline.fit(X_train, y_train)
    sources = transformed_to_source_features(
        ranking_pipeline, columns, categorical_features
    )
    importances = ranking_pipeline.named_steps["model"].feature_importances_
    ranking = (
        pd.DataFrame({"source": sources, "importance": importances})
        .groupby("source", as_index=False)["importance"]
        .sum()
        .sort_values(["importance", "source"], ascending=[False, True])
    )
    return ranking.head(top_k)["source"].tolist()


# ==================================================
# 4. inner CVによる閾値選択
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
    tied = np.flatnonzero(np.isclose(scores, best_score))
    best_index = tied[np.argmin(np.abs(THRESHOLD_CANDIDATES[tied] - 0.5))]
    return float(THRESHOLD_CANDIDATES[best_index]), float(best_score)


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    top_k: int | None,
    random_state: int,
) -> tuple[float, float]:
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    inner_oof = np.zeros(len(X), dtype=float)
    for train_index, valid_index in inner_cv.split(X, y):
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]
        selected = select_top_features(X_train, y_train, categorical_features, top_k)
        pipeline = build_pipeline(selected, categorical_features)
        pipeline.fit(X_train[selected], y_train)
        inner_oof[valid_index] = pipeline.predict_proba(X.iloc[valid_index][selected])[:, 1]
    return select_threshold(y, inner_oof)


# ==================================================
# 5. ネストクロスバリデーション・モデル比較
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_features: list[str],
    variant: str,
    top_k: int | None,
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
    selected_features_by_fold: list[list[str]] = []
    importance_frames: list[pd.DataFrame] = []
    transformed_feature_counts: list[int] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]
        threshold, inner_f1 = calculate_inner_threshold(
            X_train,
            y_train,
            categorical_features,
            top_k,
            random_state=RANDOM_STATE + fold,
        )
        selected = select_top_features(
            X_train, y_train, categorical_features, top_k
        )
        pipeline = build_pipeline(selected, categorical_features)
        pipeline.fit(X_train[selected], y_train)
        valid_probabilities = pipeline.predict_proba(X_valid[selected])[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)

        oof_probabilities[valid_index] = valid_probabilities
        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(float(fold_f1))
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        selected_features_by_fold.append(selected)

        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        transformed_feature_counts.append(len(transformed_names))
        importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_names, "importance": importances, "fold": fold}
            )
        )
        print(
            f"{variant}, outer fold {fold}: selected={len(selected)}, "
            f"threshold={threshold:.3f}, inner OOF F1={inner_f1:.4f}, "
            f"valid F1={fold_f1:.4f}"
        )

    importance_long = pd.concat(importance_frames, ignore_index=True)
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
        "variant": variant,
        "oof_probabilities": oof_probabilities,
        "oof_predictions": oof_predictions,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "selected_features_by_fold": selected_features_by_fold,
        "transformed_feature_counts": transformed_feature_counts,
        "final_threshold": float(np.mean(fold_thresholds)),
        "mean_f1": float(np.mean(fold_scores)),
        "std_f1": float(np.std(fold_scores)),
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "feature_importance": feature_importance,
    }


def choose_variant(results: dict[str, dict[str, object]]) -> str:
    """平均outer-fold F1を優先し、同点なら少ない特徴量を選ぶ。"""
    complexity = {"all": 22, "top20": 20, "top10": 10}
    return max(
        results,
        key=lambda variant: (
            results[variant]["mean_f1"],
            results[variant]["nested_oof_f1"],
            -complexity[variant],
        ),
    )


# ==================================================
# 6. 結果図・実行
# ==================================================

def plot_feature_importance(
    feature_importance: pd.DataFrame,
    selected_variant: str,
) -> None:
    plot_data = feature_importance.sort_values("importance", ascending=True)
    height = max(7.0, 0.34 * len(plot_data) + 2.2)
    fig, ax = plt.subplots(figsize=(12, height))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color="#2563EB")
    maximum = max(float(plot_data["importance"].max()), 1.0)
    ax.set_xlim(0, maximum * 1.20)
    for bar, value in zip(bars, plot_data["importance"]):
        ax.text(
            bar.get_width() + maximum * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )
    fig.suptitle(
        f"Feature importance — {VARIANT_LABELS[selected_variant]}",
        y=0.998,
        fontsize=14,
    )
    ax.set_title(
        "LightGBM split importance averaged over 5 outer-fold models; "
        "categorical levels are not aggregated",
        fontsize=9,
        pad=10,
    )
    ax.set_xlabel("Mean split importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(
    results: dict[str, dict[str, object]],
    selected_variant: str,
) -> None:
    folds = np.arange(1, OUTER_N_SPLITS + 1)
    width = 0.24
    colors = {"all": "#64748B", "top20": "#2563EB", "top10": "#F59E0B"}
    fig, ax = plt.subplots(figsize=(12, 7))
    for index, variant in enumerate(VARIANTS):
        result = results[variant]
        positions = folds + (index - 1) * width
        bars = ax.bar(
            positions,
            result["fold_scores"],
            width=width,
            color=colors[variant],
            label=VARIANT_LABELS[variant],
        )
        for bar, value in zip(bars, result["fold_scores"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.009,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
        ax.axhline(
            result["mean_f1"],
            color=colors[variant],
            linestyle="--",
            linewidth=1,
            alpha=0.7,
        )
    summaries = []
    for variant in VARIANTS:
        result = results[variant]
        marker = "  SELECTED" if variant == selected_variant else ""
        summaries.append(
            f"{VARIANT_LABELS[variant]}: mean={result['mean_f1']:.4f}, "
            f"std={result['std_f1']:.4f}, nested OOF F1={result['nested_oof_f1']:.4f}{marker}"
        )
    ax.text(
        0.01,
        0.98,
        "\n".join(summaries),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    fig.suptitle("Outer-fold F1 by feature-count variant", y=0.995, fontsize=14)
    ax.set_title(
        "Zero-based bars evaluate untouched outer folds; feature selection and thresholds use training/inner CV only",
        fontsize=9,
        pad=10,
    )
    ax.set_xticks(folds, [f"Fold {fold}" for fold in folds])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X, _, _, categorical, excluded, generated_count = prepare_features(train, test)
    target_candidates = [column for column in train.columns if column not in test.columns]
    if TARGET_COLUMN is not None:
        target_column = TARGET_COLUMN
    elif len(target_candidates) == 1:
        target_column = target_candidates[0]
    else:
        raise ValueError(f"目的変数列を一意に特定できません: {target_candidates}")
    y = train[target_column].astype(int)

    results = {
        variant: run_nested_cv(X, y, categorical, variant, top_k)
        for variant, top_k in VARIANTS.items()
    }
    selected_variant = choose_variant(results)
    selected_result = results[selected_variant]
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    plot_feature_importance(selected_result["feature_importance"], selected_variant)
    plot_f1_scores(results, selected_variant)

    print()
    print("=" * 100)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(
        f"Generated features: {generated_count}; usable source features: {len(X.columns)}; "
        f"excluded all-missing: {excluded}"
    )
    for variant in VARIANTS:
        result = results[variant]
        print(
            f"{variant}: thresholds={result['fold_thresholds']}, "
            f"fold F1={result['fold_scores']}, mean={result['mean_f1']:.4f}, "
            f"std={result['std_f1']:.4f}, nested OOF F1={result['nested_oof_f1']:.4f}, "
            f"transformed counts={result['transformed_feature_counts']}"
        )
    print(
        f"Selected variant: {selected_variant}; "
        f"final threshold: {selected_result['final_threshold']:.4f}; "
        f"transformed features in importance figure: "
        f"{len(selected_result['feature_importance'])}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()
