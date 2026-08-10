"""
実験ID: exp09
実験名: 業界3%未満統合の固定ルール再評価
著者: Codex

目的・仮説:
- all_features_v1の業界統合ルールを、目的変数由来の重要度基準から
  「各学習データ内で出現率3%未満」に変更した固定条件で再評価する。
- 絶対件数ではなく割合を使い、inner/outer foldのサイズが違っても同じ強さの
  統合基準を適用する。
- 業界以外はall_features_v1の18数値特徴量をそのまま使用する。

特徴量・前処理:
- AllFeaturesV1TransformerをPipeline内でfitし、19特徴量を生成する。
- 業界頻度の計算は各inner/outer学習foldだけで行い、3%未満と未出現業界を
  「その他」に統合する。
- 数値18列は中央値補完、業界1列は最頻値補完 + One-Hot Encoding。
- 特徴量生成、頻度統合、補完、One-Hot Encodingは各学習foldだけでfitする。

モデル・CV・閾値:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7、
  random_state=42（exp04・exp07・exp08と同一）。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 各outer foldでinner OOF F1を最大化する分類閾値を選び、未使用のouter validationで
  一度だけ評価する。同点時は0.5に近い分類閾値を選ぶ。
- 最終分類閾値は5個のouter fold閾値の平均とする。

出力:
- experiments/exp_base/results/exp09/feature_importance.png
- experiments/exp_base/results/exp09/f1_scores.png
- CSV/JSON、予測値、submissionは出力しない。

実行結果（2026-08-10）:
- 生成入力特徴量数: 19（数値18、カテゴリ1）。追加の全欠損除外: なし。
- outer foldごとの保持業界数: 13/12/14/14/13。
- fold分類閾値: 0.300/0.275/0.280/0.155/0.265、最終閾値=0.2550。
- inner OOF F1: 0.7152/0.6624/0.6645/0.6575/0.6429。
- outer fold F1: 0.6076/0.7297/0.6420/0.6598/0.7105、平均=0.6699、
  標準偏差=0.0447、nested OOF F1=0.6683。
- 変換後特徴量のfold間和集合: 34。
- 解釈: 固定3%ルールは、1〜36件をinnerで選択したexp08のnested OOF F1=0.6736
  より0.0053低い。一方で業界基準が単純で再現可能であり、件数探索の不安定性や
  選択処理を本番Pipelineへ持ち込まずに済む利点がある。
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

from calc_features.all_features_v1 import (
    INDUSTRY_COLUMN,
    INDUSTRY_MIN_FREQUENCY,
    AllFeaturesV1Transformer,
    all_features_v1,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp09"
EXPERIMENT_NAME = "fixed three-percent industry grouping"
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

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_base" / "results" / EXPERIMENT_ID


# ==================================================
# 2. 特徴量スキーマ・Pipeline
# ==================================================

def resolve_feature_schema(train: pd.DataFrame) -> tuple[list[str], list[str]]:
    """列名と型だけを確定する。実際のfold変換はPipeline内で行う。"""
    preview = all_features_v1(train)
    categorical_features = [INDUSTRY_COLUMN]
    numeric_features = [
        column for column in preview.columns if column not in categorical_features
    ]
    return numeric_features, categorical_features


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """3%頻度統合を含む全変換とLightGBMを構築する。"""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        verbose_feature_names_out=False,
    )
    preprocessor.set_output(transform="pandas")
    return Pipeline(
        steps=[
            (
                "features",
                AllFeaturesV1Transformer(
                    industry_min_frequency=INDUSTRY_MIN_FREQUENCY
                ),
            ),
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**MODEL_PARAMS)),
        ]
    )


# ==================================================
# 3. 分類閾値選択
# ==================================================

def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """F1最大の分類閾値を返し、同点なら0.5に近い値を選ぶ。"""
    scores = np.array(
        [
            f1_score(y_true, (probabilities >= threshold).astype(int))
            for threshold in THRESHOLD_CANDIDATES
        ]
    )
    best_score = float(scores.max())
    best_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = best_indices[
        np.argmin(np.abs(THRESHOLD_CANDIDATES[best_indices] - 0.5))
    ]
    return float(THRESHOLD_CANDIDATES[best_index]), best_score


def calculate_inner_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int,
) -> tuple[float, float]:
    """outer学習部分だけでinner OOF予測を作り、分類閾値を選ぶ。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    inner_oof_probabilities = np.zeros(len(X), dtype=float)
    for train_index, valid_index in inner_cv.split(X, y):
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X.iloc[train_index], y.iloc[train_index])
        inner_oof_probabilities[valid_index] = pipeline.predict_proba(
            X.iloc[valid_index]
        )[:, 1]
    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 4. ネステッドCV
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, object]:
    """固定3%ルールをouter validationで評価する。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    retained_counts: list[int] = []
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
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        retained = pipeline.named_steps["features"].retained_industries_
        retained_counts.append(len(retained))

        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        feature_importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_names, "importance": importances, "fold": fold}
            )
        )
        print(
            f"outer fold {fold}: retained={retained_counts[-1]}, "
            f"threshold={threshold:.3f}, inner OOF F1={inner_f1:.4f}, "
            f"valid F1={fold_f1:.4f}"
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
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "retained_counts": retained_counts,
        "mean_f1": float(np.mean(fold_scores)),
        "std_f1": float(np.std(fold_scores)),
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "final_threshold": float(np.mean(fold_thresholds)),
        "feature_importance": feature_importance,
    }


# ==================================================
# 5. 結果図
# ==================================================

def configure_japanese_font() -> str:
    """利用可能な日本語フォントを設定し、選択名を返す。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Yu Gothic", "Meiryo", "Noto Sans CJK JP", "MS Gothic"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    plt.rcParams["axes.unicode_minus"] = False
    return "matplotlib default"


def plot_feature_importance(result: dict[str, object]) -> None:
    """全変換後特徴量のouter-fold平均split重要度を描く。"""
    importance = result["feature_importance"]
    plot_data = importance.sort_values("importance", ascending=True)
    figure_height = max(9.0, 0.38 * len(plot_data) + 2.8)
    fig, ax = plt.subplots(figsize=(13, figure_height))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color="#10B981")
    max_importance = max(float(plot_data["importance"].max()), 1.0)
    ax.set_xlim(0, max_importance * 1.16)
    for bar, value in zip(bars, plot_data["importance"]):
        ax.text(
            bar.get_width() + max_importance * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8,
        )
    ax.set_title("業界3%未満統合モデルの全特徴量重要度", fontsize=16, pad=32)
    fig.text(
        0.5,
        0.965,
        "LightGBM split importanceを外側5-foldモデルで特徴名を揃えて平均。使用頻度であり、方向性や因果効果ではない。",
        ha="center",
        va="top",
        fontsize=10,
    )
    ax.set_xlabel("平均 split importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(result: dict[str, object]) -> None:
    """outer foldごとのゼロ基準F1を描く。"""
    folds = np.arange(1, OUTER_N_SPLITS + 1)
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.bar(folds, result["fold_scores"], color="#10B981", width=0.65)
    for bar, value, threshold, retained in zip(
        bars,
        result["fold_scores"],
        result["fold_thresholds"],
        result["retained_counts"],
    ):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"F1={value:.4f}\n閾値={threshold:.3f}, 保持={retained}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.axhline(
        result["mean_f1"],
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"fold平均={result['mean_f1']:.4f}",
    )
    ax.text(
        0.01,
        0.97,
        f"nested OOF F1={result['nested_oof_f1']:.4f}\n"
        f"fold標準偏差={result['std_f1']:.4f}\n"
        f"最終分類閾値={result['final_threshold']:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    ax.set_xticks(folds, [f"Fold {fold}" for fold in folds])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right")
    fig.suptitle("業界3%未満統合・outer-fold F1", fontsize=17, y=0.985)
    fig.text(
        0.5,
        0.95,
        "各barは学習に未使用のouter fold。3%統合は各学習fold、分類閾値はinner OOFだけで決定。",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# 6. 実行・結果表示
# ==================================================

def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if TARGET_COLUMN not in train.columns:
        target_candidates = [column for column in train.columns if column not in test.columns]
        if len(target_candidates) != 1:
            raise ValueError(
                "trainにだけ存在する目的変数列を一意に特定できません: "
                f"{target_candidates}"
            )
        target_column = target_candidates[0]
    else:
        target_column = TARGET_COLUMN
    y = train[target_column].astype(int)

    numeric_features, categorical_features = resolve_feature_schema(train)
    result = run_nested_cv(
        train,
        y,
        numeric_features,
        categorical_features,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    plot_feature_importance(result)
    plot_f1_scores(result)

    print()
    print("=" * 100)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Japanese plot font: {font_name}")
    print(f"Industry minimum frequency: {INDUSTRY_MIN_FREQUENCY:.1%}")
    print(f"Positive rate: {y.mean():.4%}")
    print(
        f"Generated input features: {len(numeric_features) + len(categorical_features)} "
        f"(numeric={len(numeric_features)}, categorical={len(categorical_features)})"
    )
    print("Excluded all-missing features after all_features_v1: []")
    print(f"Retained industry counts: {result['retained_counts']}")
    print(f"Fold thresholds: {result['fold_thresholds']}")
    print(f"Inner OOF F1: {result['inner_scores']}")
    print(f"Outer fold F1: {result['fold_scores']}")
    print(f"Fold mean F1: {result['mean_f1']:.4f}")
    print(f"Fold std F1: {result['std_f1']:.4f}")
    print(f"Nested OOF F1: {result['nested_oof_f1']:.4f}")
    print(f"Final threshold: {result['final_threshold']:.4f}")
    print(f"Transformed feature union: {len(result['feature_importance'])}")
    print("=" * 100)


if __name__ == "__main__":
    main()
