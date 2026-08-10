"""
実験ID: exp08
実験名: 業界頻度閾値1〜36件のネステッド選択
著者: Codex

目的・仮説:
- 業界の出現頻度閾値を1〜36件から選ぶ際の選択バイアスを避けるため、
  各outer学習部分のinner CVだけで頻度閾値と分類閾値を同時選択する。
- 未使用のouter validationでは、inner CVで選ばれた構成を一度だけ評価する。
- 業界以外はall_features_v1と完全に同じ18数値特徴量を使用する。

特徴量・前処理:
- all_features_v1(df)で19特徴量を生成し、業界列だけ入力データの未集約値に戻す。
- 数値18列は中央値補完、業界1列は最頻値補完 + One-Hot Encoding。
- 業界頻度集計、補完、One-Hot Encodingは各学習foldだけでfitする。
- 学習foldで閾値未満または未出現の業界は「その他」へ割り当てる。

モデル・CV・選択規則:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7、
  random_state=42（exp04・exp07と同一）。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 各outer foldのinner OOFで、min_count=1〜36を比較する。各min_countについて
  F1最大の分類閾値を0.05〜0.95から選び、最もinner OOF F1が高いmin_countを採用する。
- min_countが同点なら小さい値、分類閾値が同点なら0.5に近い値を選ぶ。
- outer validationはmin_countと分類閾値のどちらの選択にも使用しない。
- 最終運用設定は全学習データの5-fold OOFで同じ探索を行って決める。この探索値は
  設定決定専用で、性能推定にはouter nested OOF F1だけを使用する。

出力:
- experiments/exp_base/results/exp08/feature_importance.png
- experiments/exp_base/results/exp08/f1_scores.png
- CSV/JSON、予測値、submissionは出力しない。

実行結果（2026-08-10）:
- 入力特徴量数: 19（数値18、カテゴリ1）。全欠損による除外: なし。
- outer foldで選択されたmin_count: 22/10/13/27/20。
- 対応する保持業界数: 10/21/18/7/10。
- 選択された分類閾値: 0.315/0.260/0.275/0.460/0.240（平均0.3100）。
- 選択時のinner OOF F1: 0.7291/0.6835/0.6688/0.6816/0.6541。
- outer fold F1: 0.6410/0.6842/0.6420/0.6765/0.7250、平均=0.6737、
  標準偏差=0.0311、nested OOF F1=0.6736。
- outer選択モデルの変換後特徴量のfold間和集合: 40。
- 全学習データ内の最終設定用5-fold探索: min_count=34、分類閾値=0.3350、
  選択用OOF F1=0.6790。この0.6790は性能推定値として使用しない。
- 解釈: min_countはouter fold間で10〜27と大きく変動し、単一の最適値は不安定。
  1〜36件から選択する手順自体の性能推定はnested OOF F1=0.6736であり、
  exp07でouter結果を直接比較した最高値0.6821より保守的である。最終運用候補は
  min_count=34だが、設定の安定性確認には追加seedまたは外部holdoutが望ましい。
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.all_features_v1 import all_features_v1


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp08"
EXPERIMENT_NAME = "nested industry frequency threshold selection"
TARGET_COLUMN = "購入フラグ"
INDUSTRY_COLUMN = "業界"
OTHER_INDUSTRY_LABEL = "その他"

MIN_COUNT_CANDIDATES = tuple(range(1, 37))
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)

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
FINAL_SELECTION_N_SPLITS = 5
RANDOM_STATE = 42

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_base" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """all_features_v1の19列を生成し、業界だけ未集約の入力値へ戻す。"""
    features = all_features_v1(df).copy()
    features[INDUSTRY_COLUMN] = df[INDUSTRY_COLUMN].astype("object")
    return features


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    """特徴量を生成し、型を揃え、trainで全欠損の列だけを除外する。"""
    train_features = calculate_features(train)
    test_features = calculate_features(test)
    all_missing_columns = [
        column for column in train_features.columns if train_features[column].isna().all()
    ]
    train_features = train_features.drop(columns=all_missing_columns)
    test_features = test_features.drop(columns=all_missing_columns)

    categorical_features = [INDUSTRY_COLUMN]
    numeric_features = [
        column for column in train_features.columns if column not in categorical_features
    ]
    train_features[numeric_features] = train_features[numeric_features].astype(float)
    test_features[numeric_features] = test_features[numeric_features].astype(float)
    train_features[INDUSTRY_COLUMN] = train_features[INDUSTRY_COLUMN].astype("object")
    test_features[INDUSTRY_COLUMN] = test_features[INDUSTRY_COLUMN].astype("object")
    return (
        train_features,
        test_features,
        numeric_features,
        categorical_features,
        all_missing_columns,
    )


# ==================================================
# 3. fold内の頻度統合・前処理・モデル
# ==================================================

class IndustryFrequencyGrouper(TransformerMixin, BaseEstimator):
    """fitデータでmin_count未満の業界を「その他」に統合する。"""

    def __init__(
        self,
        min_count: int,
        industry_column: str = INDUSTRY_COLUMN,
        other_label: str = OTHER_INDUSTRY_LABEL,
    ) -> None:
        self.min_count = min_count
        self.industry_column = industry_column
        self.other_label = other_label

    def fit(self, X: pd.DataFrame, y: object = None) -> IndustryFrequencyGrouper:
        counts = X[self.industry_column].value_counts(dropna=True)
        self.retained_industries_ = frozenset(
            counts.index[counts >= self.min_count].tolist()
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        industry = result[self.industry_column]
        is_retained = industry.isin(self.retained_industries_)
        result.loc[industry.notna() & ~is_retained, self.industry_column] = self.other_label
        return result


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    min_count: int,
) -> Pipeline:
    """fold内でfitする頻度統合、前処理、LightGBMを構築する。"""
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
            ("industry_grouper", IndustryFrequencyGrouper(min_count=min_count)),
            ("preprocessor", preprocessor),
            ("model", LGBMClassifier(**MODEL_PARAMS)),
        ]
    )


# ==================================================
# 4. inner CVでの頻度閾値・分類閾値選択
# ==================================================

def select_classification_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """F1最大の分類閾値を返し、同点なら0.5に最も近い値を選ぶ。"""
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


def search_frequency_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    n_splits: int,
    random_state: int,
) -> dict[str, object]:
    """指定データ内のOOFだけでmin_countと分類閾値を同時に選ぶ。"""
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(cv.split(X, y))
    scores: list[float] = []
    thresholds: list[float] = []

    for min_count in MIN_COUNT_CANDIDATES:
        oof_probabilities = np.zeros(len(X), dtype=float)
        for train_index, valid_index in splits:
            pipeline = build_pipeline(
                numeric_features,
                categorical_features,
                min_count,
            )
            pipeline.fit(X.iloc[train_index], y.iloc[train_index])
            oof_probabilities[valid_index] = pipeline.predict_proba(
                X.iloc[valid_index]
            )[:, 1]
        threshold, score = select_classification_threshold(y, oof_probabilities)
        scores.append(score)
        thresholds.append(threshold)

    score_array = np.asarray(scores)
    best_score = float(score_array.max())
    best_indices = np.flatnonzero(np.isclose(score_array, best_score))
    best_index = int(best_indices[0])
    return {
        "selected_min_count": MIN_COUNT_CANDIDATES[best_index],
        "selected_threshold": thresholds[best_index],
        "selected_score": scores[best_index],
        "scores": scores,
        "thresholds": thresholds,
    }


# ==================================================
# 5. ネステッドCV
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, object]:
    """innerで全選択を完結させ、outer validationで一度だけ評価する。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    selected_min_counts: list[int] = []
    selected_thresholds: list[float] = []
    selected_inner_scores: list[float] = []
    retained_counts: list[int] = []
    inner_score_curves: list[list[float]] = []
    feature_importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y), start=1):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        search = search_frequency_threshold(
            X_train,
            y_train,
            numeric_features,
            categorical_features,
            n_splits=INNER_N_SPLITS,
            random_state=RANDOM_STATE + fold,
        )
        min_count = int(search["selected_min_count"])
        threshold = float(search["selected_threshold"])
        pipeline = build_pipeline(numeric_features, categorical_features, min_count)
        pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        selected_min_counts.append(min_count)
        selected_thresholds.append(threshold)
        selected_inner_scores.append(float(search["selected_score"]))
        inner_score_curves.append(list(search["scores"]))
        retained = pipeline.named_steps["industry_grouper"].retained_industries_
        retained_counts.append(len(retained))

        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        feature_importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_names, "importance": importances, "fold": fold}
            )
        )
        print(
            f"outer fold {fold}: selected min_count={min_count}, "
            f"threshold={threshold:.3f}, inner OOF F1={search['selected_score']:.4f}, "
            f"retained={retained_counts[-1]}, valid F1={fold_f1:.4f}"
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
        "selected_min_counts": selected_min_counts,
        "selected_thresholds": selected_thresholds,
        "selected_inner_scores": selected_inner_scores,
        "retained_counts": retained_counts,
        "inner_score_curves": inner_score_curves,
        "mean_f1": float(np.mean(fold_scores)),
        "std_f1": float(np.std(fold_scores)),
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "mean_outer_threshold": float(np.mean(selected_thresholds)),
        "feature_importance": feature_importance,
    }


# ==================================================
# 6. 結果図
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
    """outerで選択された5モデルの全変換後特徴量重要度を描く。"""
    importance = result["feature_importance"]
    plot_data = importance.sort_values("importance", ascending=True)
    figure_height = max(9.0, 0.38 * len(plot_data) + 2.8)
    fig, ax = plt.subplots(figsize=(13, figure_height))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color="#2563EB")
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
    ax.set_title("ネステッド選択後モデルの全特徴量重要度", fontsize=16, pad=32)
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


def plot_f1_scores(
    result: dict[str, object],
    final_search: dict[str, object],
) -> None:
    """outer F1とinnerのmin_count探索曲線を1枚に描く。"""
    fig, (ax_f1, ax_search) = plt.subplots(
        2,
        1,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )
    folds = np.arange(1, OUTER_N_SPLITS + 1)
    bars = ax_f1.bar(folds, result["fold_scores"], color="#2563EB", width=0.65)
    for bar, value, min_count, threshold in zip(
        bars,
        result["fold_scores"],
        result["selected_min_counts"],
        result["selected_thresholds"],
    ):
        ax_f1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"F1={value:.4f}\n件数={min_count}, 閾値={threshold:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax_f1.axhline(
        result["mean_f1"],
        color="#DC2626",
        linestyle="--",
        linewidth=1.5,
        label=f"fold平均={result['mean_f1']:.4f}",
    )
    ax_f1.text(
        0.01,
        0.97,
        f"nested OOF F1={result['nested_oof_f1']:.4f}\n"
        f"fold標準偏差={result['std_f1']:.4f}",
        transform=ax_f1.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    ax_f1.set_xticks(folds, [f"Fold {fold}" for fold in folds])
    ax_f1.set_ylabel("F1 score")
    ax_f1.set_ylim(0, 1.0)
    ax_f1.grid(axis="y", alpha=0.25)
    ax_f1.legend(loc="lower right")

    colors = plt.cm.viridis(np.linspace(0.12, 0.88, OUTER_N_SPLITS))
    candidates = np.asarray(MIN_COUNT_CANDIDATES)
    for fold, (scores, selected, color) in enumerate(
        zip(
            result["inner_score_curves"],
            result["selected_min_counts"],
            colors,
        ),
        start=1,
    ):
        ax_search.plot(candidates, scores, color=color, alpha=0.72, label=f"Outer {fold}内")
        selected_index = MIN_COUNT_CANDIDATES.index(selected)
        ax_search.scatter(
            selected,
            scores[selected_index],
            color=color,
            s=42,
            zorder=4,
        )
    ax_search.plot(
        candidates,
        final_search["scores"],
        color="#DC2626",
        linewidth=2.2,
        label="全学習データ内（最終設定用）",
    )
    final_min_count = int(final_search["selected_min_count"])
    final_index = MIN_COUNT_CANDIDATES.index(final_min_count)
    ax_search.scatter(
        final_min_count,
        final_search["scores"][final_index],
        color="#DC2626",
        marker="*",
        s=160,
        zorder=5,
        label=f"最終選択={final_min_count}件未満",
    )
    ax_search.set_xlabel("「その他」に統合する最小出現件数 min_count")
    ax_search.set_ylabel("inner OOF F1（分類閾値最適化後）")
    ax_search.set_xticks(np.arange(1, 37))
    ax_search.grid(alpha=0.25)
    ax_search.legend(ncol=2, fontsize=9)

    fig.suptitle("業界頻度閾値1〜36件のネステッド選択", fontsize=17, y=0.99)
    fig.text(
        0.5,
        0.965,
        "上段: 選択に未使用のouter foldをゼロ基準で評価。下段: 各outer学習部分だけで行った探索。",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


# ==================================================
# 7. 実行・結果表示
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

    X, _, numeric_features, categorical_features, excluded = prepare_features(train, test)
    result = run_nested_cv(X, y, numeric_features, categorical_features)
    print("全学習データ内で最終運用設定を選択中...")
    final_search = search_frequency_threshold(
        X,
        y,
        numeric_features,
        categorical_features,
        n_splits=FINAL_SELECTION_N_SPLITS,
        random_state=RANDOM_STATE + 1000,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    plot_feature_importance(result)
    plot_f1_scores(result, final_search)

    print()
    print("=" * 100)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Japanese plot font: {font_name}")
    print(f"Positive rate: {y.mean():.4%}")
    print(
        f"Input features: {len(X.columns)} "
        f"(numeric={len(numeric_features)}, categorical={len(categorical_features)})"
    )
    print(f"Excluded all-missing features: {excluded}")
    print(f"Selected min_counts by outer fold: {result['selected_min_counts']}")
    print(f"Retained industry counts by outer fold: {result['retained_counts']}")
    print(f"Selected thresholds by outer fold: {result['selected_thresholds']}")
    print(f"Selected inner OOF F1 by outer fold: {result['selected_inner_scores']}")
    print(f"Outer fold F1: {result['fold_scores']}")
    print(f"Fold mean F1: {result['mean_f1']:.4f}")
    print(f"Fold std F1: {result['std_f1']:.4f}")
    print(f"Nested OOF F1: {result['nested_oof_f1']:.4f}")
    print(f"Mean outer-selected classification threshold: {result['mean_outer_threshold']:.4f}")
    print(
        f"Final deployment setting (not a performance estimate): "
        f"min_count={final_search['selected_min_count']}, "
        f"threshold={final_search['selected_threshold']:.4f}, "
        f"selection OOF F1={final_search['selected_score']:.4f}"
    )
    print(f"Transformed feature union: {len(result['feature_importance'])}")
    print("=" * 100)


if __name__ == "__main__":
    main()
