"""
実験ID: exp07
実験名: 業界カテゴリの出現頻度による統合閾値比較
著者: Codex

目的・仮説:
- 目的変数を用いた特徴量重要度ではなく、業界カテゴリの出現頻度だけを使って
  少数カテゴリを「その他」に統合し、業界の過学習を抑えられるか比較する。
- 比較条件は全業界を保持、学習fold内で5件未満、10件未満、15件未満、
  20件未満、25件未満、30件未満を「その他」に統合する7条件とする。
- 業界以外はall_features_v1と完全に同じ特徴量を使用し、条件間で変更しない。

特徴量・前処理:
- all_features_v1(df)で19特徴量を生成し、業界列だけ入力データの未集約値に戻す。
- 数値特徴量はall_features_v1の18列、カテゴリ特徴量は業界1列。
- 数値は中央値補完、業界は最頻値補完 + One-Hot Encoding。
- 頻度集計、補完、One-Hot Encodingは各inner/outer学習foldだけでfitする。
- validationにしか現れない業界は「その他」へ割り当てる。

モデル・CV・閾値:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7、
  random_state=42（既存exp04と同一）。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 各条件・outer foldでinner OOF F1を最大化する閾値を選び、outer validationには
  一切触れずに評価する。同点時は0.5に近い閾値を採用する。
- 最終閾値は5個のouter fold閾値の平均とする。

出力:
- experiments/exp_base/results/exp07/feature_importance.png
- experiments/exp_base/results/exp07/f1_scores.png
- CSV/JSON、予測値、submissionは出力しない。

実行結果（2026-08-10、7条件版）:
- 入力特徴量数: 19（数値18、カテゴリ1）。全欠損による除外: なし。
- 変換後特徴量のfold間和集合: A=49、B=44、C=40、D=36、E=33、F=28、G=27。
- Exp07-A: 保持業界数=30/31/29/30/30、閾値=0.310/0.235/0.220/0.155/0.245、
  fold F1=0.6234/0.7000/0.6067/0.6737/0.7342、平均=0.6676、標準偏差=0.0473、
  nested OOF F1=0.6667、最終閾値=0.2330。
- Exp07-B: 保持業界数=24/23/24/24/25、閾値=0.310/0.235/0.220/0.155/0.245、
  fold F1=0.6234/0.7000/0.6067/0.6737/0.7342、平均=0.6676、標準偏差=0.0473、
  nested OOF F1=0.6667、最終閾値=0.2330。
- Exp07-C: 保持業界数=21/21/20/21/20、閾値=0.310/0.260/0.230/0.325/0.235、
  fold F1=0.6234/0.6842/0.6000/0.7105/0.7654、平均=0.6767、標準偏差=0.0597、
  nested OOF F1=0.6750、最終閾値=0.2720。
- Exp07-D: 保持業界数=16/14/16/16/15、閾値=0.285/0.275/0.205/0.445/0.270、
  fold F1=0.6420/0.7297/0.5934/0.7042/0.6757、平均=0.6690、標準偏差=0.0478、
  nested OOF F1=0.6650、最終閾値=0.2960。
- Exp07-E: 保持業界数=11/11/12/13/10、閾値=0.320/0.270/0.300/0.170/0.240、
  fold F1=0.6216/0.7105/0.6410/0.6737/0.7250、平均=0.6744、標準偏差=0.0394、
  nested OOF F1=0.6749、最終閾値=0.2600。
- Exp07-F: 保持業界数=9/9/9/8/9、閾値=0.315/0.295/0.235/0.460/0.270、
  fold F1=0.6329/0.7324/0.6207/0.5970/0.7467、平均=0.6659、標準偏差=0.0614、
  nested OOF F1=0.6649、最終閾値=0.3150。
- Exp07-G: 保持業界数=7/6/6/7/7、閾値=0.360/0.225/0.285/0.380/0.230、
  fold F1=0.5833/0.7317/0.6420/0.7027/0.7407、平均=0.6801、標準偏差=0.0594、
  nested OOF F1=0.6821、最終閾値=0.2960。
- 解釈: nested OOF F1は30件未満統合が最高。次点は10件未満と20件未満である。
  5件未満統合は全保持と予測結果が同一だった。条件差は小さく非単調であり、
  採用判断には追加seedでの確認が必要である。
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

EXPERIMENT_ID = "exp07"
EXPERIMENT_NAME = "industry frequency grouping comparison"
TARGET_COLUMN = "購入フラグ"
INDUSTRY_COLUMN = "業界"
OTHER_INDUSTRY_LABEL = "その他"

CONDITIONS: dict[str, int | None] = {
    "Exp07-A": None,
    "Exp07-B": 5,
    "Exp07-C": 10,
    "Exp07-D": 15,
    "Exp07-E": 20,
    "Exp07-F": 25,
    "Exp07-G": 30,
}
CONDITION_LABELS = {
    "Exp07-A": "全業界を保持",
    "Exp07-B": "5件未満をその他",
    "Exp07-C": "10件未満をその他",
    "Exp07-D": "15件未満をその他",
    "Exp07-E": "20件未満をその他",
    "Exp07-F": "25件未満をその他",
    "Exp07-G": "30件未満をその他",
}

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
        min_count: int | None,
        industry_column: str = INDUSTRY_COLUMN,
        other_label: str = OTHER_INDUSTRY_LABEL,
    ) -> None:
        self.min_count = min_count
        self.industry_column = industry_column
        self.other_label = other_label

    def fit(self, X: pd.DataFrame, y: object = None) -> IndustryFrequencyGrouper:
        if self.min_count is None:
            self.retained_industries_ = None
        else:
            counts = X[self.industry_column].value_counts(dropna=True)
            self.retained_industries_ = frozenset(
                counts.index[counts >= self.min_count].tolist()
            )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result = X.copy()
        if self.retained_industries_ is None:
            return result
        industry = result[self.industry_column]
        is_retained = industry.isin(self.retained_industries_)
        result.loc[industry.notna() & ~is_retained, self.industry_column] = self.other_label
        return result


def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
    min_count: int | None,
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
# 4. 閾値選択
# ==================================================

def select_threshold(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> tuple[float, float]:
    """F1最大の閾値を返し、同点なら0.5に最も近いものを選ぶ。"""
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
    min_count: int | None,
    random_state: int,
) -> tuple[float, float]:
    """outer学習部分だけでinner OOF予測を作り、閾値を選ぶ。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    inner_oof_probabilities = np.zeros(len(X), dtype=float)
    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        pipeline = build_pipeline(numeric_features, categorical_features, min_count)
        pipeline.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
        inner_oof_probabilities[inner_valid_index] = pipeline.predict_proba(
            X.iloc[inner_valid_index]
        )[:, 1]
    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 5. ネステッドCV・7条件比較
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    condition: str,
    min_count: int | None,
) -> dict[str, object]:
    """指定した頻度条件を、共通outer splitで評価する。"""
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
            min_count,
            random_state=RANDOM_STATE + fold,
        )
        pipeline = build_pipeline(numeric_features, categorical_features, min_count)
        pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(float(fold_f1))
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)
        retained = pipeline.named_steps["industry_grouper"].retained_industries_
        retained_counts.append(X_train[INDUSTRY_COLUMN].nunique() if retained is None else len(retained))

        transformed_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
        importances = pipeline.named_steps["model"].feature_importances_
        feature_importance_frames.append(
            pd.DataFrame(
                {"feature": transformed_names, "importance": importances, "fold": fold}
            )
        )
        print(
            f"{condition}, outer fold {fold}: retained={retained_counts[-1]}, "
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
        "condition": condition,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "retained_counts": retained_counts,
        "final_threshold": float(np.mean(fold_thresholds)),
        "mean_f1": float(np.mean(fold_scores)),
        "std_f1": float(np.std(fold_scores)),
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
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


def plot_feature_importance(
    results_by_condition: dict[str, dict[str, object]],
) -> None:
    """7条件の全変換後特徴量を条件別パネルに描く。"""
    max_feature_count = max(
        len(result["feature_importance"]) for result in results_by_condition.values()
    )
    panel_height = max(10.0, 0.30 * max_feature_count + 2.5)
    fig, axes = plt.subplots(2, 4, figsize=(38, panel_height * 2), squeeze=False)
    colors = ("#2563EB", "#64748B", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899", "#0891B2")
    flat_axes = axes.ravel()
    for ax, condition, color in zip(flat_axes, CONDITIONS, colors):
        importance = results_by_condition[condition]["feature_importance"]
        plot_data = importance.sort_values("importance", ascending=True)
        bars = ax.barh(plot_data["feature"], plot_data["importance"], color=color)
        max_importance = max(float(plot_data["importance"].max()), 1.0)
        ax.set_xlim(0, max_importance * 1.20)
        for bar, value in zip(bars, plot_data["importance"]):
            ax.text(
                bar.get_width() + max_importance * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.1f}",
                va="center",
                fontsize=7,
            )
        ax.set_title(f"{condition}: {CONDITION_LABELS[condition]}", fontsize=13)
        ax.set_xlabel("平均 split importance")
        ax.grid(axis="x", alpha=0.25)
    for ax in flat_axes[len(CONDITIONS):]:
        ax.axis("off")
    fig.suptitle("業界の頻度統合条件別・全特徴量重要度", fontsize=18, y=0.995)
    fig.text(
        0.5,
        0.978,
        "LightGBM split importanceを外側5-foldモデルで平均。モデル内の使用頻度であり、方向性や因果効果ではない。",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(
    results_by_condition: dict[str, dict[str, object]],
) -> None:
    """outer foldごとのゼロ基準F1と条件別の要約を描く。"""
    folds = np.arange(1, OUTER_N_SPLITS + 1)
    colors = ("#2563EB", "#64748B", "#F59E0B", "#10B981", "#8B5CF6", "#EC4899", "#0891B2")
    fig, ax = plt.subplots(figsize=(14, 8.5))
    for index, (condition, color) in enumerate(zip(CONDITIONS, colors)):
        result = results_by_condition[condition]
        width = 0.11
        positions = folds + (index - (len(CONDITIONS) - 1) / 2) * width
        bars = ax.bar(
            positions,
            result["fold_scores"],
            width=width,
            color=color,
            label=f"{condition}: {CONDITION_LABELS[condition]}",
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
        ax.axhline(
            result["mean_f1"],
            color=color,
            linestyle="--",
            linewidth=1.2,
            alpha=0.55,
        )

    summaries = []
    for condition in CONDITIONS:
        result = results_by_condition[condition]
        summaries.append(
            f"{condition}: mean={result['mean_f1']:.4f}, "
            f"std={result['std_f1']:.4f}, nested OOF F1={result['nested_oof_f1']:.4f}"
        )
    ax.text(
        0.01,
        0.98,
        "\n".join(summaries),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92},
    )
    fig.suptitle("業界カテゴリの出現頻度7条件・outer-fold F1比較", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.95,
        "各barは学習に未使用のouter fold。頻度統合と閾値選択は対応する学習部分だけで実施。",
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.set_xticks(folds, [f"Fold {fold}" for fold in folds])
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
    results_by_condition: dict[str, dict[str, object]] = {}
    for condition, min_count in CONDITIONS.items():
        results_by_condition[condition] = run_nested_cv(
            X,
            y,
            numeric_features,
            categorical_features,
            condition,
            min_count,
        )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    plot_feature_importance(results_by_condition)
    plot_f1_scores(results_by_condition)

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
    for condition in CONDITIONS:
        result = results_by_condition[condition]
        print(
            f"{condition} ({CONDITION_LABELS[condition]}): "
            f"transformed={len(result['feature_importance'])}, "
            f"retained={result['retained_counts']}, "
            f"thresholds={result['fold_thresholds']}, "
            f"fold F1={result['fold_scores']}, "
            f"mean={result['mean_f1']:.4f}, std={result['std_f1']:.4f}, "
            f"nested OOF F1={result['nested_oof_f1']:.4f}, "
            f"final threshold={result['final_threshold']:.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
