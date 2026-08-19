"""
実験ID: exp08
実験名: all_features_v3＋組織図特徴量の累積Top-k比較
著者: Codex

目的・仮説:
- submission3のall_features_v3（23列）を基準に、exp04のouter平均split
  importance順で組織図特徴を1列ずつ累積追加し、Top1〜Top15を比較する。
- 少数の組織図特徴だけで、汎用15特徴を全追加したときより安定したF1を得られるか
  検証する。

特徴量・前処理:
- Top-k順位はTOP_K_TREE_FEATURESへ固定し、k=0の基準を含む16構成を評価する。
- AllFeaturesV3TransformerとTreeOfCorpTransformerを各CV学習fold内でfitする。
- 数値列は学習fold中央値で補完し、業界は学習fold最頻値補完後にOne-Hot Encodingする。
- 学習foldで全欠損の列だけを除外する。TF-IDF/SVDを含む全変換はfold内で学習する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3,
  num_leaves=7, random_state=42, importance_type=split。
- outer StratifiedKFold=5、inner StratifiedKFold=4、seed=42。
- 各k・各outer foldの閾値は、そのouter学習部分のinner OOF予測だけから選ぶ。
- nested OOF F1最大のkを採用し、同点は小さいkを優先する。追加候補が基準以下なら
  k=0を推奨する。

出力:
- experiments/exp_tree_of_corp/results/exp08/feature_importance.png
- experiments/exp_tree_of_corp/results/exp08/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果:
- 最良はk=2（DX変革組織有無＋平均分岐数）。生成25列、One-Hot後
  38/37/39/39/38列、全欠損除外なし。
- 最良k=2のfold閾値は0.230/0.185/0.315/0.270/0.235、最終閾値=0.2470。
  fold F1は0.8056/0.8608/0.7733/0.7654/0.8267、平均=0.8063、
  標準偏差=0.0350、nested OOF F1=0.8063。
- k=0はOOF 0.7744・閾値0.3260を、k=15はOOF 0.7989・閾値0.2770を
  再現した。各kのnested OOF F1は次のとおり。
  k=0:0.7744, 1:0.7804, 2:0.8063, 3:0.7901, 4:0.7927, 5:0.7958,
  6:0.7907, 7:0.7938, 8:0.7837, 9:0.8010, 10:0.7918, 11:0.7949,
  12:0.7947, 13:0.8000, 14:0.8000, 15:0.7989。
- 各kのfold閾値（fold 0〜4）は次のとおり。
  k=0:0.415/0.255/0.205/0.445/0.310,
  1:0.240/0.280/0.160/0.270/0.280,
  2:0.230/0.185/0.315/0.270/0.235,
  3:0.275/0.325/0.315/0.440/0.330,
  4:0.200/0.230/0.175/0.405/0.340,
  5:0.235/0.155/0.265/0.410/0.305,
  6:0.175/0.295/0.325/0.225/0.295,
  7:0.155/0.315/0.310/0.220/0.310,
  8:0.165/0.165/0.175/0.350/0.290,
  9:0.225/0.160/0.205/0.340/0.320,
  10:0.165/0.160/0.205/0.345/0.335,
  11:0.165/0.290/0.205/0.255/0.320,
  12:0.310/0.295/0.165/0.270/0.320,
  13:0.345/0.320/0.225/0.330/0.300,
  14:0.345/0.320/0.225/0.330/0.300,
  15:0.345/0.185/0.225/0.330/0.300。
- Top-k探索は同じデータ上の探索であり、小差を未知データでの優劣とは断定しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
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
from sklearn.utils.validation import check_is_fitted


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.all_features_v1 import INDUSTRY_COLUMN  # noqa: E402
from calc_features.all_features_v3 import AllFeaturesV3Transformer  # noqa: E402
from calc_features.tree_of_corp import (  # noqa: E402
    TREE_OF_CORP_FEATURE_COLUMNS,
    TreeOfCorpTransformer,
)
from experiments.exp_tree_of_corp._common import configure_japanese_font  # noqa: E402


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp08"
EXPERIMENT_NAME = "v3＋組織図特徴量の累積Top-k比較"
TARGET_COLUMN = "購入フラグ"

TOP_K_TREE_FEATURES = (
    "DX変革組織有無",
    "平均分岐数",
    "第一階層組織数",
    "デジタル組織数",
    "組織ノード数",
    "組織機能多様性",
    "海外組織数",
    "研究開発組織数",
    "生産・製造組織数",
    "デジタル組織最小階層",
    "調達・購買組織数",
    "IT運用組織有無",
    "組織最大階層",
    "階層解析可能フラグ",
    "デジタル組織経営直下フラグ",
)
K_VALUES = tuple(range(len(TOP_K_TREE_FEATURES) + 1))

MODEL_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": -1,
    "importance_type": "split",
}
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_tree_of_corp" / "results" / EXPERIMENT_ID

LIGHTGBM_NAME_TRANSLATION = str.maketrans(
    {"[": "［", "]": "］", "{": "（", "}": "）", '"': "”", ":": "：", ",": "・"}
)


@dataclass
class PreparedSplit:
    train: pd.DataFrame
    valid: pd.DataFrame
    raw_feature_names: list[str]
    excluded_all_missing: list[str]
    base_feature_count: int


@dataclass
class CandidateAccumulator:
    k: int
    oof_predictions: np.ndarray
    fold_scores: list[float] = field(default_factory=list)
    fold_thresholds: list[float] = field(default_factory=list)
    inner_scores: list[float] = field(default_factory=list)
    transformed_counts: list[int] = field(default_factory=list)
    excluded_all_missing: set[str] = field(default_factory=set)
    importance_frames: list[pd.DataFrame] = field(default_factory=list)


@dataclass
class CandidateResult:
    k: int
    fold_scores: list[float]
    fold_thresholds: list[float]
    inner_scores: list[float]
    generated_feature_count: int
    transformed_counts: list[int]
    excluded_all_missing: list[str]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_threshold: float


# ==================================================
# 2. データ読み込み
# ==================================================

def load_data() -> tuple[pd.DataFrame, pd.Series]:
    train = pd.read_csv(TRAIN_PATH)
    if TARGET_COLUMN not in train.columns:
        raise KeyError(f"train.csvに{TARGET_COLUMN}列がありません")
    y = train[TARGET_COLUMN].astype(int)
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError(f"{TARGET_COLUMN}は0/1である必要があります")
    return train.drop(columns=[TARGET_COLUMN]), y


# ==================================================
# 3. 特徴量前処理
# ==================================================

class CumulativeTreeFeatureBuilder(BaseEstimator, TransformerMixin):
    """all_features_v3へ指定した組織図特徴だけを追加するTransformer。"""

    def __init__(self, tree_columns: tuple[str, ...] = TOP_K_TREE_FEATURES) -> None:
        self.tree_columns = tree_columns

    def fit(self, X: pd.DataFrame, y: Any = None) -> "CumulativeTreeFeatureBuilder":
        del y
        unknown = set(self.tree_columns) - set(TREE_OF_CORP_FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"未定義の組織図特徴があります: {sorted(unknown)}")
        self.base_transformer_ = AllFeaturesV3Transformer(random_state=RANDOM_STATE)
        self.tree_transformer_ = TreeOfCorpTransformer(include_artifact=False)
        base = self.base_transformer_.fit_transform(X)
        tree = self.tree_transformer_.fit_transform(X).loc[:, list(self.tree_columns)]
        duplicated = set(base.columns).intersection(tree.columns)
        if duplicated:
            raise ValueError(f"v3と組織図特徴が重複しています: {sorted(duplicated)}")
        self.base_feature_names_ = base.columns.tolist()
        self.feature_names_out_ = [*self.base_feature_names_, *tree.columns.tolist()]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "base_transformer_",
                "tree_transformer_",
                "base_feature_names_",
                "feature_names_out_",
            ],
        )
        combined = pd.concat(
            [
                self.base_transformer_.transform(X),
                self.tree_transformer_.transform(X).loc[:, list(self.tree_columns)],
            ],
            axis=1,
        )
        if combined.columns.duplicated().any():
            raise ValueError("累積Top-k特徴量に重複列があります")
        return combined.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def build_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in columns if column == INDUSTRY_COLUMN]
    numeric = [column for column in columns if column not in categorical]
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
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
            ),
        ],
        verbose_feature_names_out=False,
    )


def _sanitize_feature_names(names: list[str]) -> list[str]:
    sanitized = [name.translate(LIGHTGBM_NAME_TRANSLATION) for name in names]
    if len(set(sanitized)) != len(sanitized):
        raise ValueError("LightGBM用の安全化後に特徴量名が重複しました")
    return sanitized


def prepare_split(X_train: pd.DataFrame, X_valid: pd.DataFrame) -> PreparedSplit:
    """学習側だけで全Top-k共通の特徴生成・前処理をfitする。"""
    builder = CumulativeTreeFeatureBuilder(TOP_K_TREE_FEATURES)
    train_generated = builder.fit_transform(X_train)
    valid_generated = builder.transform(X_valid)
    raw_names = builder.get_feature_names_out()
    excluded = [name for name in raw_names if train_generated[name].isna().all()]
    kept = [name for name in raw_names if name not in excluded]

    preprocessor = build_preprocessor(kept)
    train_values = preprocessor.fit_transform(train_generated.loc[:, kept])
    valid_values = preprocessor.transform(valid_generated.loc[:, kept])
    transformed_names = _sanitize_feature_names(
        preprocessor.get_feature_names_out().tolist()
    )
    train_frame = pd.DataFrame(
        np.asarray(train_values), index=X_train.index, columns=transformed_names
    )
    valid_frame = pd.DataFrame(
        np.asarray(valid_values), index=X_valid.index, columns=transformed_names
    )
    return PreparedSplit(
        train=train_frame,
        valid=valid_frame,
        raw_feature_names=raw_names,
        excluded_all_missing=excluded,
        base_feature_count=len(builder.base_feature_names_),
    )


def candidate_columns(prepared: PreparedSplit, k: int) -> list[str]:
    if k not in K_VALUES:
        raise ValueError(f"kは{K_VALUES[0]}〜{K_VALUES[-1]}である必要があります: {k}")
    selected_tree = set(TOP_K_TREE_FEATURES[:k])
    all_tree = set(TOP_K_TREE_FEATURES)
    return [
        name
        for name in prepared.train.columns
        if name not in all_tree or name in selected_tree
    ]


# ==================================================
# 4. LightGBM・inner CV閾値選択
# ==================================================

def fit_model(
    prepared: PreparedSplit,
    y_train: pd.Series,
    k: int,
) -> tuple[LGBMClassifier, list[str]]:
    columns = candidate_columns(prepared, k)
    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(prepared.train.loc[:, columns], y_train.loc[prepared.train.index])
    return model, columns


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
    best_score = float(scores.max())
    tied = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        tied,
        key=lambda index: (abs(THRESHOLD_CANDIDATES[index] - 0.5), index),
    )
    return float(THRESHOLD_CANDIDATES[best_index]), best_score


def calculate_inner_results(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    seed: int,
) -> dict[int, tuple[float, float]]:
    probabilities = {k: np.zeros(len(X), dtype=float) for k in K_VALUES}
    cv = StratifiedKFold(n_splits=INNER_N_SPLITS, shuffle=True, random_state=seed)
    for train_index, valid_index in cv.split(X, y):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train = y.iloc[train_index]
        prepared = prepare_split(X_train, X_valid)
        for k in K_VALUES:
            model, columns = fit_model(prepared, y_train, k)
            probabilities[k][valid_index] = model.predict_proba(
                prepared.valid.loc[:, columns]
            )[:, 1]
    return {k: select_threshold(y, probabilities[k]) for k in K_VALUES}


# ==================================================
# 5. ネステッド・クロスバリデーション
# ==================================================

def _finalize_candidate(
    accumulator: CandidateAccumulator,
    y: pd.Series,
    base_feature_count: int,
) -> CandidateResult:
    importance = (
        pd.concat(accumulator.importance_frames, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .reset_index()
    )
    return CandidateResult(
        k=accumulator.k,
        fold_scores=accumulator.fold_scores,
        fold_thresholds=accumulator.fold_thresholds,
        inner_scores=accumulator.inner_scores,
        generated_feature_count=base_feature_count + accumulator.k,
        transformed_counts=accumulator.transformed_counts,
        excluded_all_missing=sorted(accumulator.excluded_all_missing),
        feature_importance=importance,
        nested_oof_f1=float(f1_score(y, accumulator.oof_predictions)),
        final_threshold=float(np.mean(accumulator.fold_thresholds)),
    )


def run_all_candidates(X: pd.DataFrame, y: pd.Series) -> dict[int, CandidateResult]:
    accumulators = {
        k: CandidateAccumulator(k=k, oof_predictions=np.zeros(len(X), dtype=int))
        for k in K_VALUES
    }
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    base_feature_count: int | None = None

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
        y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]
        inner_results = calculate_inner_results(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            seed=RANDOM_STATE + fold + 1,
        )

        prepared = prepare_split(X_train, X_valid)
        if base_feature_count is None:
            base_feature_count = prepared.base_feature_count
        elif base_feature_count != prepared.base_feature_count:
            raise ValueError("outer fold間でv3特徴量数が一致しません")

        for k in K_VALUES:
            threshold, inner_f1 = inner_results[k]
            model, columns = fit_model(prepared, y_train, k)
            probabilities = model.predict_proba(prepared.valid.loc[:, columns])[:, 1]
            predictions = (probabilities >= threshold).astype(int)
            fold_f1 = float(f1_score(y_valid, predictions))
            accumulator = accumulators[k]
            accumulator.oof_predictions[valid_index] = predictions
            accumulator.fold_scores.append(fold_f1)
            accumulator.fold_thresholds.append(threshold)
            accumulator.inner_scores.append(inner_f1)
            accumulator.transformed_counts.append(len(columns))
            selected_raw = set(prepared.raw_feature_names[: base_feature_count + k])
            accumulator.excluded_all_missing.update(
                set(prepared.excluded_all_missing).intersection(selected_raw)
            )
            accumulator.importance_frames.append(
                pd.DataFrame(
                    {
                        "feature": columns,
                        f"fold_{fold}": model.feature_importances_,
                    }
                ).set_index("feature")
            )
            print(
                f"exp08 k={k:02d} fold={fold}: inner F1={inner_f1:.4f}, "
                f"threshold={threshold:.3f}, outer F1={fold_f1:.4f}, "
                f"generated={base_feature_count + k}, transformed={len(columns)}",
                flush=True,
            )
    if base_feature_count is None:
        raise RuntimeError("outer CVが実行されませんでした")
    return {
        k: _finalize_candidate(accumulator, y, base_feature_count)
        for k, accumulator in accumulators.items()
    }


def choose_best_k(results: dict[int, CandidateResult]) -> int:
    baseline = results[0].nested_oof_f1
    best_added_score = max(results[k].nested_oof_f1 for k in K_VALUES if k > 0)
    if best_added_score <= baseline or np.isclose(best_added_score, baseline):
        return 0
    tied = [
        k
        for k in K_VALUES
        if k > 0 and np.isclose(results[k].nested_oof_f1, best_added_score)
    ]
    return min(tied)


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def make_feature_importance_figure(result: CandidateResult) -> plt.Figure:
    data = result.feature_importance.sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(12, max(7.0, 0.42 * len(data) + 3.0)))
    bars = ax.barh(data["feature"], data["importance"], color="#2563EB")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title(f"exp08 最良Top-{result.k}構成の特徴量重要度", fontsize=16, pad=32)
    ax.text(
        0.5,
        1.01,
        "split importance・outer 5-fold平均（利用頻度であり、方向や因果を示さない）",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    ax.set_xlabel("平均 split importance")
    ax.set_ylabel(f"変換後特徴量（全{len(data)}列）")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.15)
    fig.tight_layout()
    return fig


def make_f1_figure(
    results: dict[int, CandidateResult],
    best_k: int,
) -> plt.Figure:
    best = results[best_k]
    fig, (comparison_ax, fold_ax) = plt.subplots(
        2,
        1,
        figsize=(12, 11),
        gridspec_kw={"height_ratios": [1.05, 1.0]},
    )

    ks = np.array(K_VALUES)
    scores = np.array([results[k].nested_oof_f1 for k in K_VALUES])
    comparison_ax.plot(ks, scores, marker="o", color="#2563EB", linewidth=2)
    comparison_ax.scatter([best_k], [best.nested_oof_f1], color="#DC2626", s=90, zorder=3)
    comparison_ax.axhline(
        results[0].nested_oof_f1,
        color="#64748B",
        linestyle="--",
        linewidth=1.5,
        label=f"Top-0 baseline: {results[0].nested_oof_f1:.4f}",
    )
    for k, score in zip(ks, scores):
        comparison_ax.annotate(
            f"{score:.4f}",
            (k, score),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            rotation=45,
        )
    comparison_ax.set_xticks(ks)
    comparison_ax.set_xlabel("累積追加した組織図特徴量数 k")
    comparison_ax.set_ylabel("Nested OOF F1")
    comparison_ax.set_title("Top-0〜Top-15の累積比較", fontsize=15)
    comparison_ax.grid(alpha=0.2)
    comparison_ax.legend(loc="lower right")

    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(best.fold_scores))
    fold_std = float(np.std(best.fold_scores))
    bars = fold_ax.bar(folds, best.fold_scores, color="#E07A3F", width=0.65)
    fold_ax.bar_label(
        bars,
        labels=[f"{score:.4f}" for score in best.fold_scores],
        padding=3,
    )
    fold_ax.axhline(mean_score, color="#244A64", linestyle="--", linewidth=1.5)
    fold_ax.set_ylim(0, 1)
    fold_ax.set_xticks(folds)
    fold_ax.set_xlabel("Outer validation fold（0始まり）")
    fold_ax.set_ylabel("F1")
    fold_ax.set_title(f"最良Top-{best_k}構成のouter fold別F1", fontsize=15, pad=28)
    fold_ax.text(
        0.5,
        1.01,
        "各barは学習に未使用のouter foldを、inner CVだけで選んだ閾値で評価",
        transform=fold_ax.transAxes,
        ha="center",
        va="bottom",
        color="#4B5563",
    )
    fold_ax.text(
        0.98,
        0.56,
        f"Nested OOF F1: {best.nested_oof_f1:.4f}\n"
        f"Fold mean: {mean_score:.4f}\nFold std: {fold_std:.4f}\n"
        f"Final threshold: {best.final_threshold:.4f}",
        transform=fold_ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fold_ax.grid(axis="y", alpha=0.2)
    fold_ax.set_axisbelow(True)
    fig.suptitle("exp08 組織図特徴量の累積Top-k実験", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def print_summary(results: dict[int, CandidateResult], best_k: int, font: str) -> None:
    print("\n" + "=" * 116)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print("k  added_feature                    OOF_F1  mean     std      threshold  folds")
    for k in K_VALUES:
        result = results[k]
        added = "baseline" if k == 0 else TOP_K_TREE_FEATURES[k - 1]
        folds = "/".join(f"{score:.4f}" for score in result.fold_scores)
        print(
            f"{k:>2} {added:<32} {result.nested_oof_f1:.4f}  "
            f"{np.mean(result.fold_scores):.4f}  {np.std(result.fold_scores):.4f}  "
            f"{result.final_threshold:.4f}     {folds}"
        )
        print(
            f"   fold thresholds={[round(value, 3) for value in result.fold_thresholds]}, "
            f"generated={result.generated_feature_count}, "
            f"transformed={result.transformed_counts}, "
            f"excluded={result.excluded_all_missing}"
        )
    best = results[best_k]
    print(f"Best k: {best_k}")
    print(f"Selected tree features: {list(TOP_K_TREE_FEATURES[:best_k])}")
    print(f"Best nested OOF F1: {best.nested_oof_f1:.4f}")
    print(f"Best final threshold: {best.final_threshold:.4f}")
    print(f"Plot font: {font}")
    print("=" * 116)


def main() -> None:
    if set(TOP_K_TREE_FEATURES) != set(TREE_OF_CORP_FEATURE_COLUMNS):
        raise ValueError("Top-k順位は汎用組織図15特徴を過不足なく含む必要があります")
    X, y = load_data()
    results = run_all_candidates(X, y)
    best_k = choose_best_k(results)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font = configure_japanese_font()
    importance_figure = make_feature_importance_figure(results[best_k])
    f1_figure = make_f1_figure(results, best_k)
    importance_figure.savefig(
        RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight"
    )
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(importance_figure)
    plt.close(f1_figure)
    print_summary(results, best_k, font)


if __name__ == "__main__":
    main()
