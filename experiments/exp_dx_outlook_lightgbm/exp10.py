"""
実験ID: exp10
実験名: SVD成分数のリーク防止ネステッド比較
著者: Codex

目的・仮説:
- 全データで得た重要度を見て第2成分を選び、同じデータで評価する選択バイアスを避ける。
- SVDの先頭1, 2, 3, 5, 10, 20, 30成分をinner CVだけで比較し、未見outer foldで
  成分数選択を含む手順全体の性能を評価する。

特徴量・前処理:
- 入力は「今後のDX展望」1列。MeCab（UniDic）で名詞・動詞を抽出する。
- 各foldの学習部分だけでTF-IDFと30次元TruncatedSVDをfitする。
- 各候補は同じfold内SVDの先頭k成分を使う。重要度による成分の事後選択はしない。
- 全欠損の入力特徴量だけを除外対象とする。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7,
  random_state=42、importance_type=split。
- outer StratifiedKFold=5、inner=4。各outer学習部分のinner OOF F1で成分数と閾値を選ぶ。
- F1同点時は少ない成分数を選び、閾値同点時は0.5に近い値を選ぶ。
- 最終成分数とthresholdは、全学習データ内の4-fold OOFで改めて選ぶ。

出力:
- experiments/exp_dx_outlook_lightgbm/results/exp10/feature_importance.png
- experiments/exp_dx_outlook_lightgbm/results/exp10/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（2026-08-10実行）:
- 入力特徴量数: 1、SVD生成数: 各fold 30、選択後特徴量数: 10, 30, 20, 2, 30
- TF-IDF語彙数: 3912, 3887, 3866, 3835, 3909
- 除外した全欠損特徴量: なし
- outer fold選択成分数: 10, 30, 20, 2, 30
- fold threshold: 0.275, 0.235, 0.295, 0.225, 0.225
- fold F1: 0.5783, 0.6078, 0.6154, 0.4898, 0.5227
- fold F1 mean ± std: 0.5628 ± 0.0489
- nested OOF F1: 0.5612、最終成分数: 2、最終threshold: 0.2250
- 全データ内inner候補F1: 1=0.3906, 2=0.5588, 3=0.5410, 5=0.5347,
  10=0.5404, 20=0.5416, 30=0.5455（設定選択用であり汎化性能の推定値ではない）
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from experiments.exp_dx_outlook_lightgbm._common import (
    INNER_N_SPLITS,
    MODEL_PARAMS,
    OUTER_N_SPLITS,
    RANDOM_STATE,
    build_model,
    build_text_transformer,
    configure_japanese_font,
    load_data,
    make_feature_importance_figure,
    select_threshold,
)


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp10"
EXPERIMENT_NAME = "SVD成分数のリーク防止ネステッド比較"
PARTS_OF_SPEECH = ("名詞", "動詞")
COMPONENT_CANDIDATES = (1, 2, 3, 5, 10, 20, 30)
MAX_COMPONENTS = max(COMPONENT_CANDIDATES)
FEATURE_LABEL = "名詞＋動詞・SVD成分数をinner CVで選択"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook_lightgbm" / "results" / EXPERIMENT_ID


@dataclass
class ComponentSearchResult:
    """成分数選択を含むネステッドCVのメモリ内結果。"""

    y: pd.Series
    oof_predictions: np.ndarray
    fold_scores: list[float]
    fold_thresholds: list[float]
    selected_components: list[int]
    selected_inner_scores: list[float]
    vocabulary_counts: list[int]
    transformed_counts: list[int]
    feature_importance: pd.DataFrame
    nested_oof_f1: float
    final_components: int
    final_threshold: float
    final_candidate_scores: dict[int, float]
    excluded_all_missing: list[str]


def select_first_components(transformed: pd.DataFrame, count: int) -> pd.DataFrame:
    """fold内でfitした30次元SVDから、順序を事前固定した先頭k列だけを返す。"""
    if transformed.shape[1] < count:
        raise ValueError(
            f"要求した成分数={count}が生成済み成分数={transformed.shape[1]}を超えています。"
        )
    selected = transformed.iloc[:, :count].copy()
    selected.columns = [f"dx_outlook_svd_component_{index + 1:02d}" for index in range(count)]
    return selected


# ==================================================
# 2. データ読み込み
# ==================================================

def evaluate_component_candidates(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[int, float, dict[int, float], dict[int, float]]:
    """inner OOFだけで各成分数の最適閾値とF1を比較する。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probabilities = {
        count: np.zeros(len(X), dtype=float) for count in COMPONENT_CANDIDATES
    }

    for train_index, valid_index in inner_cv.split(X, y):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]

        transformer = build_text_transformer(PARTS_OF_SPEECH)
        transformed_train_all = transformer.fit_transform(X_train)
        transformed_valid_all = transformer.transform(X_valid)

        for count in COMPONENT_CANDIDATES:
            transformed_train = select_first_components(transformed_train_all, count)
            transformed_valid = select_first_components(transformed_valid_all, count)
            model = build_model()
            model.fit(transformed_train, y_train)
            probabilities[count][valid_index] = model.predict_proba(transformed_valid)[:, 1]

    thresholds: dict[int, float] = {}
    scores: dict[int, float] = {}
    for count in COMPONENT_CANDIDATES:
        thresholds[count], scores[count] = select_threshold(y, probabilities[count])

    # 同点なら、モデルを単純に保つため候補順（少ない成分数）を優先する。
    selected_count = max(COMPONENT_CANDIDATES, key=lambda count: (scores[count], -count))
    return selected_count, thresholds[selected_count], scores, thresholds


# ==================================================
# 3. 特徴量前処理
# 4. LightGBM
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_nested_component_search(
    X: pd.DataFrame,
    y: pd.Series,
    excluded_all_missing: list[str],
) -> ComponentSearchResult:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    selected_components: list[int] = []
    selected_inner_scores: list[float] = []
    vocabulary_counts: list[int] = []
    transformed_counts: list[int] = []
    importance_series: list[pd.Series] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        selected_count, threshold, candidate_scores, _ = evaluate_component_candidates(
            X_train.reset_index(drop=True),
            y_train.reset_index(drop=True),
            seed=RANDOM_STATE + fold + 1,
        )
        transformer = build_text_transformer(PARTS_OF_SPEECH)
        transformed_train = select_first_components(
            transformer.fit_transform(X_train), selected_count
        )
        transformed_valid = select_first_components(
            transformer.transform(X_valid), selected_count
        )
        model = build_model()
        model.fit(transformed_train, y_train)
        valid_probabilities = model.predict_proba(transformed_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        selected_components.append(selected_count)
        selected_inner_scores.append(candidate_scores[selected_count])
        vocabulary_counts.append(len(transformer.vectorizer_.get_feature_names_out()))
        transformed_counts.append(transformed_train.shape[1])
        importance_series.append(
            pd.Series(
                model.feature_importances_,
                index=transformed_train.columns,
                name=f"fold_{fold}",
                dtype=float,
            )
        )
        print(
            f"Fold {fold}: selected={selected_count}, "
            f"inner F1={candidate_scores[selected_count]:.4f}, threshold={threshold:.3f}, "
            f"outer F1={fold_f1:.4f}, vocabulary={vocabulary_counts[-1]}"
        )

    # あるfoldで未選択だった成分は、そのfoldのモデルでは利用されないため重要度0とする。
    feature_importance = (
        pd.concat(importance_series, axis=1)
        .fillna(0.0)
        .mean(axis=1)
        .rename("importance")
        .sort_values(ascending=False)
        .rename_axis("feature")
        .reset_index()
    )
    final_components, final_threshold, final_candidate_scores, _ = (
        evaluate_component_candidates(X, y, seed=RANDOM_STATE)
    )

    return ComponentSearchResult(
        y=y,
        oof_predictions=oof_predictions,
        fold_scores=fold_scores,
        fold_thresholds=fold_thresholds,
        selected_components=selected_components,
        selected_inner_scores=selected_inner_scores,
        vocabulary_counts=vocabulary_counts,
        transformed_counts=transformed_counts,
        feature_importance=feature_importance,
        nested_oof_f1=float(f1_score(y, oof_predictions)),
        final_components=final_components,
        final_threshold=final_threshold,
        final_candidate_scores=final_candidate_scores,
        excluded_all_missing=excluded_all_missing,
    )


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def make_component_search_f1_figure(result: ComponentSearchResult) -> plt.Figure:
    """outer fold F1と、最終設定用inner候補比較を1枚に分けて示す。"""
    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(result.fold_scores))
    fold_std = float(np.std(result.fold_scores))
    fig, (fold_ax, search_ax) = plt.subplots(1, 2, figsize=(15, 6))

    bars = fold_ax.bar(folds, result.fold_scores, color="#E07A3F", width=0.65)
    fold_ax.bar_label(bars, labels=[f"{score:.4f}" for score in result.fold_scores], padding=3)
    fold_ax.axhline(mean_score, color="#244A64", linestyle="--", linewidth=1.5)
    fold_ax.set_ylim(0, 1)
    fold_ax.set_xticks(folds)
    fold_ax.set_xlabel("Outer validation fold（0始まり）")
    fold_ax.set_ylabel("F1")
    fold_ax.set_title("未見outer foldのF1", pad=18)
    fold_ax.text(
        0.98,
        0.97,
        f"Nested OOF F1: {result.nested_oof_f1:.4f}\n"
        f"Fold mean: {mean_score:.4f}\nFold std: {fold_std:.4f}\n"
        f"選択成分数: {result.selected_components}",
        transform=fold_ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    fold_ax.grid(axis="y", alpha=0.2)

    counts = list(COMPONENT_CANDIDATES)
    scores = [result.final_candidate_scores[count] for count in counts]
    search_ax.plot(counts, scores, marker="o", color="#2563EB", linewidth=2)
    for count, score in zip(counts, scores):
        search_ax.annotate(f"{score:.4f}", (count, score), xytext=(0, 8),
                           textcoords="offset points", ha="center", fontsize=8)
    search_ax.scatter(
        [result.final_components],
        [result.final_candidate_scores[result.final_components]],
        s=120,
        facecolors="none",
        edgecolors="#DC2626",
        linewidths=2,
        label=f"最終選択: {result.final_components}成分",
    )
    search_ax.set_ylim(0, 1)
    search_ax.set_xticks(counts)
    search_ax.set_xlabel("先頭から使用するSVD成分数")
    search_ax.set_ylabel("Inner OOF F1（設定選択用）")
    search_ax.set_title("全学習データ内4-foldでの候補比較", pad=18)
    search_ax.legend(loc="lower right")
    search_ax.grid(alpha=0.2)

    fig.suptitle(f"{EXPERIMENT_ID} SVD成分数のネステッド比較", fontsize=16, y=0.98)
    fig.text(
        0.5,
        0.925,
        "左: 成分数と閾値をinner CVだけで選んだ未見outer評価 / "
        "右: 最終学習設定を選ぶためのinner比較（outer性能ではない）",
        ha="center",
        color="#4B5563",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig


def print_summary(result: ComponentSearchResult, font_name: str) -> None:
    print("\n" + "=" * 88)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Parts of speech: {list(PARTS_OF_SPEECH)}")
    print(f"Component candidates: {list(COMPONENT_CANDIDATES)}")
    print("Raw feature count: 1")
    print(f"Generated SVD features per outer fold: {[MAX_COMPONENTS] * OUTER_N_SPLITS}")
    print(f"Selected feature counts by outer fold: {result.transformed_counts}")
    print(f"TF-IDF vocabulary counts by outer fold: {result.vocabulary_counts}")
    print(f"Excluded all-missing features: {result.excluded_all_missing}")
    print(f"Model params: {MODEL_PARAMS}")
    print(f"Selected components by outer fold: {result.selected_components}")
    print(f"Selected inner F1: {[round(value, 4) for value in result.selected_inner_scores]}")
    print(f"Fold thresholds: {[round(value, 3) for value in result.fold_thresholds]}")
    print(f"Fold F1: {[round(value, 4) for value in result.fold_scores]}")
    print(
        f"Fold F1 mean ± std: {np.mean(result.fold_scores):.4f} ± "
        f"{np.std(result.fold_scores):.4f}"
    )
    print(f"Nested OOF F1: {result.nested_oof_f1:.4f}")
    print(
        "Full-data inner candidate F1: "
        f"{{{', '.join(f'{count}: {score:.4f}' for count, score in result.final_candidate_scores.items())}}}"
    )
    print(f"Final components: {result.final_components}")
    print(f"Final threshold: {result.final_threshold:.4f}")
    print(f"Plot font: {font_name}")
    print("=" * 88)


def main() -> None:
    X, y, excluded = load_data(TRAIN_PATH)
    result = run_nested_component_search(X, y, excluded)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    feature_figure = make_feature_importance_figure(
        result.feature_importance, EXPERIMENT_ID, FEATURE_LABEL
    )
    f1_figure = make_component_search_f1_figure(result)
    feature_figure.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    f1_figure.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(feature_figure)
    plt.close(f1_figure)
    print_summary(result, font_name)


if __name__ == "__main__":
    main()
