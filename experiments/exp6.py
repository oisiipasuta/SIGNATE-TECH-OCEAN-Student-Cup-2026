"""
実験ID: exp6
実験名: 3軸（実行能力・必要性・意欲）のアブレーション
著者: oisiipasuta

目的・仮説:
- DX商材の購入を説明する「実行能力」「必要性」「意欲」の3軸について、
  単独3構成、2軸の組み合わせ3構成、3軸すべての計7構成を比較する。
- 同じCV分割・モデル・閾値選択を用い、予測性能だけでなく購入企業の構造を
  解釈する。ただし、特徴量重要度はモデル内の利用頻度であり因果や効果方向ではない。

軸と特徴量:
- 実行能力（9列）: 財務余力4指標、企業規模2指標、IT部門、
  ソフトウェア投資、セキュリティ整備。
- 必要性（4列）: 現行ツール満足度、DX全体不満度、DX成果不足度、
  DX成果実感度。DX成果不足度は実感度の逆尺度なので両者は完全に冗長だが、
  仮説で指定された両表現をそのまま比較対象に含める。
- 意欲（4列）: DX抵抗感、DX戦略明確度、セミナー参加度、情報収集度。

特徴量生成・前処理:
- 既存のcalc_features関数を再利用し、逆尺度だけを本実験内で生成する。
- 数値特徴量は各学習fold内で中央値補完する。

モデル・評価:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 各外側foldの閾値は、その外側学習データ内のinner OOF F1だけで選ぶ。
- 閾値候補は0.05～0.95（0.005刻み）、同点なら0.5に近い候補を選ぶ。
- 最終thresholdは各構成の外側fold閾値の平均とする。

出力:
- experiments/results/exp6/feature_importance.png（Exp6-G、全17特徴量）
- experiments/results/exp6/f1_scores.png（Exp6-A～Gのfold別F1と集約値）
- CSV/JSON、予測値、submissionは出力しない。

結果（2026-08-08実行後に更新）:
- 入力特徴量数: A=9, B=4, C=4, D=13, E=13, F=8, G=17
- 変換後特徴量数: A=9, B=4, C=4, D=13, E=13, F=8, G=17
- 全欠損のため除外した特徴量: なし
- A: threshold=0.270, 0.365, 0.195, 0.150, 0.140 / final=0.2240
  fold F1=0.5870, 0.5634, 0.5833, 0.5778, 0.5169
  mean ± std=0.5657 ± 0.0257 / nested OOF F1=0.5662
- B: threshold=0.090, 0.150, 0.170, 0.185, 0.115 / final=0.1420
  fold F1=0.3774, 0.4552, 0.4058, 0.4500, 0.4192
  mean ± std=0.4215 ± 0.0288 / nested OOF F1=0.4198
- C: threshold=0.060, 0.150, 0.110, 0.090, 0.095 / final=0.1010
  fold F1=0.4091, 0.3776, 0.3871, 0.4074, 0.4138
  mean ± std=0.3990 ± 0.0141 / nested OOF F1=0.4000
- D: threshold=0.210, 0.275, 0.275, 0.290, 0.345 / final=0.2790
  fold F1=0.5652, 0.6988, 0.6190, 0.6234, 0.5397
  mean ± std=0.6092 ± 0.0549 / nested OOF F1=0.6115
- E: threshold=0.225, 0.270, 0.295, 0.235, 0.160 / final=0.2370
  fold F1=0.6437, 0.5641, 0.6098, 0.6429, 0.6000
  mean ± std=0.6121 ± 0.0297 / nested OOF F1=0.6128
- F: threshold=0.210, 0.170, 0.060, 0.125, 0.085 / final=0.1300
  fold F1=0.3964, 0.3934, 0.3758, 0.4058, 0.4430
  mean ± std=0.4029 ± 0.0223 / nested OOF F1=0.4035
- G: threshold=0.305, 0.180, 0.220, 0.190, 0.095 / final=0.1980
  fold F1=0.5823, 0.6824, 0.6596, 0.6444, 0.7071
  mean ± std=0.6551 ± 0.0421 / nested OOF F1=0.6577
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from matplotlib import font_manager
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features.adoption_barriers import calculate_adoption_barrier_features
from calc_features.excute_capacity import calculate_execution_features
from calc_features.motivation import calculate_motivation_features


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp6"
EXPERIMENT_NAME = "3軸アブレーション"
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

EXECUTION_FEATURES = [
    "営業利益率",
    "営業CFマージン",
    "自己資本比率",
    "借入金比率",
    "ソフトウェア投資比率",
    "IT部門有無",
    "セキュリティ整備度",
    "log_売上",
    "log_従業員数",
]
NECESSITY_FEATURES = [
    "現行ツール満足度",
    "DX全体不満度",
    "DX成果不足度",
    "DX成果実感度",
]
MOTIVATION_FEATURES = [
    "DX抵抗感",
    "DX戦略明確度",
    "セミナー参加度",
    "情報収集度",
]
AXIS_FEATURES = {
    "実行能力": EXECUTION_FEATURES,
    "必要性": NECESSITY_FEATURES,
    "意欲": MOTIVATION_FEATURES,
}
VARIANTS = {
    "A 実行能力": ("実行能力",),
    "B 必要性": ("必要性",),
    "C 意欲": ("意欲",),
    "D 実行能力＋必要性": ("実行能力", "必要性"),
    "E 実行能力＋意欲": ("実行能力", "意欲"),
    "F 必要性＋意欲": ("必要性", "意欲"),
    "G 3軸すべて": ("実行能力", "必要性", "意欲"),
}

TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_axis_features(df: pd.DataFrame) -> pd.DataFrame:
    """既存関数を使って3軸の全17特徴量を作る。"""
    execution = calculate_execution_features(df).loc[:, EXECUTION_FEATURES]
    barriers = calculate_adoption_barrier_features(df)
    motivation = calculate_motivation_features(df)

    necessity = pd.DataFrame(index=df.index)
    necessity["現行ツール満足度"] = barriers["現行ツール満足度"]
    dx_satisfaction = pd.to_numeric(df["アンケート２"], errors="coerce")
    dx_satisfaction = dx_satisfaction.where(dx_satisfaction.between(1, 5))
    necessity["DX全体不満度"] = 6.0 - dx_satisfaction
    necessity["DX成果実感度"] = barriers["DX成果実感度"]
    necessity["DX成果不足度"] = 6.0 - necessity["DX成果実感度"]

    motivation_axis = pd.DataFrame(index=df.index)
    motivation_axis["DX抵抗感"] = barriers["DX抵抗感"]
    for feature in MOTIVATION_FEATURES[1:]:
        motivation_axis[feature] = motivation[feature]

    features = pd.concat([execution, necessity, motivation_axis], axis=1)
    return features.astype(float)


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """train/testを同じ17列へ揃え、全欠損列だけを除外する。"""
    train_features = calculate_axis_features(train)
    test_features = calculate_axis_features(test)
    all_missing = [
        column for column in train_features if train_features[column].isna().all()
    ]
    return (
        train_features.drop(columns=all_missing),
        test_features.drop(columns=all_missing),
        all_missing,
    )


def variant_columns(axis_names: tuple[str, ...]) -> list[str]:
    return [feature for axis in axis_names for feature in AXIS_FEATURES[axis]]


# ==================================================
# 3. 特徴量前処理・モデル
# ==================================================

def build_pipeline() -> Pipeline:
    """foldごとに中央値補完とLightGBMを新規作成する。"""
    imputer = SimpleImputer(strategy="median")
    imputer.set_output(transform="pandas")
    return Pipeline(
        steps=[
            ("imputer", imputer),
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
    random_state: int,
) -> tuple[float, float]:
    """外側学習データのinner OOF予測だけで閾値を決める。"""
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=random_state,
    )
    inner_oof_probabilities = np.zeros(len(X), dtype=float)
    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        pipeline = build_pipeline()
        pipeline.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
        inner_oof_probabilities[inner_valid_index] = pipeline.predict_proba(
            X.iloc[inner_valid_index]
        )[:, 1]
    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 5. 7構成のネストクロスバリデーション
# ==================================================

def run_variant(
    X: pd.DataFrame,
    y: pd.Series,
    collect_importance: bool,
) -> dict[str, object]:
    """1構成を評価し、必要なら全特徴量のfold平均split重要度も返す。"""
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    inner_scores: list[float] = []
    importance_frames: list[pd.DataFrame] = []

    for fold, (train_index, valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[train_index]
        X_valid = X.iloc[valid_index]
        y_train = y.iloc[train_index]
        y_valid = y.iloc[valid_index]

        threshold, inner_f1 = calculate_inner_threshold(
            X_train,
            y_train,
            random_state=RANDOM_STATE + fold + 1,
        )
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)
        valid_probabilities = pipeline.predict_proba(X_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = f1_score(y_valid, valid_predictions)

        oof_predictions[valid_index] = valid_predictions
        fold_scores.append(float(fold_f1))
        fold_thresholds.append(threshold)
        inner_scores.append(inner_f1)

        if collect_importance:
            importance_frames.append(
                pd.DataFrame(
                    {
                        "feature": X.columns,
                        "importance": pipeline.named_steps["model"].feature_importances_,
                        "fold": fold,
                    }
                )
            )

    result: dict[str, object] = {
        "raw_feature_count": len(X.columns),
        "transformed_feature_count": len(X.columns),
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "final_threshold": float(np.mean(fold_thresholds)),
    }
    if collect_importance:
        importance_by_fold = pd.concat(importance_frames).pivot_table(
            index="feature",
            columns="fold",
            values="importance",
            fill_value=0.0,
        )
        result["feature_importance"] = (
            importance_by_fold.reindex(columns=range(OUTER_N_SPLITS), fill_value=0.0)
            .mean(axis=1)
            .rename("importance")
            .reset_index()
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
    return result


def run_ablation(X: pd.DataFrame, y: pd.Series) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for variant, axes in VARIANTS.items():
        columns = variant_columns(axes)
        print(f"\n{variant}: {len(columns)} features")
        result = run_variant(
            X.loc[:, columns],
            y,
            collect_importance=variant.startswith("G "),
        )
        results[variant] = result
        print(
            f"  thresholds={result['fold_thresholds']}\n"
            f"  fold F1={result['fold_scores']}\n"
            f"  mean={np.mean(result['fold_scores']):.4f}, "
            f"std={np.std(result['fold_scores']):.4f}, "
            f"nested OOF={result['nested_oof_f1']:.4f}"
        )
    return results


# ==================================================
# 6. 実験結果・図の出力
# ==================================================

def configure_japanese_font() -> str:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ("Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP"):
        if name in available:
            plt.rcParams["font.family"] = name
            return name
    return str(plt.rcParams["font.family"])


def feature_axis(feature: str) -> str:
    for axis, features in AXIS_FEATURES.items():
        if feature in features:
            return axis
    raise KeyError(feature)


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    """Exp6-Gの全17特徴量を、軸を色分けして降順で描く。"""
    plot_data = feature_importance.sort_values("importance", ascending=True).copy()
    colors = {"実行能力": "#2563EB", "必要性": "#D97706", "意欲": "#059669"}
    bar_colors = [colors[feature_axis(feature)] for feature in plot_data["feature"]]
    fig, ax = plt.subplots(figsize=(12, max(7.5, 0.42 * len(plot_data) + 2.8)))
    bars = ax.barh(plot_data["feature"], plot_data["importance"], color=bar_colors)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_title("Exp6-G 3軸すべての特徴量重要度", fontsize=16, fontweight="bold", pad=30)
    ax.text(
        0.5,
        1.01,
        "LightGBM split importance・外側5-fold平均（モデルの利用頻度であり因果や方向ではない）",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4B5563",
    )
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=axis)
        for axis, color in colors.items()
    ]
    ax.legend(handles=legend_handles, loc="lower right")
    ax.set_xlabel("平均 split importance")
    ax.set_ylabel("特徴量（全17列）")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(results: dict[str, dict[str, object]]) -> None:
    """A～Gごとに外側5-foldのF1を同一スケールで比較する。"""
    fig, axes = plt.subplots(2, 4, figsize=(18, 10), sharex=True, sharey=True)
    flat_axes = axes.ravel()
    variants = list(VARIANTS)
    y_max = min(
        1.0,
        max(max(results[variant]["fold_scores"]) for variant in variants) + 0.12,
    )

    for ax, variant in zip(flat_axes, variants):
        result = results[variant]
        fold_scores = result["fold_scores"]
        fold_mean = float(np.mean(fold_scores))
        fold_std = float(np.std(fold_scores))
        folds = np.arange(OUTER_N_SPLITS)
        bars = ax.bar(folds, fold_scores, color="#2F6B8A", width=0.62)
        ax.bar_label(
            bars,
            labels=[f"{score:.3f}" for score in fold_scores],
            padding=3,
            fontsize=8,
        )
        ax.axhline(fold_mean, color="#DC2626", linestyle="--", linewidth=1.4)
        ax.text(
            0.03,
            0.97,
            f"mean={fold_mean:.4f}\nstd={fold_std:.4f}\nOOF={result['nested_oof_f1']:.4f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#9CA3AF"},
        )
        ax.set_title(variant, fontsize=11, fontweight="bold")
        ax.set_xticks(folds)
        ax.set_ylim(0.0, y_max)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

    summary_ax = flat_axes[-1]
    summary_ax.axis("off")
    ranking = sorted(
        variants,
        key=lambda variant: results[variant]["nested_oof_f1"],
        reverse=True,
    )
    summary_lines = ["nested OOF F1 順位", ""] + [
        f"{rank}. {variant}: {results[variant]['nested_oof_f1']:.4f}"
        for rank, variant in enumerate(ranking, start=1)
    ]
    summary_ax.text(
        0.05,
        0.95,
        "\n".join(summary_lines),
        ha="left",
        va="top",
        fontsize=12,
        linespacing=1.45,
    )

    fig.suptitle("Exp6 3軸アブレーション：外側検証fold別 F1", fontsize=18, fontweight="bold")
    fig.text(
        0.5,
        0.94,
        "各barは学習に未使用の外側fold（0始まり）を評価し、閾値は内側CVだけで選択。全パネル同一スケール。",
        ha="center",
        fontsize=10,
        color="#4B5563",
    )
    fig.supxlabel("外側fold（0始まり）")
    fig.supylabel("F1")
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.91))
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    X, _, all_missing = prepare_features(train, test)
    y = train[TARGET_COLUMN].astype(int)
    results = run_ablation(X, y)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_japanese_font()
    plot_feature_importance(results["G 3軸すべて"]["feature_importance"])
    plot_f1_scores(results)

    print("\n" + "=" * 88)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Excluded all-missing: {all_missing}")
    print(f"Plot font: {font_name}")
    for variant, result in results.items():
        print(
            f"{variant}: raw/transformed={result['raw_feature_count']}/"
            f"{result['transformed_feature_count']}, final threshold="
            f"{result['final_threshold']:.4f}, nested OOF F1="
            f"{result['nested_oof_f1']:.4f}"
        )
    print("=" * 88)


if __name__ == "__main__":
    main()
