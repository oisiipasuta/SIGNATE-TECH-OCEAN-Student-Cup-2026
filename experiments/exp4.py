"""
実験ID: exp4
実験名: 業界特徴量の有無・低重要度カテゴリ統合の比較
著者: oisiipasuta

目的・仮説:
- 業界ダミーが汎化性能を改善するか、同一のネステッドCV分割で比較する。
- Exp4-Aは業界ダミーあり、Exp4-Bは業界特徴量なし、Exp4-Cは過去のexp1重要度で
  「業界_化学」より下だったカテゴリを「その他」に統合する。
- Aの平均F1が高くてもfold間標準偏差が大きい場合は、少数業界への過適合を疑う。

Exp4-Cの事前固定ルール:
- 個別に残す業界: 自動車・乗り物、IT、建設・工事、商社、機械、運輸・物流、
  医療・福祉、化学。
- 上記以外の業界は「その他」に統合する。
- このルールはexp1の重要度順位から実験前に固定し、今回の目的変数やCV結果から
  選び直さない。

特徴量・前処理:
- calc_featuresのうちdx_outlook.pyを除く5モジュールをexp1と同じ順序で使用する。
- 数値特徴量は中央値補完、カテゴリ特徴量は最頻値補完 + One-Hot Encoding。
- 補完・One-Hot Encodingは各CV foldの学習データだけでfitする。

モデル・CV・閾値:
- LightGBM: n_estimators=500, learning_rate=0.03, max_depth=3, num_leaves=7。
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 各条件・outer foldでinner OOF F1を最大化する閾値を選び、outer validationには
  一切触れずに評価する。同点時は0.5に近い閾値を採用する。
- 最終閾値は5個のouter fold閾値の平均とする。

出力:
- experiments/results/exp4/feature_importance.png
- experiments/results/exp4/f1_scores.png
- CSV/JSON、予測値、submissionは出力しない。

実行結果（2026-08-08実行後に更新）:
- 入力特徴量数: A=22、B=21、C=22。
- 変換後特徴量数: A=55、B=24、C=33。
- 全欠損のため除外した特徴量: 人材不足フラグ、予算制約フラグ、組織部門数、
  組織階層数、業務種類数、現場課題数、システム刷新フラグ、導入時期フラグ。
- Exp4-A: fold閾値=0.275/0.235/0.285/0.375/0.340、
  fold F1=0.6098/0.6835/0.6410/0.7200/0.6269、平均=0.6562、標準偏差=0.0402、
  nested OOF F1=0.6562、最終閾値=0.3020。
- Exp4-B: fold閾値=0.315/0.220/0.280/0.230/0.130、
  fold F1=0.5385/0.7407/0.6279/0.6437/0.7021、平均=0.6506、標準偏差=0.0692、
  nested OOF F1=0.6526、最終閾値=0.2350。
- Exp4-C: fold閾値=0.270/0.300/0.220/0.375/0.330、
  fold F1=0.6118/0.7397/0.5909/0.7200/0.6849、平均=0.6695、標準偏差=0.0587、
  nested OOF F1=0.6650、最終閾値=0.2990。
- 解釈: CはA比で平均F1が+0.0133、nested OOF F1が+0.0088だが、fold標準偏差は
  +0.0185。低重要度業界の統合は有望だが、Aより安定とはいえず追加seedでの確認が必要。
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

BASE_DIR = Path(__file__).resolve().parent.parent
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

EXPERIMENT_ID = "exp4"
EXPERIMENT_NAME = "industry feature ablation and category grouping"
TARGET_COLUMN = "購入フラグ"
INDUSTRY_COLUMN = "業界"
OTHER_INDUSTRY_LABEL = "その他"

# exp1の平均split importanceで「業界_化学」以上だったカテゴリを事前固定する。
RETAINED_INDUSTRIES = frozenset(
    {
        "自動車・乗り物",
        "IT",
        "建設・工事",
        "商社",
        "機械",
        "運輸・物流",
        "医療・福祉",
        "化学",
    }
)

CONDITIONS = {
    "Exp4-A": "業界ダミーあり",
    "Exp4-B": "業界特徴量なし",
    "Exp4-C": "低重要度業界をその他に統合",
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
RESULT_DIR = BASE_DIR / "experiments" / "results" / EXPERIMENT_ID


# ==================================================
# 2. データ読み込み・特徴量生成
# ==================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """exp1と同じ5つの特徴量生成関数を、同じ順序で適用する。"""
    feature_frames = [
        calculate_execution_features(df),
        calculate_motivation_features(df),
        calculate_adoption_barrier_features(df),
        calculate_necessity_features(df),
        calculate_purchase_timing_features(df),
    ]
    return pd.concat(feature_frames, axis=1)


def prepare_base_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """特徴量を生成し、trainで全欠損の列だけを除外する。"""
    train_features = calculate_features(train)
    test_features = calculate_features(test)
    all_missing_columns = [
        column for column in train_features.columns if train_features[column].isna().all()
    ]
    train_features = train_features.drop(columns=all_missing_columns)
    test_features = test_features.drop(columns=all_missing_columns)
    return train_features, test_features, all_missing_columns


def apply_condition(features: pd.DataFrame, condition: str) -> pd.DataFrame:
    """A/B/Cの業界特徴量ルールを適用したコピーを返す。"""
    result = features.copy()
    if condition == "Exp4-B":
        return result.drop(columns=[INDUSTRY_COLUMN])
    if condition == "Exp4-C":
        industry = result[INDUSTRY_COLUMN].astype("string")
        result[INDUSTRY_COLUMN] = industry.where(
            industry.isin(RETAINED_INDUSTRIES), OTHER_INDUSTRY_LABEL
        )
    elif condition != "Exp4-A":
        raise ValueError(f"未知の比較条件です: {condition}")
    return result


def resolve_feature_types(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """sklearnへ安全に渡せるdtypeに揃える。"""
    categorical_features = train_features.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    numeric_features = [
        column for column in train_features.columns if column not in categorical_features
    ]
    train_features = train_features.copy()
    test_features = test_features.copy()
    train_features[numeric_features] = train_features[numeric_features].astype(float)
    test_features[numeric_features] = test_features[numeric_features].astype(float)
    for column in categorical_features:
        train_features[column] = train_features[column].astype("object")
        test_features[column] = test_features[column].astype("object")
    return train_features, test_features, numeric_features, categorical_features


# ==================================================
# 3. 前処理・モデル
# ==================================================

def build_pipeline(
    numeric_features: list[str],
    categorical_features: list[str],
) -> Pipeline:
    """fold内でfitする前処理とLightGBMを構築する。"""
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
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
            )
        )
    preprocessor = ColumnTransformer(
        transformers=transformers,
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
        pipeline = build_pipeline(numeric_features, categorical_features)
        pipeline.fit(X.iloc[inner_train_index], y.iloc[inner_train_index])
        inner_oof_probabilities[inner_valid_index] = pipeline.predict_proba(
            X.iloc[inner_valid_index]
        )[:, 1]
    return select_threshold(y, inner_oof_probabilities)


# ==================================================
# 5. ネステッドCV・3条件比較
# ==================================================

def run_nested_cv(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    condition: str,
) -> dict[str, object]:
    """指定条件を共通outer splitで評価する。"""
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
        fold_f1 = f1_score(y_valid, valid_predictions)

        oof_probabilities[valid_index] = valid_probabilities
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
            f"{condition}, outer fold {fold}: threshold={threshold:.3f}, "
            f"inner OOF F1={inner_f1:.4f}, valid F1={fold_f1:.4f}"
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
        "oof_probabilities": oof_probabilities,
        "oof_predictions": oof_predictions,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "inner_scores": inner_scores,
        "final_threshold": float(np.mean(fold_thresholds)),
        "mean_f1": float(np.mean(fold_scores)),
        "std_f1": float(np.std(fold_scores)),
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "feature_importance": feature_importance,
    }


# ==================================================
# 6. 結果図・実行
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
    """3条件の全変換後特徴量を、条件別パネルに描く。"""
    max_feature_count = max(
        len(result["feature_importance"])
        for result in results_by_condition.values()
    )
    figure_height = max(8.0, 0.36 * max_feature_count + 2.5)
    fig, axes = plt.subplots(1, 3, figsize=(28, figure_height), squeeze=False)
    colors = ("#2563EB", "#64748B", "#F59E0B")
    for ax, (condition, label), color in zip(
        axes[0], CONDITIONS.items(), colors
    ):
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
        ax.set_title(f"{condition}: {label}", fontsize=13)
        ax.set_xlabel("平均 split importance")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("特徴量重要度（全変換後特徴量）", fontsize=18, y=0.995)
    fig.text(
        0.5,
        0.975,
        "LightGBM split importanceを外側5-foldモデルで平均。使用頻度であり、方向性や因果効果ではない。",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_f1_scores(
    results_by_condition: dict[str, dict[str, object]],
) -> None:
    """outer foldごとのF1と各条件の要約を描く。"""
    folds = np.arange(1, OUTER_N_SPLITS + 1)
    width = 0.24
    colors = ("#2563EB", "#64748B", "#F59E0B")
    fig, ax = plt.subplots(figsize=(13, 8))
    for index, ((condition, label), color) in enumerate(zip(CONDITIONS.items(), colors)):
        result = results_by_condition[condition]
        positions = folds + (index - 1) * width
        bars = ax.bar(
            positions,
            result["fold_scores"],
            width=width,
            color=color,
            label=f"{condition}: {label}",
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
    fig.suptitle("業界特徴量3条件のouter-fold F1比較", fontsize=16, y=0.985)
    fig.text(
        0.5,
        0.95,
        "各barは学習に未使用のouter fold。閾値は対応するouter学習部分のinner OOF予測だけで選択。",
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

    base_train, base_test, excluded = prepare_base_features(train, test)
    feature_sets: dict[str, tuple[pd.DataFrame, list[str], list[str]]] = {}
    for condition in CONDITIONS:
        condition_train = apply_condition(base_train, condition)
        condition_test = apply_condition(base_test, condition)
        condition_train, _, numeric_features, categorical_features = resolve_feature_types(
            condition_train, condition_test
        )
        feature_sets[condition] = (
            condition_train,
            numeric_features,
            categorical_features,
        )

    results_by_condition = {}
    for condition, (X, numeric_features, categorical_features) in feature_sets.items():
        results_by_condition[condition] = run_nested_cv(
            X, y, numeric_features, categorical_features, condition
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
    print(f"Base input features: {len(base_train.columns)}; excluded all-missing: {excluded}")
    print(f"Exp4-C retained industries: {sorted(RETAINED_INDUSTRIES)}")
    for condition in CONDITIONS:
        X, _, _ = feature_sets[condition]
        result = results_by_condition[condition]
        print(
            f"{condition} ({CONDITIONS[condition]}): input={len(X.columns)}, "
            f"transformed={len(result['feature_importance'])}, "
            f"thresholds={result['fold_thresholds']}, fold F1={result['fold_scores']}, "
            f"mean={result['mean_f1']:.4f}, std={result['std_f1']:.4f}, "
            f"nested OOF F1={result['nested_oof_f1']:.4f}, "
            f"final threshold={result['final_threshold']:.4f}"
        )
    print("=" * 100)


if __name__ == "__main__":
    main()
