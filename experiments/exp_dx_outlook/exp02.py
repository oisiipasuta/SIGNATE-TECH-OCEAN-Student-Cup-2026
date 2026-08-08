"""
実験ID: exp02
実験名: 品詞別TF-IDF・SVD + ロジスティック回帰
著者: Codex

目的・仮説:
- 「今後のDX展望」を名詞、動詞、形容詞・形状詞、副詞に分け、それぞれ独立に
  TF-IDF化してからSVD圧縮する。
- 全品詞を一緒にTF-IDF化したときに頻出名詞へ埋もれやすい、動詞・形容詞・副詞の
  情報を独立した低次元ブロックとしてモデルへ渡す。
- 742件の小標本に合わせ、合計36次元に抑えた上でロジスティック回帰の正則化を比較する。

特徴量・前処理:
- 入力: 「今後のDX展望」1列。欠損は空文字列として扱う。
- 名詞TF-IDF -> SVD 16次元、動詞 -> 8次元、形容詞・形状詞 -> 8次元、
  副詞 -> 4次元（合計36次元）。各TF-IDFはmin_df=2。
- MeCab + unidic-liteによる品詞抽出と既存のDXOutlookTfidfSVDを再利用する。
- 学習を伴わない文書単位のMeCab品詞抽出だけは実行開始時に1回キャッシュする。
  語彙、IDF、SVD、標準化はキャッシュせず、各CV学習foldだけでfitする。
- TF-IDF、SVD、標準化は各CV学習foldだけでfitし、検証foldにはtransformのみ行う。

モデル・正則化候補:
- LogisticRegression(solver="saga", max_iter=5000)。
- L2: C=0.01, 0.1, 1, 10
- L1: C=0.01, 0.1, 1
- Elastic Net: l1_ratio=0.5、C=0.1, 1, 10
- 各outer fold内のinner OOF F1だけで、正則化候補と分類閾値を同時に選ぶ。

評価設計:
- outer StratifiedKFold=5、inner StratifiedKFold=4、random_state=42。
- 閾値候補は0.05〜0.95を0.005刻み。outer validationは選択に使わない。
- 最終thresholdはouter foldで選ばれた閾値の平均。
- 重要度は標準化後36特徴に対する各outerモデルの絶対係数の平均であり、
  モデル内での利用度を示すだけで、方向や因果を表さない。

出力:
- experiments/exp_dx_outlook/results/exp02/feature_importance.png
- experiments/exp_dx_outlook/results/exp02/f1_scores.png
- CSV、JSON、予測値、submissionは出力しない。

実行結果（実行後に更新）:
- 入力特徴量数: 1（「今後のDX展望」）
- SVD後特徴量数: 36（名詞16、動詞8、形容詞・形状詞8、副詞4）
- 除外した全欠損特徴量: なし
- outer fold別TF-IDF語彙数（名詞/動詞/形容詞・形状詞/副詞）:
  2038/690/176/65, 2012/688/180/65, 2023/682/176/65,
  1993/691/177/61, 2025/687/175/63
- fold選択正則化: L1 C=0.1, L1 C=0.1, Elastic Net(0.5) C=0.1,
  L1 C=0.1, Elastic Net(0.5) C=0.1
- fold threshold: 0.250, 0.270, 0.275, 0.285, 0.250
- fold F1: 0.6667, 0.5138, 0.5974, 0.5149, 0.5682
- fold F1 mean ± std: 0.5722 ± 0.0571
- nested OOF F1: 0.5664
- 最終threshold: 0.2660
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from calc_features import DXOutlookTfidfSVD


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp02"
EXPERIMENT_NAME = "品詞別TF-IDF・SVD + ロジスティック回帰"
TRAIN_PATH = BASE_DIR / "data" / "train.csv"
TEST_PATH = BASE_DIR / "data" / "test.csv"
RESULT_DIR = BASE_DIR / "experiments" / "exp_dx_outlook" / "results" / EXPERIMENT_ID
TEXT_COLUMN = "今後のDX展望"

OUTER_N_SPLITS = 5
INNER_N_SPLITS = 4
RANDOM_STATE = 42
THRESHOLD_CANDIDATES = np.linspace(0.05, 0.95, 181)


@dataclass(frozen=True)
class PosBlock:
    key: str
    label: str
    parts_of_speech: tuple[str, ...]
    n_components: int


POS_BLOCKS = (
    PosBlock("noun", "名詞", ("名詞",), 16),
    PosBlock("verb", "動詞", ("動詞",), 8),
    PosBlock("adjective", "形容詞・形状詞", ("形容詞", "形状詞"), 8),
    PosBlock("adverb", "副詞", ("副詞",), 4),
)
TOTAL_SVD_COMPONENTS = sum(block.n_components for block in POS_BLOCKS)


@dataclass(frozen=True)
class RegularizationConfig:
    label: str
    c: float
    l1_ratio: float


REGULARIZATION_CONFIGS = (
    *(
        RegularizationConfig(f"L2 / C={c:g}", c, 0.0)
        for c in (0.01, 0.1, 1.0, 10.0)
    ),
    *(
        RegularizationConfig(f"L1 / C={c:g}", c, 1.0)
        for c in (0.01, 0.1, 1.0)
    ),
    *(
        RegularizationConfig(f"ElasticNet(0.5) / C={c:g}", c, 0.5)
        for c in (0.1, 1.0, 10.0)
    ),
)


# ==================================================
# 2. データ読み込み
# ==================================================

def resolve_target_column(train: pd.DataFrame, test: pd.DataFrame) -> str:
    train_only = [column for column in train.columns if column not in test.columns]
    if len(train_only) != 1:
        raise ValueError(f"trainだけに存在する目的変数列が1列ではありません: {train_only}")
    return train_only[0]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str, list[str]]:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    target_column = resolve_target_column(train, test)
    if TEXT_COLUMN not in train or TEXT_COLUMN not in test:
        raise KeyError(f"train/testに `{TEXT_COLUMN}` 列が必要です。")
    excluded_all_missing = [TEXT_COLUMN] if train[TEXT_COLUMN].isna().all() else []
    if excluded_all_missing:
        raise ValueError(f"対象テキスト列が全欠損です: {excluded_all_missing}")
    return (
        train[[TEXT_COLUMN]].copy(),
        test[[TEXT_COLUMN]].copy(),
        train[target_column].astype(int),
        target_column,
        excluded_all_missing,
    )


# ==================================================
# 3. 品詞別TF-IDF・SVDとモデル
# ==================================================

def feature_names() -> list[str]:
    return [
        f"{block.key}_svd_{component + 1:02d}"
        for block in POS_BLOCKS
        for component in range(block.n_components)
    ]


@dataclass
class FittedPosTransformer:
    vectorizer: TfidfVectorizer
    svd: TruncatedSVD


def tokenize_pos_blocks(X: pd.DataFrame) -> pd.DataFrame:
    """学習を伴わないMeCab解析を文書単位で一度だけ行う。"""
    texts = X[TEXT_COLUMN].fillna("").astype(str)
    tokenized = pd.DataFrame(index=X.index)
    for block in POS_BLOCKS:
        tokenizer = DXOutlookTfidfSVD(
            n_components=1,
            text_column=TEXT_COLUMN,
            target_parts_of_speech=block.parts_of_speech,
        )
        tokenized[block.key] = texts.map(
            lambda text: " ".join(tokenizer._tokenize(text))
        )
    return tokenized


def fit_pos_transformers(
    X: pd.DataFrame,
) -> tuple[dict[str, FittedPosTransformer], pd.DataFrame, dict[str, int]]:
    transformers: dict[str, FittedPosTransformer] = {}
    frames: list[pd.DataFrame] = []
    vocabulary_sizes: dict[str, int] = {}
    for block in POS_BLOCKS:
        vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            token_pattern=None,
            lowercase=False,
            min_df=2,
        )
        tfidf = vectorizer.fit_transform(X[block.key])
        if block.n_components > tfidf.shape[1]:
            raise ValueError(
                f"{block.label}のn_components={block.n_components}は、"
                f"学習foldのTF-IDF語彙数{tfidf.shape[1]}以下にしてください。"
            )
        svd = TruncatedSVD(
            n_components=block.n_components,
            random_state=RANDOM_STATE,
        )
        reduced = pd.DataFrame(
            svd.fit_transform(tfidf),
            index=X.index,
            columns=[
            f"{block.key}_svd_{component + 1:02d}"
            for component in range(block.n_components)
            ],
        )
        transformers[block.key] = FittedPosTransformer(vectorizer, svd)
        frames.append(reduced)
        vocabulary_sizes[block.key] = len(vectorizer.vocabulary_)
    features = pd.concat(frames, axis=1)
    if list(features.columns) != feature_names():
        raise RuntimeError("品詞別SVD特徴量名の順序が想定と一致しません。")
    return transformers, features, vocabulary_sizes


def transform_pos_blocks(
    transformers: dict[str, FittedPosTransformer],
    X: pd.DataFrame,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for block in POS_BLOCKS:
        transformer = transformers[block.key]
        tfidf = transformer.vectorizer.transform(X[block.key])
        reduced = pd.DataFrame(
            transformer.svd.transform(tfidf),
            index=X.index,
            columns=[
                f"{block.key}_svd_{component + 1:02d}"
                for component in range(block.n_components)
            ],
        )
        frames.append(reduced)
    return pd.concat(frames, axis=1)


def build_model(config: RegularizationConfig) -> LogisticRegression:
    return LogisticRegression(
        C=config.c,
        l1_ratio=config.l1_ratio,
        solver="saga",
        max_iter=5000,
        tol=1e-4,
        random_state=RANDOM_STATE,
    )


def fit_model(
    X: np.ndarray,
    y: pd.Series,
    config: RegularizationConfig,
) -> LogisticRegression:
    model = build_model(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    return model


# ==================================================
# 4. 正則化・閾値選択
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
    best_score = float(scores.max())
    best_indices = np.flatnonzero(np.isclose(scores, best_score))
    best_index = min(
        best_indices,
        key=lambda index: (abs(THRESHOLD_CANDIDATES[index] - 0.5), index),
    )
    return float(THRESHOLD_CANDIDATES[best_index]), best_score


def select_regularization_and_threshold(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int,
) -> tuple[RegularizationConfig, float, float]:
    inner_cv = StratifiedKFold(
        n_splits=INNER_N_SPLITS,
        shuffle=True,
        random_state=seed,
    )
    probabilities_by_config = {
        config: np.zeros(len(X), dtype=float) for config in REGULARIZATION_CONFIGS
    }

    for inner_train_index, inner_valid_index in inner_cv.split(X, y):
        X_train = X.iloc[inner_train_index]
        X_valid = X.iloc[inner_valid_index]
        y_train = y.iloc[inner_train_index]
        transformers, train_features, _ = fit_pos_transformers(X_train)
        valid_features = transform_pos_blocks(transformers, X_valid)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_features)
        scaled_valid = scaler.transform(valid_features)
        for config in REGULARIZATION_CONFIGS:
            model = fit_model(scaled_train, y_train, config)
            probabilities_by_config[config][inner_valid_index] = model.predict_proba(
                scaled_valid
            )[:, 1]

    best: tuple[float, int, RegularizationConfig, float] | None = None
    for config_index, config in enumerate(REGULARIZATION_CONFIGS):
        threshold, score = select_threshold(y, probabilities_by_config[config])
        candidate = (score, -config_index, config, threshold)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    if best is None:
        raise RuntimeError("正則化候補を選択できませんでした。")
    return best[2], best[3], best[0]


# ==================================================
# 5. ネステッド・クロスバリデーション
# ==================================================

def run_nested_cv(X: pd.DataFrame, y: pd.Series) -> dict[str, object]:
    outer_cv = StratifiedKFold(
        n_splits=OUTER_N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    oof_predictions = np.zeros(len(X), dtype=int)
    fold_scores: list[float] = []
    fold_thresholds: list[float] = []
    fold_configs: list[RegularizationConfig] = []
    vocabulary_sizes: list[dict[str, int]] = []
    importance_frames: list[pd.Series] = []

    for fold, (outer_train_index, outer_valid_index) in enumerate(outer_cv.split(X, y)):
        X_train = X.iloc[outer_train_index].reset_index(drop=True)
        y_train = y.iloc[outer_train_index].reset_index(drop=True)
        X_valid = X.iloc[outer_valid_index]
        y_valid = y.iloc[outer_valid_index]

        config, threshold, inner_f1 = select_regularization_and_threshold(
            X_train,
            y_train,
            RANDOM_STATE + fold + 1,
        )
        transformers, train_features, fold_vocabulary = fit_pos_transformers(X_train)
        valid_features = transform_pos_blocks(transformers, X_valid)
        scaler = StandardScaler()
        scaled_train = scaler.fit_transform(train_features)
        scaled_valid = scaler.transform(valid_features)
        model = fit_model(scaled_train, y_train, config)
        valid_probabilities = model.predict_proba(scaled_valid)[:, 1]
        valid_predictions = (valid_probabilities >= threshold).astype(int)
        fold_f1 = float(f1_score(y_valid, valid_predictions))

        oof_predictions[outer_valid_index] = valid_predictions
        fold_scores.append(fold_f1)
        fold_thresholds.append(threshold)
        fold_configs.append(config)
        vocabulary_sizes.append(fold_vocabulary)
        importance_frames.append(
            pd.Series(
                np.abs(model.coef_[0]),
                index=feature_names(),
                name=f"fold_{fold}",
            )
        )
        print(
            f"Outer fold {fold}: inner F1={inner_f1:.4f}, {config.label}, "
            f"threshold={threshold:.3f}, outer F1={fold_f1:.4f}, "
            f"vocabulary={fold_vocabulary}"
        )

    importance = pd.concat(importance_frames, axis=1)
    importance["mean_abs_coefficient"] = importance.mean(axis=1)
    importance = importance.sort_values("mean_abs_coefficient", ascending=False)
    return {
        "oof_predictions": oof_predictions,
        "fold_scores": fold_scores,
        "fold_thresholds": fold_thresholds,
        "fold_configs": fold_configs,
        "vocabulary_sizes": vocabulary_sizes,
        "feature_importance": importance,
        "nested_oof_f1": float(f1_score(y, oof_predictions)),
        "final_threshold": float(np.mean(fold_thresholds)),
    }


# ==================================================
# 6. 実験結果・可視化
# ==================================================

def configure_japanese_font() -> None:
    candidates = [
        Path("C:/Windows/Fonts/NotoSansJP-VF.ttf"),
        Path("C:/Windows/Fonts/YuGothM.ttc"),
        Path("C:/Windows/Fonts/meiryo.ttc"),
    ]
    for path in candidates:
        if path.exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(
                fname=path
            ).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def plot_feature_importance(feature_importance: pd.DataFrame) -> None:
    values = feature_importance["mean_abs_coefficient"].sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(12, 12))
    positions = np.arange(len(values))
    colors = [
        "#2878B5" if name.startswith("noun_")
        else "#E07A3F" if name.startswith("verb_")
        else "#5B9A49" if name.startswith("adjective_")
        else "#8B65A5"
        for name in values.index
    ]
    ax.barh(positions, values.to_numpy(), color=colors)
    ax.set_yticks(positions)
    ax.set_yticklabels(values.index, fontsize=8)
    ax.set_xlabel("outer-fold平均絶対係数（重要度）")
    ax.set_title(
        "品詞別SVD 36特徴の完全な重要度順位\n"
        "標準化後のロジスティック回帰係数をouter foldで平均（方向・因果ではない）"
    )
    offset = max(float(values.max()) * 0.008, 1e-8)
    for position, value in zip(positions, values.to_numpy()):
        ax.text(value + offset, position, f"{value:.5f}", va="center", fontsize=7)
    ax.set_xlim(0, float(values.max()) * 1.20 if values.max() > 0 else 1.0)
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "feature_importance.png", dpi=160)
    plt.close(fig)


def plot_f1_scores(
    fold_scores: list[float],
    nested_oof_f1: float,
    fold_std: float,
) -> None:
    folds = np.arange(OUTER_N_SPLITS)
    mean_score = float(np.mean(fold_scores))
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(folds, fold_scores, color="#E07A3F", width=0.65)
    ax.axhline(
        mean_score,
        color="#244A64",
        linestyle="--",
        linewidth=1.5,
        label=f"fold mean = {mean_score:.4f}",
    )
    ax.set_ylim(0, 1)
    ax.set_xticks(folds)
    ax.set_xlabel("Outer validation fold（0始まり）")
    ax.set_ylabel("F1")
    ax.set_title(
        "品詞別TF-IDF・SVD Logistic Regression: outer-fold F1\n"
        "各barは未使用outer foldを、inner CV選択の正則化・閾値で評価"
    )
    for bar, score in zip(bars, fold_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.018,
            f"{score:.4f}",
            ha="center",
            va="bottom",
        )
    ax.text(
        0.98,
        0.97,
        f"Nested OOF F1: {nested_oof_f1:.4f}\nFold std: {fold_std:.4f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "f1_scores.png", dpi=160)
    plt.close(fig)


def main() -> None:
    configure_japanese_font()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    X_raw, _X_test, y, target_column, excluded_all_missing = load_data()
    X = tokenize_pos_blocks(X_raw)
    result = run_nested_cv(X, y)
    fold_scores = result["fold_scores"]
    fold_std = float(np.std(fold_scores))
    plot_feature_importance(result["feature_importance"])
    plot_f1_scores(fold_scores, result["nested_oof_f1"], fold_std)

    print()
    print("=" * 80)
    print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
    print(f"Target column: {target_column}")
    print(f"Text column: {TEXT_COLUMN}")
    print("Raw feature count: 1")
    print(f"SVD feature count: {TOTAL_SVD_COMPONENTS}")
    print(f"Excluded all-missing features: {excluded_all_missing}")
    print(f"Vocabulary sizes by outer fold: {result['vocabulary_sizes']}")
    print(
        "Selected regularization: "
        f"{[config.label for config in result['fold_configs']]}"
    )
    print(f"Fold thresholds: {[round(value, 3) for value in result['fold_thresholds']]}")
    print(f"Fold F1: {[round(value, 4) for value in fold_scores]}")
    print(
        f"Fold F1 mean ± std: {np.mean(fold_scores):.4f} ± {fold_std:.4f}"
    )
    print(f"Nested OOF F1: {result['nested_oof_f1']:.4f}")
    print(f"Final threshold: {result['final_threshold']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
