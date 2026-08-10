"""業界の3%未満統合を含む、学習用特徴量セットv1を生成する。"""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .adoption_barriers import calculate_adoption_barrier_features
from .excute_capacity import calculate_execution_features
from .motivation import calculate_motivation_features
from .necessity import calculate_necessity_features
from .purchase_timing import calculate_purchase_timing_features


INDUSTRY_COLUMN = "業界"
OTHER_INDUSTRY_LABEL = "その他"
INDUSTRY_MIN_FREQUENCY = 0.03

# 実験3で除外した全行欠損特徴量と重複候補。
EXCLUDED_FEATURE_COLUMNS = (
    "人材不足フラグ",
    "予算制約フラグ",
    "組織部門数",
    "組織階層数",
    "業務種類数",
    "現場課題数",
    "システム刷新フラグ",
    "導入時期フラグ",
    "従業員規模",
    "現行ツール状態",
    "赤字・CF不足フラグ",
)


def _calculate_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """業界統合前の19特徴量を返す。"""
    return pd.concat(
        [
            calculate_execution_features(df),
            calculate_motivation_features(df),
            calculate_adoption_barrier_features(df),
            calculate_necessity_features(df),
            calculate_purchase_timing_features(df),
        ],
        axis=1,
    ).drop(columns=list(EXCLUDED_FEATURE_COLUMNS))


def select_retained_industries(
    industry: pd.Series,
    min_frequency: float = INDUSTRY_MIN_FREQUENCY,
) -> frozenset[object]:
    """出現率がmin_frequency以上の業界を返す。"""
    if not 0.0 < min_frequency <= 1.0:
        raise ValueError("min_frequencyは0より大きく1以下で指定してください")
    frequencies = industry.value_counts(normalize=True, dropna=True)
    return frozenset(frequencies.index[frequencies >= min_frequency].tolist())


def _group_industry(
    features: pd.DataFrame,
    retained_industries: Collection[object],
) -> pd.DataFrame:
    """指定された業界以外を「その他」にしたコピーを返す。"""
    result = features.copy()
    industry = result[INDUSTRY_COLUMN].astype("string")
    is_retained = industry.isin(retained_industries)
    result.loc[industry.notna() & ~is_retained, INDUSTRY_COLUMN] = OTHER_INDUSTRY_LABEL
    return result


class AllFeaturesV1Transformer(BaseEstimator, TransformerMixin):
    """学習データで3%未満の業界を決めて19特徴量へ変換する。"""

    def __init__(self, *, industry_min_frequency: float = INDUSTRY_MIN_FREQUENCY) -> None:
        self.industry_min_frequency = industry_min_frequency

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AllFeaturesV1Transformer":
        del y
        features = _calculate_base_features(X)
        self.retained_industries_ = select_retained_industries(
            features[INDUSTRY_COLUMN],
            self.industry_min_frequency,
        )
        self.feature_names_out_ = features.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["retained_industries_", "feature_names_out_"])
        features = _calculate_base_features(X)
        grouped = _group_industry(features, self.retained_industries_)
        return grouped.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def all_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """同じDataFrame内の3%未満業界を統合し、19特徴量を返す。

    ``dx_outlook.py`` を除く5モジュールの特徴量を生成し、実験3で指定した
    11特徴量を除外する。業界はこのDataFrame内の出現率が3%未満なら
    「その他」に統合する。検証・test変換では、学習データのルールを再利用する
    ``AllFeaturesV1Transformer`` を使用する。入力DataFrameは変更しない。
    """
    return AllFeaturesV1Transformer().fit_transform(df)


__all__ = [
    "EXCLUDED_FEATURE_COLUMNS",
    "INDUSTRY_MIN_FREQUENCY",
    "INDUSTRY_COLUMN",
    "OTHER_INDUSTRY_LABEL",
    "AllFeaturesV1Transformer",
    "all_features_v1",
    "select_retained_industries",
]
