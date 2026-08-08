"""実験3・4の結果を反映した、学習用特徴量セット v1 を生成する。"""

from __future__ import annotations

import pandas as pd

from .adoption_barriers import calculate_adoption_barrier_features
from .excute_capacity import calculate_execution_features
from .motivation import calculate_motivation_features
from .necessity import calculate_necessity_features
from .purchase_timing import calculate_purchase_timing_features


INDUSTRY_COLUMN = "業界"
OTHER_INDUSTRY_LABEL = "その他"

# 実験4の平均split importanceで「業界_機械」より上だった業界だけを残す。
# したがって「機械」自体と、それより重要度が低かった業界は「その他」になる。
RETAINED_INDUSTRIES = frozenset(
    {
        "自動車・乗り物",
        "IT",
        "建設・工事",
        "商社",
    }
)

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


def all_features_v1(df: pd.DataFrame) -> pd.DataFrame:
    """実験3・4で採用したルールを一括適用して19特徴量を返す。

    ``dx_outlook.py`` を除く5モジュールの特徴量を生成し、実験3で指定した
    11特徴量を除外する。さらに、業界は実験4-Dに従い「機械」以下の重要度
    だったカテゴリを「その他」に統合する。入力DataFrameは変更しない。
    """
    features = pd.concat(
        [
            calculate_execution_features(df),
            calculate_motivation_features(df),
            calculate_adoption_barrier_features(df),
            calculate_necessity_features(df),
            calculate_purchase_timing_features(df),
        ],
        axis=1,
    ).drop(columns=list(EXCLUDED_FEATURE_COLUMNS))

    industry = features[INDUSTRY_COLUMN].astype("string")
    features[INDUSTRY_COLUMN] = industry.where(
        industry.isin(RETAINED_INDUSTRIES), OTHER_INDUSTRY_LABEL
    )
    return features


__all__ = [
    "EXCLUDED_FEATURE_COLUMNS",
    "INDUSTRY_COLUMN",
    "OTHER_INDUSTRY_LABEL",
    "RETAINED_INDUSTRIES",
    "all_features_v1",
]
