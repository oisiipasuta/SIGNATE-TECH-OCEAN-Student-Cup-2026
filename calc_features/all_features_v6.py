"""all_features_v3へDX変革組織有無だけを加えた特徴量セットv6。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .all_features_v1 import INDUSTRY_MIN_FREQUENCY
from .all_features_v3 import AllFeaturesV3Transformer
from .tree_of_corp import TREE_OF_CORP_FEATURE_COLUMNS, TreeOfCorpTransformer


ALL_FEATURES_V6_TREE_COLUMNS = ("DX変革組織有無",)


class AllFeaturesV6Transformer(BaseEstimator, TransformerMixin):
    """v3の23列とDX変革組織有無を結合して24列を返す。"""

    def __init__(
        self,
        *,
        min_df: int | float = 1,
        max_features: int | None = None,
        random_state: int = 42,
        industry_min_frequency: float = INDUSTRY_MIN_FREQUENCY,
    ) -> None:
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state
        self.industry_min_frequency = industry_min_frequency

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AllFeaturesV6Transformer":
        del y
        if not set(ALL_FEATURES_V6_TREE_COLUMNS).issubset(TREE_OF_CORP_FEATURE_COLUMNS):
            raise ValueError("all_features_v6の組織図特徴が汎用特徴スキーマにありません")
        self.base_transformer_ = AllFeaturesV3Transformer(
            min_df=self.min_df,
            max_features=self.max_features,
            random_state=self.random_state,
            industry_min_frequency=self.industry_min_frequency,
        )
        self.tree_transformer_ = TreeOfCorpTransformer(include_artifact=False)
        base_features = self.base_transformer_.fit_transform(X)
        tree_features = self.tree_transformer_.fit_transform(X).loc[
            :, list(ALL_FEATURES_V6_TREE_COLUMNS)
        ]
        duplicated = set(base_features.columns).intersection(tree_features.columns)
        if duplicated:
            raise ValueError(f"all_features_v3とDX変革組織特徴が重複しています: {duplicated}")
        self.feature_names_out_ = [
            *base_features.columns.tolist(),
            *tree_features.columns.tolist(),
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["base_transformer_", "tree_transformer_", "feature_names_out_"],
        )
        combined = pd.concat(
            [
                self.base_transformer_.transform(X),
                self.tree_transformer_.transform(X).loc[
                    :, list(ALL_FEATURES_V6_TREE_COLUMNS)
                ],
            ],
            axis=1,
        )
        if combined.columns.duplicated().any():
            duplicated = combined.columns[combined.columns.duplicated()].tolist()
            raise ValueError(f"all_features_v6の特徴量名が重複しています: {duplicated}")
        return combined.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def all_features_v6(df: pd.DataFrame) -> pd.DataFrame:
    """1つのDataFrameでv6特徴量を学習・計算する簡易関数。"""
    return AllFeaturesV6Transformer().fit_transform(df)


__all__ = [
    "ALL_FEATURES_V6_TREE_COLUMNS",
    "AllFeaturesV6Transformer",
    "all_features_v6",
]
