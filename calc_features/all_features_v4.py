"""all_features_v3へ汎用組織図15特徴を加えた特徴量セットv4。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .all_features_v1 import INDUSTRY_MIN_FREQUENCY
from .all_features_v3 import AllFeaturesV3Transformer
from .tree_of_corp import TREE_OF_CORP_FEATURE_COLUMNS, TreeOfCorpTransformer


ALL_FEATURES_V4_TREE_COLUMNS = TREE_OF_CORP_FEATURE_COLUMNS


class AllFeaturesV4Transformer(BaseEstimator, TransformerMixin):
    """v3の23列と汎用組織図15列を結合して38列を返す。

    v3内部の業界統合、TF-IDF、SVDは ``fit`` に渡されたデータだけで学習する。
    組織図は競技固有の完全一致フラグを含めず、汎用的な構造・機能・デジタル
    体制特徴だけを計算する。CVではfoldごとに新しいインスタンスを作る。
    """

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

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AllFeaturesV4Transformer":
        del y
        self.base_transformer_ = AllFeaturesV3Transformer(
            min_df=self.min_df,
            max_features=self.max_features,
            random_state=self.random_state,
            industry_min_frequency=self.industry_min_frequency,
        )
        self.tree_transformer_ = TreeOfCorpTransformer(include_artifact=False)

        base_features = self.base_transformer_.fit_transform(X)
        tree_features = self.tree_transformer_.fit_transform(X)
        duplicated = set(base_features.columns).intersection(tree_features.columns)
        if duplicated:
            raise ValueError(f"all_features_v3と組織図特徴が重複しています: {duplicated}")
        if tuple(tree_features.columns) != ALL_FEATURES_V4_TREE_COLUMNS:
            raise ValueError("組織図特徴のスキーマがall_features_v4の定義と一致しません")

        self.feature_names_out_ = [
            *base_features.columns.tolist(),
            *tree_features.columns.tolist(),
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=[
                "base_transformer_",
                "tree_transformer_",
                "feature_names_out_",
            ],
        )
        combined = pd.concat(
            [
                self.base_transformer_.transform(X),
                self.tree_transformer_.transform(X),
            ],
            axis=1,
        )
        if combined.columns.duplicated().any():
            duplicated = combined.columns[combined.columns.duplicated()].tolist()
            raise ValueError(f"all_features_v4の特徴量名が重複しています: {duplicated}")
        return combined.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def all_features_v4(df: pd.DataFrame) -> pd.DataFrame:
    """1つのDataFrameでv4特徴量を学習・計算する簡易関数。

    検証・テストデータがある場合はデータリークを避けるため、この関数を
    データごとに呼ばず ``AllFeaturesV4Transformer`` を使用する。
    """
    return AllFeaturesV4Transformer().fit_transform(df)


__all__ = [
    "ALL_FEATURES_V4_TREE_COLUMNS",
    "AllFeaturesV4Transformer",
    "all_features_v4",
]
