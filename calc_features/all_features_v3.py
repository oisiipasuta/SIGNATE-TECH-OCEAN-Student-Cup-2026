"""all_features_v2へ未使用アンケート3・6・10を加えた特徴量セットv3。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .all_features_v1 import INDUSTRY_MIN_FREQUENCY
from .all_features_v2 import AllFeaturesV2Transformer


ADDITIONAL_SURVEY_COLUMNS = (
    "アンケート３",
    "アンケート６",
    "アンケート１０",
)


def calculate_additional_survey_features(df: pd.DataFrame) -> pd.DataFrame:
    """v2の最終出力に含まれないアンケート3列を有効回答へ変換する。"""
    missing = [column for column in ADDITIONAL_SURVEY_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"all_features_v3の計算に必要なカラムがありません: {missing}")

    features = pd.DataFrame(index=df.index)
    for column in ADDITIONAL_SURVEY_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce")
        features[column] = values.where(values.between(1, 5))
    return features


class AllFeaturesV3Transformer(BaseEstimator, TransformerMixin):
    """v2の20列と未使用アンケート3列を結合して23列を返す。

    v2内部の業界統合、TF-IDF、SVDは ``fit`` に渡されたデータだけで学習する。
    CVではfoldごとに新しいインスタンスを作り、学習部分へ ``fit_transform``、
    検証部分へ ``transform`` を適用する。
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

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AllFeaturesV3Transformer":
        del y
        self.base_transformer_ = AllFeaturesV2Transformer(
            min_df=self.min_df,
            max_features=self.max_features,
            random_state=self.random_state,
            industry_min_frequency=self.industry_min_frequency,
        )
        base_features = self.base_transformer_.fit_transform(X)
        survey_features = calculate_additional_survey_features(X)
        duplicated = set(base_features.columns).intersection(survey_features.columns)
        if duplicated:
            raise ValueError(f"all_features_v2と追加特徴量が重複しています: {duplicated}")

        self.feature_names_out_ = [
            *base_features.columns.tolist(),
            *survey_features.columns.tolist(),
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["base_transformer_", "feature_names_out_"],
        )
        combined = pd.concat(
            [
                self.base_transformer_.transform(X),
                calculate_additional_survey_features(X),
            ],
            axis=1,
        )
        return combined.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def all_features_v3(df: pd.DataFrame) -> pd.DataFrame:
    """1つのDataFrameでv3特徴量を学習・計算する簡易関数。

    検証・テストデータがある場合はデータリークを避けるため、この関数を
    データごとに呼ばず ``AllFeaturesV3Transformer`` を使用する。
    """
    return AllFeaturesV3Transformer().fit_transform(df)


__all__ = [
    "ADDITIONAL_SURVEY_COLUMNS",
    "AllFeaturesV3Transformer",
    "all_features_v3",
    "calculate_additional_survey_features",
]
