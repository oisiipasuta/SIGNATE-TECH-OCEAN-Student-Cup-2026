"""all_features_v1へDX展望のSVD第2成分を加えた特徴量セットv2。"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .all_features_v1 import all_features_v1
from .dx_outlook import DXOutlookTfidfSVD, get_dx_outlook_feature_columns


DX_OUTLOOK_N_COMPONENTS = 30
DX_OUTLOOK_SECOND_FEATURE = get_dx_outlook_feature_columns(
    DX_OUTLOOK_N_COMPONENTS
)[1]
DX_OUTLOOK_PARTS_OF_SPEECH = ("名詞", "動詞")


class AllFeaturesV2Transformer(BaseEstimator, TransformerMixin):
    """v1の19列と、DX展望SVD第2成分を結合して20列を返す。

    TF-IDFとSVDは ``fit`` に渡されたデータだけで学習する。クロス
    バリデーションではfoldごとに新しいインスタンスを作り、学習部分へ
    ``fit_transform``、検証部分へ ``transform`` を適用する。
    """

    def __init__(
        self,
        *,
        min_df: int | float = 1,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: Any = None) -> "AllFeaturesV2Transformer":
        del y
        base_features = all_features_v1(X)
        if DX_OUTLOOK_SECOND_FEATURE in base_features.columns:
            raise ValueError(
                f"all_features_v1と追加特徴量の列名が重複しています: "
                f"{DX_OUTLOOK_SECOND_FEATURE}"
            )

        self.dx_outlook_transformer_ = DXOutlookTfidfSVD(
            n_components=DX_OUTLOOK_N_COMPONENTS,
            target_parts_of_speech=DX_OUTLOOK_PARTS_OF_SPEECH,
            min_df=self.min_df,
            max_features=self.max_features,
            random_state=self.random_state,
        )
        self.dx_outlook_transformer_.fit(X)
        self.feature_names_out_ = [
            *base_features.columns.tolist(),
            DX_OUTLOOK_SECOND_FEATURE,
        ]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["dx_outlook_transformer_", "feature_names_out_"],
        )
        base_features = all_features_v1(X)
        dx_features = self.dx_outlook_transformer_.transform(X).loc[
            :, [DX_OUTLOOK_SECOND_FEATURE]
        ]
        combined = pd.concat([base_features, dx_features], axis=1)
        return combined.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def all_features_v2(df: pd.DataFrame) -> pd.DataFrame:
    """1つのDataFrameでv2特徴量を学習・計算する簡易関数。

    検証・テストデータがある場合はデータリークを避けるため、この関数を
    データごとに呼ばず ``AllFeaturesV2Transformer`` を使用する。
    """
    return AllFeaturesV2Transformer().fit_transform(df)


__all__ = [
    "DX_OUTLOOK_N_COMPONENTS",
    "DX_OUTLOOK_PARTS_OF_SPEECH",
    "DX_OUTLOOK_SECOND_FEATURE",
    "AllFeaturesV2Transformer",
    "all_features_v2",
]
