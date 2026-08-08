"""DX の「推進意欲」に関するアンケート特徴量を生成する。"""

from __future__ import annotations

import pandas as pd


MOTIVATION_FEATURE_COLUMNS = [
    "DX戦略明確度",
    "情報収集度",
    "セミナー参加度",
]

_SOURCE_COLUMNS = {
    "DX戦略明確度": "アンケート１",
    "情報収集度": "アンケート１１",
    "セミナー参加度": "アンケート９",
}


def calculate_motivation_features(df: pd.DataFrame) -> pd.DataFrame:
    """推進意欲のアンケート特徴量を入力と同じindexで返す。

    アンケート値を数値に変換し、設問の定義外である1～5以外は欠損値にする。
    「今後のDX展望」由来の特徴量は ``dx_outlook.py`` で後から実装する。
    """
    missing = [column for column in _SOURCE_COLUMNS.values() if column not in df.columns]
    if missing:
        raise KeyError(f"特徴量計算に必要なカラムがありません: {missing}")

    result = pd.DataFrame(index=df.index)
    for feature, source in _SOURCE_COLUMNS.items():
        values = pd.to_numeric(df[source], errors="coerce")
        result[feature] = values.where(values.between(1, 5)).astype("Float64")
    return result


def add_motivation_features(df: pd.DataFrame) -> pd.DataFrame:
    """入力を変更せず、推進意欲のアンケート特徴量を追加したコピーを返す。"""
    return pd.concat([df.copy(), calculate_motivation_features(df)], axis=1)
