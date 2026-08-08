"""DX商材の「必要性」に関する特徴量を作成する。

企業概要・組織図を解析する特徴量は、抽出方法を別途検討するため、現時点では
欠損値を返す仮実装としている。
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = (
    "業界",
    "従業員数",
    "事業所数",
    "工場数",
    "店舗数",
)

NECESSITY_FEATURE_COLUMNS = (
    "業界",
    "従業員規模",
    "拠点総数",
    "組織部門数",
    "組織階層数",
    "業務種類数",
)

BUSINESS_CATEGORIES = (
    "生産管理",
    "在庫管理",
    "物流",
    "店舗運営",
    "顧客管理",
    "保守・点検",
    "バックオフィス",
    "データ分析",
)


def _validate_columns(df: pd.DataFrame) -> None:
    """現時点で計算する特徴量に必要な入力カラムを検証する。"""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"必要性特徴量の計算に必要なカラムがありません: {missing}")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """文字列を含む入力を数値へ変換し、変換できない値は欠損値にする。"""
    return pd.to_numeric(df[column], errors="coerce")


def _placeholder_feature(index: pd.Index) -> pd.Series:
    """後で実装する特徴量用の、欠損値だけを持つ列を返す。"""
    return pd.Series(pd.NA, index=index, dtype="Float64")


def _extract_organization_department_count(df: pd.DataFrame) -> pd.Series:
    """組織図から部門数を抽出する（仮実装）。"""
    # TODO: 「組織図」の表記を正規化し、重複を除いた部門数を数える。
    return _placeholder_feature(df.index)


def _extract_organization_hierarchy_count(df: pd.DataFrame) -> pd.Series:
    """組織図から階層数を抽出する（仮実装）。"""
    # TODO: 「組織図」の区切り・インデントを解析し、最大階層数を求める。
    return _placeholder_feature(df.index)


def _extract_business_type_count(df: pd.DataFrame) -> pd.Series:
    """企業概要に含まれる業務カテゴリ数を抽出する（仮実装）。"""
    # TODO: 「企業概要」から BUSINESS_CATEGORIES の各カテゴリの有無を判定し、
    #       該当カテゴリ数（0～8）を数える。
    return _placeholder_feature(df.index)


def calculate_necessity_features(df: pd.DataFrame) -> pd.DataFrame:
    """入力と同じindexを持つ必要性特徴量を返す。

    「従業員規模」は対数変換せず、元の従業員数を数値化した値を使う。
    拠点総数は事業所数・工場数・店舗数の合計で、3項目すべてが欠損の行のみ
    欠損値とする。組織図・企業概要由来の3特徴量は現時点では欠損値となる。
    """
    _validate_columns(df)

    location_columns = ["事業所数", "工場数", "店舗数"]
    locations = pd.concat(
        [_numeric(df, column) for column in location_columns],
        axis=1,
    )

    features = pd.DataFrame(index=df.index)
    features["業界"] = df["業界"]
    features["従業員規模"] = _numeric(df, "従業員数")
    features["拠点総数"] = locations.sum(axis=1, min_count=1)
    features["組織部門数"] = _extract_organization_department_count(df)
    features["組織階層数"] = _extract_organization_hierarchy_count(df)
    features["業務種類数"] = _extract_business_type_count(df)

    return features.loc[:, NECESSITY_FEATURE_COLUMNS]


def add_necessity_features(df: pd.DataFrame) -> pd.DataFrame:
    """元のDataFrameを変更せず、必要性特徴量を追加したコピーを返す。"""
    result = df.copy()
    features = calculate_necessity_features(df)
    for column in NECESSITY_FEATURE_COLUMNS:
        # 「業界」は元データにも存在するため、同じ値で維持される。
        result[column] = features[column]
    return result

