"""DX商材の「購買タイミング」に関する特徴量を作成する。

「今後のDX展望」を解析する特徴量は、抽出方法を別途実装するため、現時点では
欠損値を返す仮実装としている。
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = (
    "アンケート２",
    "アンケート６",
    "アンケート７",
    "アンケート８",
)

PURCHASE_TIMING_FEATURE_COLUMNS = (
    "DX全体不満度",
    "DX成果不足度",
    "現行ツール状態",
    "現場課題数",
    "システム刷新フラグ",
    "導入時期フラグ",
)

TOOL_STATUS_CATEGORIES = (
    "ツール未導入",
    "ツール導入済み・不満",
    "ツール導入済み・普通",
    "ツール導入済み・満足",
)

WORKPLACE_ISSUE_CATEGORIES = (
    "老朽化・レガシー",
    "手作業・紙・Excel",
    "データ分断",
    "属人化",
    "人手不足",
    "非効率・二重入力",
    "セキュリティ課題",
)


def _validate_columns(df: pd.DataFrame) -> None:
    """現時点で計算する特徴量に必要な入力カラムを検証する。"""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"購買タイミング特徴量の計算に必要なカラムがありません: {missing}")


def _valid_survey_score(df: pd.DataFrame, column: str) -> pd.Series:
    """アンケート回答を数値化し、1～5以外を欠損値にする。"""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.where(values.between(1, 5)).astype("Float64")


def _calculate_tool_status(df: pd.DataFrame) -> pd.Series:
    """アンケート6・7から現行ツール状態を4分類する。"""
    introduced = pd.to_numeric(df["アンケート６"], errors="coerce")
    satisfaction = _valid_survey_score(df, "アンケート７")

    status = pd.Series(pd.NA, index=df.index, dtype="string")
    status.loc[introduced.eq(2)] = "ツール未導入"
    status.loc[introduced.eq(1) & satisfaction.isin([1, 2])] = (
        "ツール導入済み・不満"
    )
    status.loc[introduced.eq(1) & satisfaction.eq(3)] = (
        "ツール導入済み・普通"
    )
    status.loc[introduced.eq(1) & satisfaction.isin([4, 5])] = (
        "ツール導入済み・満足"
    )
    return status


def _placeholder_feature(index: pd.Index) -> pd.Series:
    """後で実装するDX展望由来特徴量用の欠損列を返す。"""
    return pd.Series(pd.NA, index=index, dtype="Float64")


def _extract_workplace_issue_count(df: pd.DataFrame) -> pd.Series:
    """今後のDX展望から現場課題カテゴリ数を抽出する（仮実装）。"""
    # TODO: 「今後のDX展望」を正規化し、WORKPLACE_ISSUE_CATEGORIES の
    #       各カテゴリの有無を判定して該当数（0～7）を数える。
    return _placeholder_feature(df.index)


def _extract_system_renewal_flag(df: pd.DataFrame) -> pd.Series:
    """今後のDX展望からシステム更新・入替予定を抽出する（仮実装）。"""
    # TODO: 更新・刷新・更改・入替など、近い将来の刷新計画を表す記述を判定する。
    return _placeholder_feature(df.index)


def _extract_introduction_timing_flag(df: pd.DataFrame) -> pd.Series:
    """今後のDX展望から導入時期の明確さを抽出する（仮実装）。"""
    # TODO: 来期・年度内・年月など、具体的な導入時期を表す記述を判定する。
    return _placeholder_feature(df.index)


def calculate_purchase_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """入力と同じindexを持つ購買タイミング特徴量を返す。

    不満度・成果不足度は、それぞれアンケート2・8の有効回答を ``6 - 回答``
    で反転する。アンケート6が「いいえ（2）」ならツール未導入、アンケート6が
    「はい（1）」ならアンケート7の満足度を1～2、3、4～5に分ける。
    DX展望由来の3特徴量は現時点では欠損値となる。
    """
    _validate_columns(df)

    features = pd.DataFrame(index=df.index)
    features["DX全体不満度"] = 6 - _valid_survey_score(df, "アンケート２")
    features["DX成果不足度"] = 6 - _valid_survey_score(df, "アンケート８")
    features["現行ツール状態"] = _calculate_tool_status(df)
    features["現場課題数"] = _extract_workplace_issue_count(df)
    features["システム刷新フラグ"] = _extract_system_renewal_flag(df)
    features["導入時期フラグ"] = _extract_introduction_timing_flag(df)

    return features.loc[:, PURCHASE_TIMING_FEATURE_COLUMNS]


def add_purchase_timing_features(df: pd.DataFrame) -> pd.DataFrame:
    """元のDataFrameを変更せず、購買タイミング特徴量を追加したコピーを返す。"""
    return pd.concat([df.copy(), calculate_purchase_timing_features(df)], axis=1)
