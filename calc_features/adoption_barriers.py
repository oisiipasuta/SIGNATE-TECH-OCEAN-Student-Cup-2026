"""「導入障壁・充足済み度」に関する特徴量を作成する。

「今後のDX展望」から抽出する人材不足・予算制約は、抽出方法を後で実装
するため、現時点では欠損値を返す仮実装としている。
"""

from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = (
    "アンケート４",
    "営業利益",
    "営業CF",
    "アンケート７",
    "アンケート８",
)

ADOPTION_BARRIER_FEATURE_COLUMNS = (
    "DX抵抗感",
    "赤字・CF不足フラグ",
    "人材不足フラグ",
    "予算制約フラグ",
    "現行ツール満足度",
    "DX成果実感度",
)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise KeyError(f"導入障壁特徴量の計算に必要なカラムがありません: {missing}")


def _valid_survey_score(df: pd.DataFrame, column: str) -> pd.Series:
    """アンケート回答を数値化し、1～5以外を欠損値にする。"""
    values = pd.to_numeric(df[column], errors="coerce")
    return values.where(values.between(1, 5)).astype("Float64")


def _calculate_financial_constraint_flag(df: pd.DataFrame) -> pd.Series:
    """営業赤字または営業CF不足（負値）を1とする。

    どちらか一方でも負値なら1、観測できた値がすべて0以上なら0、両方とも
    欠損・不正値なら欠損とする。0は赤字・CF流出ではないため0に含める。
    """
    operating_profit = pd.to_numeric(df["営業利益"], errors="coerce")
    operating_cash_flow = pd.to_numeric(df["営業CF"], errors="coerce")

    result = pd.Series(pd.NA, index=df.index, dtype="Int8")
    observed = operating_profit.notna() | operating_cash_flow.notna()
    constrained = operating_profit.lt(0) | operating_cash_flow.lt(0)
    result.loc[observed] = constrained.loc[observed].astype("int8")
    return result


def _placeholder_flag(index: pd.Index) -> pd.Series:
    """後でDX展望から抽出するフラグ用の欠損列を返す。"""
    return pd.Series(pd.NA, index=index, dtype="Int8")


def _extract_talent_shortage_flag(df: pd.DataFrame) -> pd.Series:
    """今後のDX展望からIT・DX人材不足を抽出する（仮実装）。"""
    # TODO: 「今後のDX展望」の人材不足・採用難・スキル不足等を判定する。
    return _placeholder_flag(df.index)


def _extract_budget_constraint_flag(df: pd.DataFrame) -> pd.Series:
    """今後のDX展望からコスト・予算上の制約を抽出する（仮実装）。"""
    # TODO: 「今後のDX展望」の予算不足・費用負担・投資余力等を判定する。
    return _placeholder_flag(df.index)


def calculate_adoption_barrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """入力と同じindexを持つ導入障壁・充足済み度の6特徴量を返す。

    導入障壁はDX抵抗感、財務制約、人材不足、予算制約で表し、充足済み度は
    現行ツール満足度とDX成果実感度で表す。DX展望由来の2フラグは現時点では
    欠損値となる。
    """
    _validate_columns(df)

    features = pd.DataFrame(index=df.index)
    features["DX抵抗感"] = _valid_survey_score(df, "アンケート４")
    features["赤字・CF不足フラグ"] = _calculate_financial_constraint_flag(df)
    features["人材不足フラグ"] = _extract_talent_shortage_flag(df)
    features["予算制約フラグ"] = _extract_budget_constraint_flag(df)
    features["現行ツール満足度"] = _valid_survey_score(df, "アンケート７")
    features["DX成果実感度"] = _valid_survey_score(df, "アンケート８")
    return features.loc[:, ADOPTION_BARRIER_FEATURE_COLUMNS]


def add_adoption_barrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """入力を変更せず、導入障壁・充足済み度の特徴量を追加したコピーを返す。"""
    return pd.concat([df.copy(), calculate_adoption_barrier_features(df)], axis=1)
