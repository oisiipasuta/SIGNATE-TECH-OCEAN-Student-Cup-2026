"""企業の「実行能力」に関する特徴量を計算する。

コマンドライン例:
    python -m calc_features.calculate data/train.csv calc_features/train_features.csv
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


SOFTWARE_CHANGE_COLUMN = "無形固定資産変動(ソフトウェア関連)"

REQUIRED_COLUMNS = (
    "売上",
    "営業利益",
    "営業CF",
    "自己資本",
    "総資産",
    "短期借入金",
    "長期借入金",
    SOFTWARE_CHANGE_COLUMN,
    "従業員数",
    "組織図",
    "アンケート５",
)

FEATURE_COLUMNS = (
    "営業利益率",
    "営業CFマージン",
    "自己資本比率",
    "借入金比率",
    "ソフトウェア投資比率",
    "IT部門有無",
    "セキュリティ整備度",
    "log_売上",
    "log_従業員数",
)

# 「システム開発部」のような事業部まで拾わないよう、IT/DXを担当すると
# 判断できる名称と組織単位（部・室・課など）の組み合わせに限定する。
IT_DEPARTMENT_PATTERN = re.compile(
    r"(?:"
    r"情報システム|情報技術|社内システム|"
    r"IT(?:戦略|推進|企画|管理|統括|システム)?|"
    r"ICT(?:戦略|推進|企画|管理|統括)?|"
    r"DX(?:戦略|推進|企画|統括)|"
    r"デジタル(?:戦略|推進|企画|改革|統括)"
    r")(?:本部|部門|センター|部|室|課|チーム|グループ)",
    flags=re.IGNORECASE,
)


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"特徴量計算に必要なカラムがありません: {missing}")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """文字列が混じっていても数値へ変換し、不正値は欠損にする。"""
    return pd.to_numeric(df[column], errors="coerce").astype(float)


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """ゼロ除算と無限大を欠損値にする。"""
    result = numerator.div(denominator.where(denominator.ne(0)))
    return result.replace([np.inf, -np.inf], np.nan)


def _safe_log1p(values: pd.Series) -> pd.Series:
    """0以上の値だけを log1p 変換する。"""
    return np.log1p(values.where(values.ge(0)))


def _has_it_department(value: object) -> int:
    if pd.isna(value):
        return 0
    normalized = unicodedata.normalize("NFKC", str(value))
    return int(bool(IT_DEPARTMENT_PATTERN.search(normalized)))


def calculate_execution_features(df: pd.DataFrame) -> pd.DataFrame:
    """入力行と同じ index を持つ、実行能力の9特徴量を返す。

    元データではソフトウェア関連の固定資産変動がキャッシュアウトとして
    負値で記録されているため、符号を反転して投資額（正値）に直してから
    売上で割る。比率の分母が0の場合や不正な数値は ``NaN`` とする。
    """
    _validate_columns(df)

    sales = _numeric(df, "売上")
    total_assets = _numeric(df, "総資産")
    employees = _numeric(df, "従業員数")

    features = pd.DataFrame(index=df.index)
    features["営業利益率"] = _safe_divide(_numeric(df, "営業利益"), sales)
    features["営業CFマージン"] = _safe_divide(_numeric(df, "営業CF"), sales)
    features["自己資本比率"] = _safe_divide(_numeric(df, "自己資本"), total_assets)
    features["借入金比率"] = _safe_divide(
        _numeric(df, "短期借入金") + _numeric(df, "長期借入金"),
        total_assets,
    )

    software_investment = -_numeric(df, SOFTWARE_CHANGE_COLUMN)
    features["ソフトウェア投資比率"] = _safe_divide(software_investment, sales)
    features["IT部門有無"] = df["組織図"].map(_has_it_department).astype("int8")

    security = _numeric(df, "アンケート５")
    features["セキュリティ整備度"] = security.where(security.between(1, 5))
    features["log_売上"] = _safe_log1p(sales)
    features["log_従業員数"] = _safe_log1p(employees)

    return features.loc[:, FEATURE_COLUMNS]


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """識別・目的変数カラムと計算済み特徴量をまとめる。"""
    keep_columns = [c for c in ("企業ID", "購入フラグ") if c in df.columns]
    return pd.concat(
        [df[keep_columns].reset_index(drop=True),
         calculate_execution_features(df).reset_index(drop=True)],
        axis=1,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="実行能力に関する9特徴量を計算します。")
    parser.add_argument("input_csv", type=Path, help="入力CSV (train.csv / test.csv)")
    parser.add_argument("output_csv", type=Path, help="出力先CSV")
    args = parser.parse_args()

    source = pd.read_csv(args.input_csv)
    output = build_output(source)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"{len(output)}行 x {len(output.columns)}列を {args.output_csv} に保存しました。")


if __name__ == "__main__":
    main()
