"""「今後のDX展望」向けのラベル非参照キーワード辞書。

辞書v1は ``data/train.csv`` の「今後のDX展望」本文だけを確認して作成した。
購入フラグは語の選定・追加・削除・頻度確認のいずれにも使用していない。
CV評価を始めた後はv1を変更せず、変更が必要なら別バージョンとして定義する。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import re
from types import MappingProxyType
import unicodedata

import pandas as pd


DX_OUTLOOK_DICTIONARY_VERSION = "v1"
DX_OUTLOOK_DICTIONARY_TEXT_COLUMN = "今後のDX展望"

DX_OUTLOOK_DICTIONARY_DESCRIPTIONS: Mapping[str, str] = MappingProxyType({
    "EDU": "教育需要そのもの",
    "EXTERNAL": "外部教育など商材購入に近い手段",
    "EXPAND": "今後の投資・施策を増やす拡大型",
    "CAUTIOUS": "効果を確認しながら進める慎重型",
    "MAINTAIN": "現行水準や既存施策を保つ維持型",
    "SUPPRESS": "追加投資や施策を明示的に抑える抑制型",
    "NEED": "教育・DXの不足や課題を示す課題型",
})

# 並び順も辞書仕様の一部とし、実験開始後はv1を変更しない。
DX_OUTLOOK_DICTIONARY_V1: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "EDU": (
        "DX教育",
        "教育投資",
        "人材育成",
        "研修",
        "リスキリング",
        "デジタルリテラシー",
        "DXリテラシー",
        "デジタルスキル",
        "DXスキル",
        "スキル向上",
        "スキルアップ",
        "リテラシー向上",
        "リテラシー底上げ",
        "ハンズオン",
        "実践型",
        "実践的",
        "実務直結",
        "実務に即した",
        "実務と連動",
        "OJT",
        "eラーニング",
        "教育プログラム",
        "研修プログラム",
        "基礎研修",
        "学習機会",
        "学習基盤",
    ),
    "EXTERNAL": (
        "外部セミナー",
        "オンライン教材",
        "オンライン講座",
        "外部専門家",
        "専門講師",
        "外部講師",
        "外部機関",
        "外部研修",
        "eラーニング商材",
    ),
    "EXPAND": (
        "拡充",
        "拡大",
        "強化",
        "刷新",
        "加速",
        "引き上げ",
        "本格導入",
        "積極的",
        "増強",
        "大幅に拡大",
        "全面的",
        "一層強化",
        "投資を拡大",
        "追加投資",
        "プログラム拡充",
        "段階的に拡大",
        "段階的に拡充",
    ),
    "CAUTIOUS": (
        "段階的",
        "慎重",
        "見極め",
        "費用対効果",
        "コスト対効果",
        "投資対効果",
        "小規模",
        "試験的",
        "検証しながら",
        "効果検証",
        "選択的",
        "スモールスタート",
        "PoC",
        "パイロット",
        "優先順位",
        "無理のない",
        "慎重に判断",
    ),
    "MAINTAIN": (
        "維持",
        "現行水準",
        "現状維持",
        "既存プログラム",
        "延長線上",
        "既存施策",
        "現行の施策",
        "従来通り",
        "保守運用",
        "安定運用",
        "継続する方針",
        "水準を維持",
        "姿勢を維持",
    ),
    "SUPPRESS": (
        "必要最低限",
        "最小限",
        "限定的",
        "とどめる",
        "留める",
        "刷新計画はない",
        "予定はない",
        "予定していない",
        "予定しておりません",
        "予定はございません",
        "計画はありません",
        "計画はございません",
        "見送る",
        "当面見送る",
        "抑制",
        "控えめ",
        "踏み切らず",
        "踏み切らない",
        "大規模投資は行わない",
        "投資を抑える",
        "投資を抑制",
    ),
    "NEED": (
        "人材不足",
        "スキル不足",
        "人手不足",
        "抵抗",
        "抵抗感",
        "課題",
        "現場課題",
        "改善余地",
        "伸びしろ",
        "浸透不足",
        "温度差",
        "習熟度の差",
        "十分ではない",
        "未整備",
        "不足部分",
    ),
})


def normalize_dx_outlook_dictionary_text(text: object) -> str:
    """表記揺れを抑えるためNFKC化、小文字化、空白除去を行う。"""
    if text is None or pd.isna(text):
        return ""
    normalized = unicodedata.normalize("NFKC", str(text)).lower()
    return re.sub(r"\s+", "", normalized)


def find_dx_outlook_dictionary_matches(
    text: object,
    dictionary: Mapping[str, tuple[str, ...]] = DX_OUTLOOK_DICTIONARY_V1,
) -> dict[str, tuple[str, ...]]:
    """カテゴリごとに本文へ部分一致した辞書表現を返す。"""
    normalized_text = normalize_dx_outlook_dictionary_text(text)
    return {
        category: tuple(
            expression
            for expression in expressions
            if normalize_dx_outlook_dictionary_text(expression) in normalized_text
        )
        for category, expressions in dictionary.items()
    }


def get_dx_outlook_dictionary_feature_columns(
    dictionary: Mapping[str, tuple[str, ...]] = DX_OUTLOOK_DICTIONARY_V1,
) -> list[str]:
    """辞書カテゴリごとの一致種類数・総出現回数の列名を返す。"""
    columns: list[str] = []
    for category in dictionary:
        prefix = f"dx_dict_{DX_OUTLOOK_DICTIONARY_VERSION}_{category.lower()}"
        columns.extend(
            [
                f"{prefix}_matched_expressions",
                f"{prefix}_total_occurrences",
            ]
        )
    return columns


DX_OUTLOOK_DICTIONARY_FEATURE_COLUMNS = get_dx_outlook_dictionary_feature_columns()


def calculate_dx_outlook_dictionary_features(
    data: pd.DataFrame | pd.Series | Iterable[object],
    *,
    text_column: str = DX_OUTLOOK_DICTIONARY_TEXT_COLUMN,
    dictionary: Mapping[str, tuple[str, ...]] = DX_OUTLOOK_DICTIONARY_V1,
) -> pd.DataFrame:
    """固定辞書だけを使い、7カテゴリ×2種類の数値特徴量を返す。

    ``matched_expressions`` は一致した辞書表現の種類数、
    ``total_occurrences`` は各表現の本文内出現回数の合計である。同一カテゴリ内で
    長い表現が短い表現を含む場合も両方を数える。この関数は目的変数を受け取らず、
    データから語彙や閾値を学習しない。
    """
    if isinstance(data, pd.DataFrame):
        if text_column not in data.columns:
            raise KeyError(f"入力DataFrameに `{text_column}` 列がありません。")
        values = data[text_column]
        index = data.index
    elif isinstance(data, pd.Series):
        values = data
        index = data.index
    else:
        values = pd.Series(list(data), dtype="object")
        index = values.index

    normalized_expressions = {
        category: tuple(normalize_dx_outlook_dictionary_text(value) for value in expressions)
        for category, expressions in dictionary.items()
    }
    rows: list[dict[str, int]] = []
    for value in values:
        normalized_text = normalize_dx_outlook_dictionary_text(value)
        row: dict[str, int] = {}
        for category, expressions in normalized_expressions.items():
            prefix = f"dx_dict_{DX_OUTLOOK_DICTIONARY_VERSION}_{category.lower()}"
            occurrences = [normalized_text.count(expression) for expression in expressions]
            row[f"{prefix}_matched_expressions"] = sum(count > 0 for count in occurrences)
            row[f"{prefix}_total_occurrences"] = sum(occurrences)
        rows.append(row)
    return pd.DataFrame(
        rows,
        index=index,
        columns=get_dx_outlook_dictionary_feature_columns(dictionary),
        dtype="int64",
    )


def profile_dx_outlook_dictionary(
    texts: Iterable[object],
    dictionary: Mapping[str, tuple[str, ...]] = DX_OUTLOOK_DICTIONARY_V1,
) -> pd.DataFrame:
    """目的変数を受け取らず、表現ごとの文書頻度と総出現回数を集計する。"""
    normalized_texts = [normalize_dx_outlook_dictionary_text(text) for text in texts]
    document_count = len(normalized_texts)
    rows: list[dict[str, object]] = []
    for category, expressions in dictionary.items():
        for expression in expressions:
            normalized_expression = normalize_dx_outlook_dictionary_text(expression)
            frequencies = [text.count(normalized_expression) for text in normalized_texts]
            document_frequency = sum(frequency > 0 for frequency in frequencies)
            rows.append(
                {
                    "version": DX_OUTLOOK_DICTIONARY_VERSION,
                    "category": category,
                    "description": DX_OUTLOOK_DICTIONARY_DESCRIPTIONS[category],
                    "expression": expression,
                    "document_frequency": document_frequency,
                    "document_rate": (
                        document_frequency / document_count if document_count else 0.0
                    ),
                    "total_occurrences": sum(frequencies),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "DX_OUTLOOK_DICTIONARY_DESCRIPTIONS",
    "DX_OUTLOOK_DICTIONARY_FEATURE_COLUMNS",
    "DX_OUTLOOK_DICTIONARY_TEXT_COLUMN",
    "DX_OUTLOOK_DICTIONARY_V1",
    "DX_OUTLOOK_DICTIONARY_VERSION",
    "calculate_dx_outlook_dictionary_features",
    "find_dx_outlook_dictionary_matches",
    "get_dx_outlook_dictionary_feature_columns",
    "normalize_dx_outlook_dictionary_text",
    "profile_dx_outlook_dictionary",
]
