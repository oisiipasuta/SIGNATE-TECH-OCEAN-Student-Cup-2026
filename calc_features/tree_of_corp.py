"""組織図の構造とデジタル推進体制を数値特徴量へ変換する。"""

from __future__ import annotations

from collections import Counter
import re
from typing import Any
import unicodedata

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


TREE_OF_CORP_COLUMN = "組織図"

TREE_OF_CORP_FEATURE_COLUMNS = (
    "組織ノード数",
    "組織最大階層",
    "第一階層組織数",
    "平均分岐数",
    "組織機能多様性",
    "階層解析可能フラグ",
    "デジタル組織数",
    "DX変革組織有無",
    "IT運用組織有無",
    "デジタル組織最小階層",
    "デジタル組織経営直下フラグ",
    "研究開発組織数",
    "生産・製造組織数",
    "海外組織数",
    "調達・購買組織数",
)

TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS = ("DX推進室完全一致フラグ",)

TREE_OF_CORP_NORMALIZED_FEATURE_COLUMNS = (
    "デジタル組織比率",
    "第一階層組織比率",
    "組織機能密度",
    "研究開発組織比率",
    "生産・製造組織比率",
    "海外組織比率",
    "調達・購買組織比率",
)

_UNIT_SUFFIXES = (
    "本部",
    "事業部",
    "部門",
    "センター",
    "部",
    "室",
    "課",
    "支社",
    "支店",
    "営業所",
    "事業所",
    "工場",
    "店舗",
    "委員会",
    "グループ",
    "チーム",
)
_LEADERSHIP_NAMES = (
    "経営層",
    "代表取締役社長",
    "取締役会",
    "執行役員",
    "社長",
    "会長",
)
_UNIT_END_PATTERN = re.compile(
    rf"(?:{'|'.join(map(re.escape, _UNIT_SUFFIXES))})"
    r"(?:\s|$|[\]】）)])"
)
_LEADING_DRAWING_PATTERN = re.compile(
    r"^[\s\u2500-\u257f■□●○◆◇・*【】\[\]]+"
)
_HORIZONTAL_DIAGRAM_PATTERN = re.compile(r"[┌└]─{5,}|─{8,}[┐┘┬┴┼]")
_BRANCH_CHARS = "├└┣┗"

_DIGITAL_PATTERN = re.compile(
    r"情報システム|情報技術|社内システム|"
    r"(?<![A-Z])IT(?![A-Z])|ICT|DX|デジタル|"
    r"(?<![A-Z])AI(?![A-Z])|データ",
    flags=re.IGNORECASE,
)
_DX_TRANSFORMATION_PATTERN = re.compile(
    r"(?:DX|デジタル|AI|データ)[^\n]{0,20}"
    r"(?:推進|戦略|改革|変革|企画|イノベーション)",
    flags=re.IGNORECASE,
)
_IT_OPERATIONS_PATTERN = re.compile(
    r"情報システム|情報技術|社内システム|"
    r"IT(?:インフラ|運用|管理|統括|サポート)|"
    r"ICT(?:管理|統括|基盤)|システム管理",
    flags=re.IGNORECASE,
)
_DX_PROMOTION_OFFICE_PATTERN = re.compile(r"DX\s*推進\s*室", re.IGNORECASE)

_FUNCTION_PATTERNS = {
    "経営企画": re.compile(r"経営企画|経営戦略|事業企画|戦略"),
    "人事総務": re.compile(r"人事|人材|採用|総務"),
    "財務経理": re.compile(r"財務|経理"),
    "法務監査": re.compile(r"法務|監査|内部統制|コンプライアンス"),
    "営業販売": re.compile(r"営業|販売"),
    "マーケティング": re.compile(r"マーケティング|広報|宣伝"),
    "研究開発": re.compile(r"研究|開発|R&D", re.IGNORECASE),
    "生産製造": re.compile(r"生産|製造"),
    "品質": re.compile(r"品質"),
    "調達購買": re.compile(r"調達|購買|仕入"),
    "物流": re.compile(r"物流|ロジスティクス|配送"),
    "海外": re.compile(r"海外|国際|グローバル|米国|中国|アジア|欧州"),
    "新規事業": re.compile(r"新規事業|事業開発|イノベーション"),
    "デジタル": _DIGITAL_PATTERN,
}

_FEATURE_COUNT_PATTERNS = {
    "研究開発組織数": _FUNCTION_PATTERNS["研究開発"],
    "生産・製造組織数": _FUNCTION_PATTERNS["生産製造"],
    "海外組織数": _FUNCTION_PATTERNS["海外"],
    "調達・購買組織数": _FUNCTION_PATTERNS["調達購買"],
}

_DRAWING_TRANSLATION = str.maketrans(
    {
        "┣": "├",
        "┗": "└",
        "━": "─",
        "┃": "│",
    }
)


def normalize_tree_of_corp_text(value: object) -> str:
    """組織図のUnicode、改行、タブ、罫線表記を正規化する。"""
    if pd.isna(value):
        return ""
    normalized = unicodedata.normalize("NFKC", str(value))
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\t", "    ").translate(_DRAWING_TRANSLATION)
    return "\n".join(line.rstrip() for line in normalized.split("\n")).strip()


def _clean_node_label(line: str) -> str:
    label = _LEADING_DRAWING_PATTERN.sub("", line).strip()
    label = re.sub(r"\s+", " ", label)
    return label


def _is_node_label(label: str) -> bool:
    if not label or (label.startswith("(") and label.endswith(")")):
        return False
    if label in {"組織図", "株式会社組織図"}:
        return False
    return bool(_UNIT_END_PATTERN.search(label)) or any(
        name in label for name in _LEADERSHIP_NAMES
    )


def _line_depth(line: str) -> int | None:
    positions = [line.rfind(character) for character in _BRANCH_CHARS]
    branch_position = max(positions)
    if branch_position < 0:
        return None
    return 1 + int(round(branch_position / 3))


def _extract_nodes(text: str) -> list[tuple[str, int | None]]:
    nodes: list[tuple[str, int | None]] = []
    for line in text.splitlines():
        label = _clean_node_label(line)
        if _is_node_label(label):
            nodes.append((label, _line_depth(line)))
    return nodes


def _is_vertical_tree(text: str, nodes: list[tuple[str, int | None]]) -> bool:
    if not text or _HORIZONTAL_DIAGRAM_PATTERN.search(text):
        return False
    return len(nodes) >= 2 and any(depth is not None for _, depth in nodes)


def _hierarchy_features(
    nodes: list[tuple[str, int | None]],
    analyzable: bool,
) -> tuple[float, float, float]:
    if not analyzable:
        return np.nan, np.nan, np.nan

    depths = [0 if depth is None else depth for _, depth in nodes]
    maximum_depth = float(max(depths, default=0))
    first_level_count = float(sum(depth == 1 for depth in depths))

    child_counts: Counter[tuple[int, int]] = Counter()
    latest_node_at_depth: dict[int, int] = {}
    for node_index, depth in enumerate(depths):
        latest_node_at_depth[depth] = node_index
        for deeper_depth in [value for value in latest_node_at_depth if value > depth]:
            del latest_node_at_depth[deeper_depth]
        if depth > 0:
            parent_depths = [value for value in latest_node_at_depth if value < depth]
            if parent_depths:
                parent_depth = max(parent_depths)
                child_counts[(parent_depth, latest_node_at_depth[parent_depth])] += 1

    average_branching = (
        float(np.mean(list(child_counts.values()))) if child_counts else 0.0
    )
    return maximum_depth, first_level_count, average_branching


def _count_matching_nodes(
    nodes: list[tuple[str, int | None]], pattern: re.Pattern[str]
) -> int:
    return sum(bool(pattern.search(label)) for label, _ in nodes)


def _extract_row_features(value: object, include_artifact: bool) -> dict[str, float]:
    text = normalize_tree_of_corp_text(value)
    nodes = _extract_nodes(text)
    analyzable = _is_vertical_tree(text, nodes)
    maximum_depth, first_level_count, average_branching = _hierarchy_features(
        nodes, analyzable
    )

    digital_nodes = [
        (label, depth) for label, depth in nodes if _DIGITAL_PATTERN.search(label)
    ]
    digital_depths = [depth for _, depth in digital_nodes if depth is not None]
    digital_minimum_depth = (
        float(min(digital_depths)) if analyzable and digital_depths else np.nan
    )
    digital_top_level = (
        float(any(depth <= 1 for depth in digital_depths))
        if analyzable and digital_depths
        else np.nan
    )

    features: dict[str, float] = {
        "組織ノード数": float(len(nodes)),
        "組織最大階層": maximum_depth,
        "第一階層組織数": first_level_count,
        "平均分岐数": average_branching,
        "組織機能多様性": float(
            sum(
                any(pattern.search(label) for label, _ in nodes)
                for pattern in _FUNCTION_PATTERNS.values()
            )
        ),
        "階層解析可能フラグ": float(analyzable),
        "デジタル組織数": float(len(digital_nodes)),
        "DX変革組織有無": float(
            any(_DX_TRANSFORMATION_PATTERN.search(label) for label, _ in nodes)
        ),
        "IT運用組織有無": float(
            any(_IT_OPERATIONS_PATTERN.search(label) for label, _ in nodes)
        ),
        "デジタル組織最小階層": digital_minimum_depth,
        "デジタル組織経営直下フラグ": digital_top_level,
    }
    for feature_name, pattern in _FEATURE_COUNT_PATTERNS.items():
        features[feature_name] = float(_count_matching_nodes(nodes, pattern))

    if include_artifact:
        features["DX推進室完全一致フラグ"] = float(
            bool(_DX_PROMOTION_OFFICE_PATTERN.search(text))
        )
    return features


def calculate_tree_of_corp_features(
    df: pd.DataFrame,
    *,
    include_artifact: bool = False,
) -> pd.DataFrame:
    """入力と同じindexを持つ組織図特徴量を返す。

    ``include_artifact=True`` のときだけ競技データで頻出する完全一致語
    ``DX推進室`` のフラグを追加する。階層を信頼して解析できない横型・自由記述
    の図では、深さ・分岐・配置特徴を欠損にして誤推定値を使用しない。
    """
    if TREE_OF_CORP_COLUMN not in df.columns:
        raise KeyError(f"組織図特徴量の計算に必要なカラムがありません: {TREE_OF_CORP_COLUMN}")
    records = [
        _extract_row_features(value, include_artifact)
        for value in df[TREE_OF_CORP_COLUMN]
    ]
    columns = list(TREE_OF_CORP_FEATURE_COLUMNS)
    if include_artifact:
        columns.extend(TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS)
    return pd.DataFrame(records, index=df.index).loc[:, columns]


def calculate_tree_of_corp_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """組織ノード数で規模をそろえた7つの組織図特徴量を返す。

    ノード数が0の行では比率を定義できないため欠損にする。第一階層組織数が
    解析不能な横型・自由記述図では、第一階層組織比率も欠損のまま維持する。
    """
    base = calculate_tree_of_corp_features(df)
    denominator = base["組織ノード数"].where(base["組織ノード数"].gt(0))
    normalized = pd.DataFrame(index=df.index)
    normalized["デジタル組織比率"] = base["デジタル組織数"].div(denominator)
    normalized["第一階層組織比率"] = base["第一階層組織数"].div(denominator)
    normalized["組織機能密度"] = base["組織機能多様性"].div(denominator)
    normalized["研究開発組織比率"] = base["研究開発組織数"].div(denominator)
    normalized["生産・製造組織比率"] = base["生産・製造組織数"].div(denominator)
    normalized["海外組織比率"] = base["海外組織数"].div(denominator)
    normalized["調達・購買組織比率"] = base["調達・購買組織数"].div(denominator)
    return normalized.loc[:, TREE_OF_CORP_NORMALIZED_FEATURE_COLUMNS]


class TreeOfCorpTransformer(BaseEstimator, TransformerMixin):
    """組織図特徴量を返すscikit-learn互換の決定的Transformer。"""

    def __init__(self, *, include_artifact: bool = False) -> None:
        self.include_artifact = include_artifact

    def fit(self, X: pd.DataFrame, y: Any = None) -> "TreeOfCorpTransformer":
        del y
        generated = calculate_tree_of_corp_features(
            X, include_artifact=self.include_artifact
        )
        self.feature_names_out_ = generated.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_out_"])
        generated = calculate_tree_of_corp_features(
            X, include_artifact=self.include_artifact
        )
        return generated.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


class NormalizedTreeOfCorpTransformer(BaseEstimator, TransformerMixin):
    """組織ノード数で正規化した組織図特徴を返す決定的Transformer。"""

    def fit(
        self, X: pd.DataFrame, y: Any = None
    ) -> "NormalizedTreeOfCorpTransformer":
        del y
        generated = calculate_tree_of_corp_normalized_features(X)
        self.feature_names_out_ = generated.columns.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, attributes=["feature_names_out_"])
        generated = calculate_tree_of_corp_normalized_features(X)
        return generated.loc[:, self.feature_names_out_]

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


__all__ = [
    "TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS",
    "TREE_OF_CORP_COLUMN",
    "TREE_OF_CORP_FEATURE_COLUMNS",
    "TREE_OF_CORP_NORMALIZED_FEATURE_COLUMNS",
    "NormalizedTreeOfCorpTransformer",
    "TreeOfCorpTransformer",
    "calculate_tree_of_corp_features",
    "calculate_tree_of_corp_normalized_features",
    "normalize_tree_of_corp_text",
]
