"""「今後のDX展望」から作る特徴量の実装場所。

以下は後続作業で実装する予定。

- 経営主導フラグ
- 全社展開フラグ
- 計画具体性スコア
  - 実施時期
  - 対象部門
  - 導入技術
  - 導入・刷新などの行動
  - 推進担当
  - 予算
  - KPI
"""

from __future__ import annotations

import pandas as pd


DX_OUTLOOK_FEATURE_COLUMNS = [
    "経営主導フラグ",
    "全社展開フラグ",
    "計画具体性スコア",
]


def calculate_dx_outlook_features(df: pd.DataFrame) -> pd.DataFrame:
    """自由記述由来の特徴量を返す（未実装）。"""
    raise NotImplementedError(
        "「今後のDX展望」由来の特徴量は未実装です。dx_outlook.py に実装してください。"
    )
