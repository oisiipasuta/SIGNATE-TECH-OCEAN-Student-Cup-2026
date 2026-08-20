import unittest

import pandas as pd

from calc_features.tree_of_corp import (
    TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS,
    TREE_OF_CORP_FEATURE_COLUMNS,
    TREE_OF_CORP_NORMALIZED_FEATURE_COLUMNS,
    NormalizedTreeOfCorpTransformer,
    TreeOfCorpTransformer,
    calculate_tree_of_corp_features,
    calculate_tree_of_corp_normalized_features,
    normalize_tree_of_corp_text,
)


VERTICAL_TREE = """経営層
├─ DX推進室
├─ 管理本部
│  ├─ 情報システム部
│  └─ 人事部
└─ 生産本部
   ├─ 製造部
   └─ 調達部"""


class TreeOfCorpFeaturesTest(unittest.TestCase):
    def test_vertical_tree_structure_and_digital_features(self) -> None:
        source = pd.DataFrame({"組織図": [VERTICAL_TREE]}, index=[17])

        actual = calculate_tree_of_corp_features(source)

        self.assertEqual(actual.index.tolist(), [17])
        self.assertEqual(actual.columns.tolist(), list(TREE_OF_CORP_FEATURE_COLUMNS))
        self.assertEqual(actual.loc[17, "組織ノード数"], 8.0)
        self.assertEqual(actual.loc[17, "組織最大階層"], 2.0)
        self.assertEqual(actual.loc[17, "第一階層組織数"], 3.0)
        self.assertAlmostEqual(actual.loc[17, "平均分岐数"], 7 / 3)
        self.assertEqual(actual.loc[17, "階層解析可能フラグ"], 1.0)
        self.assertEqual(actual.loc[17, "デジタル組織数"], 2.0)
        self.assertEqual(actual.loc[17, "DX変革組織有無"], 1.0)
        self.assertEqual(actual.loc[17, "IT運用組織有無"], 1.0)
        self.assertEqual(actual.loc[17, "デジタル組織最小階層"], 1.0)
        self.assertEqual(actual.loc[17, "デジタル組織経営直下フラグ"], 1.0)
        self.assertEqual(actual.loc[17, "生産・製造組織数"], 2.0)
        self.assertEqual(actual.loc[17, "調達・購買組織数"], 1.0)

    def test_nfkc_tabs_and_heavy_drawing_are_normalized(self) -> None:
        value = "経営層\r\n\t┣━ ＤＸ推進室\r\n\t┗━ 管理本部"

        normalized = normalize_tree_of_corp_text(value)
        actual = calculate_tree_of_corp_features(pd.DataFrame({"組織図": [value]}))

        self.assertNotIn("\r", normalized)
        self.assertNotIn("\t", normalized)
        self.assertIn("├─ DX推進室", normalized)
        self.assertEqual(actual.loc[0, "DX変革組織有無"], 1.0)

    def test_horizontal_diagram_keeps_content_but_masks_hierarchy(self) -> None:
        value = """┌──────────┐
│ 経営層       │
└────┬─────┘
   【DX推進室】 【営業本部】"""

        actual = calculate_tree_of_corp_features(pd.DataFrame({"組織図": [value]}))

        self.assertEqual(actual.loc[0, "階層解析可能フラグ"], 0.0)
        self.assertTrue(pd.isna(actual.loc[0, "組織最大階層"]))
        self.assertTrue(pd.isna(actual.loc[0, "第一階層組織数"]))
        self.assertTrue(pd.isna(actual.loc[0, "平均分岐数"]))
        self.assertEqual(actual.loc[0, "DX変革組織有無"], 1.0)
        self.assertTrue(pd.isna(actual.loc[0, "デジタル組織最小階層"]))

    def test_description_text_is_not_counted_as_digital_organization(self) -> None:
        value = """経営層
└─ 営業本部
   └─ 顧客支援部
      (DX推進とデジタル改革を担当する)"""

        actual = calculate_tree_of_corp_features(pd.DataFrame({"組織図": [value]}))

        self.assertEqual(actual.loc[0, "デジタル組織数"], 0.0)
        self.assertEqual(actual.loc[0, "DX変革組織有無"], 0.0)

    def test_missing_value_preserves_index_and_returns_safe_values(self) -> None:
        source = pd.DataFrame({"組織図": [pd.NA]}, index=[99])

        actual = calculate_tree_of_corp_features(source)

        self.assertEqual(actual.index.tolist(), [99])
        self.assertEqual(actual.loc[99, "組織ノード数"], 0.0)
        self.assertEqual(actual.loc[99, "階層解析可能フラグ"], 0.0)
        self.assertTrue(pd.isna(actual.loc[99, "組織最大階層"]))
        self.assertEqual(actual.loc[99, "DX変革組織有無"], 0.0)

    def test_artifact_flag_is_opt_in(self) -> None:
        source = pd.DataFrame({"組織図": [VERTICAL_TREE]})

        generic = calculate_tree_of_corp_features(source)
        artifact = calculate_tree_of_corp_features(source, include_artifact=True)

        self.assertNotIn(TREE_OF_CORP_ARTIFACT_FEATURE_COLUMNS[0], generic.columns)
        self.assertEqual(artifact.loc[0, "DX推進室完全一致フラグ"], 1.0)

    def test_normalized_features_divide_counts_by_node_count(self) -> None:
        source = pd.DataFrame({"組織図": [VERTICAL_TREE]}, index=[17])

        actual = calculate_tree_of_corp_normalized_features(source)

        self.assertEqual(
            actual.columns.tolist(), list(TREE_OF_CORP_NORMALIZED_FEATURE_COLUMNS)
        )
        self.assertEqual(actual.loc[17, "デジタル組織比率"], 2 / 8)
        self.assertEqual(actual.loc[17, "第一階層組織比率"], 3 / 8)
        self.assertEqual(actual.loc[17, "生産・製造組織比率"], 2 / 8)
        self.assertEqual(actual.loc[17, "調達・購買組織比率"], 1 / 8)

    def test_normalized_features_use_missing_for_zero_nodes(self) -> None:
        source = pd.DataFrame({"組織図": [pd.NA]})

        actual = calculate_tree_of_corp_normalized_features(source)

        self.assertTrue(actual.isna().all().all())

    def test_normalized_transformer_has_stable_schema(self) -> None:
        source = pd.DataFrame(
            {"組織図": [VERTICAL_TREE, "経営層\n└─ 営業本部"]},
            index=[10, 20],
        )
        transformer = NormalizedTreeOfCorpTransformer()

        fitted = transformer.fit_transform(source.iloc[[0]])
        transformed = transformer.transform(source.iloc[[1]])

        self.assertEqual(fitted.columns.tolist(), transformed.columns.tolist())
        self.assertEqual(fitted.shape[1], len(TREE_OF_CORP_NORMALIZED_FEATURE_COLUMNS))

    def test_transformer_schema_is_stable_and_input_is_not_modified(self) -> None:
        source = pd.DataFrame(
            {"組織図": [VERTICAL_TREE, "経営層\n└─ 営業本部"]},
            index=[10, 20],
        )
        original = source.copy(deep=True)
        transformer = TreeOfCorpTransformer(include_artifact=True)

        fitted = transformer.fit_transform(source.iloc[[0]])
        transformed = transformer.transform(source.iloc[[1]])

        self.assertEqual(fitted.columns.tolist(), transformed.columns.tolist())
        self.assertEqual(fitted.shape[1], len(TREE_OF_CORP_FEATURE_COLUMNS) + 1)
        pd.testing.assert_frame_equal(source, original)

    def test_missing_required_column_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            calculate_tree_of_corp_features(pd.DataFrame({"別列": ["値"]}))


if __name__ == "__main__":
    unittest.main()
