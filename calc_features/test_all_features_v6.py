import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from calc_features.all_features_v6 import (
    ALL_FEATURES_V6_TREE_COLUMNS,
    AllFeaturesV6Transformer,
    all_features_v6,
)


class AllFeaturesV6Test(unittest.TestCase):
    def test_combines_v3_with_dx_transformation_flag_only(self) -> None:
        source = pd.DataFrame({"組織図": ["社長\n└DX推進部", "社長\n└営業部"]}, index=[10, 20])
        base = pd.DataFrame(
            {f"base_{index:02d}": [float(index), float(index + 1)] for index in range(23)},
            index=source.index,
        )
        tree = pd.DataFrame(
            {
                "DX変革組織有無": [1.0, 0.0],
                "平均分岐数": [2.0, 3.0],
            },
            index=source.index,
        )
        base_transformer = MagicMock()
        base_transformer.fit_transform.return_value = base
        base_transformer.transform.return_value = base
        tree_transformer = MagicMock()
        tree_transformer.fit_transform.return_value = tree
        tree_transformer.transform.return_value = tree

        with (
            patch(
                "calc_features.all_features_v6.AllFeaturesV3Transformer",
                return_value=base_transformer,
            ),
            patch(
                "calc_features.all_features_v6.TreeOfCorpTransformer",
                return_value=tree_transformer,
            ),
        ):
            actual = AllFeaturesV6Transformer().fit_transform(source)

        self.assertEqual(actual.shape, (2, 24))
        self.assertEqual(actual.columns[:23].tolist(), base.columns.tolist())
        self.assertEqual(actual.columns[23:].tolist(), list(ALL_FEATURES_V6_TREE_COLUMNS))
        self.assertNotIn("平均分岐数", actual.columns)
        tree_transformer.fit_transform.assert_called_once_with(source)
        tree_transformer.transform.assert_called_once_with(source)

    def test_convenience_function_does_not_modify_input(self) -> None:
        source = pd.DataFrame({"組織図": ["社長\n└DX推進部"]}, index=[10])
        original = source.copy(deep=True)
        expected = pd.DataFrame({"feature": [1.0]}, index=source.index)
        transformer = MagicMock()
        transformer.fit_transform.return_value = expected

        with patch(
            "calc_features.all_features_v6.AllFeaturesV6Transformer",
            return_value=transformer,
        ):
            actual = all_features_v6(source)

        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
