import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from calc_features.all_features_v4 import (
    ALL_FEATURES_V4_TREE_COLUMNS,
    AllFeaturesV4Transformer,
    all_features_v4,
)


class AllFeaturesV4Test(unittest.TestCase):
    def test_combines_v3_with_generic_tree_features(self) -> None:
        source = pd.DataFrame({"組織図": ["社長\n└DX推進部", "社長\n└営業部"]}, index=[10, 20])
        base = pd.DataFrame(
            {f"base_{index:02d}": [float(index), float(index + 1)] for index in range(23)},
            index=source.index,
        )
        tree = pd.DataFrame(
            {
                column: [float(index), float(index + 1)]
                for index, column in enumerate(ALL_FEATURES_V4_TREE_COLUMNS)
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
                "calc_features.all_features_v4.AllFeaturesV3Transformer",
                return_value=base_transformer,
            ),
            patch(
                "calc_features.all_features_v4.TreeOfCorpTransformer",
                return_value=tree_transformer,
            ),
        ):
            transformer = AllFeaturesV4Transformer()
            actual = transformer.fit_transform(source)

        self.assertEqual(actual.index.tolist(), source.index.tolist())
        self.assertEqual(actual.shape, (2, 38))
        self.assertEqual(actual.columns[:23].tolist(), base.columns.tolist())
        self.assertEqual(actual.columns[23:].tolist(), list(ALL_FEATURES_V4_TREE_COLUMNS))
        self.assertNotIn("DX推進室完全一致フラグ", actual.columns)
        base_transformer.fit_transform.assert_called_once_with(source)
        tree_transformer.fit_transform.assert_called_once_with(source)
        tree_transformer.transform.assert_called_once_with(source)

    def test_transform_preserves_fitted_schema_for_unknown_notation(self) -> None:
        train = pd.DataFrame({"組織図": ["社長\n└DX推進部"]}, index=[3])
        test = pd.DataFrame({"組織図": ["自由記述の未知形式"]}, index=[9])
        base_train = pd.DataFrame({"base": [1.0]}, index=train.index)
        base_transformer = MagicMock()
        base_transformer.fit_transform.return_value = base_train
        base_transformer.transform.side_effect = lambda frame: pd.DataFrame(
            {"base": [2.0] * len(frame)}, index=frame.index
        )

        with patch(
            "calc_features.all_features_v4.AllFeaturesV3Transformer",
            return_value=base_transformer,
        ):
            transformer = AllFeaturesV4Transformer().fit(train)
            actual = transformer.transform(test)

        self.assertEqual(actual.index.tolist(), [9])
        self.assertEqual(actual.columns.tolist(), transformer.get_feature_names_out())
        self.assertEqual(actual.shape[1], 1 + len(ALL_FEATURES_V4_TREE_COLUMNS))

    def test_convenience_function_does_not_modify_input(self) -> None:
        source = pd.DataFrame({"組織図": ["社長\n└DX推進部"]}, index=[10])
        original = source.copy(deep=True)
        expected = pd.DataFrame({"feature": [1.0]}, index=source.index)
        transformer = MagicMock()
        transformer.fit_transform.return_value = expected

        with patch(
            "calc_features.all_features_v4.AllFeaturesV4Transformer",
            return_value=transformer,
        ):
            actual = all_features_v4(source)

        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(actual, expected)
        transformer.fit_transform.assert_called_once_with(source)


if __name__ == "__main__":
    unittest.main()
