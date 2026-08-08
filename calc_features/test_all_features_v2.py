import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

from calc_features.all_features_v2 import (
    DX_OUTLOOK_SECOND_FEATURE,
    AllFeaturesV2Transformer,
    all_features_v2,
)


class AllFeaturesV2Test(unittest.TestCase):
    def test_combines_v1_with_only_second_svd_feature(self) -> None:
        source = pd.DataFrame(
            {"今後のDX展望": ["DXを進める", "AIを導入する"]},
            index=[10, 20],
        )
        base = pd.DataFrame(
            {"営業利益率": [0.1, 0.2], "業界": ["IT", "その他"]},
            index=source.index,
        )
        svd = pd.DataFrame(
            {
                "dx_outlook_svd_30_01": [1.0, 2.0],
                DX_OUTLOOK_SECOND_FEATURE: [3.0, 4.0],
                "dx_outlook_svd_30_03": [5.0, 6.0],
            },
            index=source.index,
        )
        dx_transformer = MagicMock()
        dx_transformer.fit.return_value = dx_transformer
        dx_transformer.transform.return_value = svd

        with (
            patch(
                "calc_features.all_features_v2.all_features_v1",
                return_value=base,
            ),
            patch(
                "calc_features.all_features_v2.DXOutlookTfidfSVD",
                return_value=dx_transformer,
            ),
        ):
            actual = AllFeaturesV2Transformer().fit_transform(source)

        self.assertEqual(actual.index.tolist(), source.index.tolist())
        self.assertEqual(
            actual.columns.tolist(),
            ["営業利益率", "業界", DX_OUTLOOK_SECOND_FEATURE],
        )
        self.assertEqual(actual[DX_OUTLOOK_SECOND_FEATURE].tolist(), [3.0, 4.0])
        dx_transformer.fit.assert_called_once_with(source)
        dx_transformer.transform.assert_called_once_with(source)

    def test_convenience_function_does_not_modify_input(self) -> None:
        source = pd.DataFrame(
            {"今後のDX展望": ["DXを進める", "AIを導入する"]},
            index=[10, 20],
        )
        original = source.copy(deep=True)
        expected = pd.DataFrame({"feature": [1.0, 2.0]}, index=source.index)
        transformer = MagicMock()
        transformer.fit_transform.return_value = expected

        with patch(
            "calc_features.all_features_v2.AllFeaturesV2Transformer",
            return_value=transformer,
        ):
            actual = all_features_v2(source)

        pd.testing.assert_frame_equal(source, original)
        pd.testing.assert_frame_equal(actual, expected)
        transformer.fit_transform.assert_called_once_with(source)


if __name__ == "__main__":
    unittest.main()
