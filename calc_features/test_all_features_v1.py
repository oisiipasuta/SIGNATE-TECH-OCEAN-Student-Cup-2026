import unittest
from unittest.mock import patch

import pandas as pd

from calc_features.all_features_v1 import (
    EXCLUDED_FEATURE_COLUMNS,
    AllFeaturesV1Transformer,
    all_features_v1,
)


class AllFeaturesV1Test(unittest.TestCase):
    def test_excludes_columns_and_groups_industries_below_three_percent(self) -> None:
        source = pd.DataFrame(index=range(100))
        industries = pd.Series(
            ["金融"] * 95 + ["IT"] * 3 + ["機械"] * 2,
            index=source.index,
            dtype="string",
        )
        excluded = {
            column: pd.Series(range(100), index=source.index)
            for column in EXCLUDED_FEATURE_COLUMNS
        }

        frames = [
            pd.DataFrame({"営業利益率": range(100)}, index=source.index),
            pd.DataFrame({"DX戦略明確度": range(100)}, index=source.index),
            pd.DataFrame(excluded, index=source.index),
            pd.DataFrame(
                {"業界": industries, "拠点総数": range(100)}, index=source.index
            ),
            pd.DataFrame({"DX全体不満度": range(100)}, index=source.index),
        ]
        patch_targets = [
            "calculate_execution_features",
            "calculate_motivation_features",
            "calculate_adoption_barrier_features",
            "calculate_necessity_features",
            "calculate_purchase_timing_features",
        ]

        patchers = [
            patch(f"calc_features.all_features_v1.{name}", return_value=frame)
            for name, frame in zip(patch_targets, frames)
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        actual = all_features_v1(source)

        self.assertEqual(actual.index.tolist(), source.index.tolist())
        self.assertTrue(set(EXCLUDED_FEATURE_COLUMNS).isdisjoint(actual.columns))
        self.assertEqual(actual["業界"].value_counts().to_dict(), {"金融": 95, "IT": 3, "その他": 2})

    def test_transform_uses_training_frequency_rule(self) -> None:
        train = pd.read_csv("data/train.csv").copy()
        transformer = AllFeaturesV1Transformer().fit(train)
        retained = transformer.retained_industries_
        validation = train.iloc[:3].copy()
        validation["業界"] = [next(iter(retained)), "未知業界", "未知業界"]

        actual = transformer.transform(validation)

        self.assertEqual(actual.loc[validation.index[0], "業界"], validation.iloc[0]["業界"])
        self.assertEqual(actual.loc[validation.index[1], "業界"], "その他")
        self.assertEqual(actual.loc[validation.index[2], "業界"], "その他")

    def test_does_not_modify_input(self) -> None:
        train = pd.read_csv("data/train.csv", nrows=3)
        original = train.copy(deep=True)

        actual = all_features_v1(train)

        pd.testing.assert_frame_equal(train, original)
        self.assertEqual(actual.shape, (3, 19))


if __name__ == "__main__":
    unittest.main()
