import unittest

import pandas as pd

from calc_features.adoption_barriers import calculate_adoption_barrier_features


class AdoptionBarrierFeaturesTest(unittest.TestCase):
    def test_direct_features_financial_flag_and_placeholders(self):
        frame = pd.DataFrame(
            {
                "アンケート４": [5, 0, 3, 2],
                "営業利益": [-1, 10, None, None],
                "営業CF": [20, -5, 0, None],
                "アンケート７": [4, 6, "不明", 1],
                "アンケート８": [5, 3, 1, None],
                "今後のDX展望": ["人材不足", "予算が課題", "", None],
            }
        )

        actual = calculate_adoption_barrier_features(frame)

        self.assertEqual(actual.loc[0, "DX抵抗感"], 5)
        self.assertTrue(pd.isna(actual.loc[1, "DX抵抗感"]))
        self.assertEqual(actual["赤字・CF不足フラグ"].iloc[:3].tolist(), [1, 1, 0])
        self.assertTrue(pd.isna(actual.loc[3, "赤字・CF不足フラグ"]))
        self.assertTrue(actual[["人材不足フラグ", "予算制約フラグ"]].isna().all().all())
        self.assertEqual(actual.loc[0, "現行ツール満足度"], 4)
        self.assertTrue(pd.isna(actual.loc[1, "現行ツール満足度"]))
        self.assertEqual(actual.loc[1, "DX成果実感度"], 3)

    def test_missing_required_column(self):
        with self.assertRaises(KeyError):
            calculate_adoption_barrier_features(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
