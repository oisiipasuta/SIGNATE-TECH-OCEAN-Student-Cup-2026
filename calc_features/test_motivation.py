import unittest

import pandas as pd

from calc_features.motivation import calculate_motivation_features


class MotivationFeaturesTest(unittest.TestCase):
    def test_questionnaire_features(self):
        df = pd.DataFrame({
            "アンケート１": [5, 0],
            "アンケート９": [4, 6],
            "アンケート１１": [3, "不明"],
        })
        actual = calculate_motivation_features(df)

        self.assertEqual(actual.loc[0, "DX戦略明確度"], 5)
        self.assertEqual(actual.loc[0, "情報収集度"], 3)
        self.assertEqual(actual.loc[0, "セミナー参加度"], 4)
        self.assertTrue(actual.loc[1].isna().all())

    def test_missing_required_column(self):
        with self.assertRaises(KeyError):
            calculate_motivation_features(pd.DataFrame())


if __name__ == "__main__":
    unittest.main()
