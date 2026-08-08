import unittest

import numpy as np
import pandas as pd

from calc_features.calculate import calculate_execution_features


class CalculateExecutionFeaturesTest(unittest.TestCase):
    def test_calculates_features_and_handles_zero_division(self):
        frame = pd.DataFrame(
            {
                "売上": [100.0, 0.0],
                "営業利益": [10.0, 5.0],
                "営業CF": [20.0, 3.0],
                "自己資本": [40.0, 1.0],
                "総資産": [200.0, 0.0],
                "短期借入金": [10.0, 1.0],
                "長期借入金": [30.0, 1.0],
                "無形固定資産変動(ソフトウェア関連)": [-5.0, -1.0],
                "従業員数": [9, 0],
                "組織図": ["管理本部 > 情報システム部", "製品システム開発部"],
                "アンケート５": [4, 9],
            }
        )

        result = calculate_execution_features(frame)

        self.assertAlmostEqual(result.loc[0, "営業利益率"], 0.1)
        self.assertAlmostEqual(result.loc[0, "営業CFマージン"], 0.2)
        self.assertAlmostEqual(result.loc[0, "自己資本比率"], 0.2)
        self.assertAlmostEqual(result.loc[0, "借入金比率"], 0.2)
        self.assertAlmostEqual(result.loc[0, "ソフトウェア投資比率"], 0.05)
        self.assertEqual(result.loc[0, "IT部門有無"], 1)
        self.assertEqual(result.loc[1, "IT部門有無"], 0)
        self.assertTrue(np.isnan(result.loc[1, "営業利益率"]))
        self.assertTrue(np.isnan(result.loc[1, "自己資本比率"]))
        self.assertTrue(np.isnan(result.loc[1, "セキュリティ整備度"]))
        self.assertAlmostEqual(result.loc[0, "log_従業員数"], np.log1p(9))


if __name__ == "__main__":
    unittest.main()
