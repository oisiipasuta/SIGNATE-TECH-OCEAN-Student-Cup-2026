import importlib.util
import unittest
from pathlib import Path

import pandas as pd


# calc_features/__init__.py にある他特徴量の開発途中コードに依存せず、単体で検証する。
MODULE_PATH = Path(__file__).with_name("necessity.py")
SPEC = importlib.util.spec_from_file_location("necessity", MODULE_PATH)
necessity = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(necessity)


class NecessityFeaturesTest(unittest.TestCase):
    def test_direct_features_and_placeholders(self):
        frame = pd.DataFrame(
            {
                "業界": ["製造", "小売"],
                "従業員数": [100, "250"],
                "事業所数": [2, None],
                "工場数": [3, None],
                "店舗数": [0, None],
                "企業概要": ["生産と物流", "店舗を運営"],
                "組織図": ["本部 > 部", "本社 > 店舗"],
            }
        )

        actual = necessity.calculate_necessity_features(frame)

        self.assertEqual(actual.loc[0, "業界"], "製造")
        self.assertEqual(actual.loc[0, "従業員規模"], 100)
        self.assertEqual(actual.loc[1, "従業員規模"], 250)
        self.assertEqual(actual.loc[0, "拠点総数"], 5)
        self.assertTrue(pd.isna(actual.loc[1, "拠点総数"]))
        self.assertTrue(actual[["組織部門数", "組織階層数", "業務種類数"]].isna().all().all())

    def test_missing_required_column(self):
        with self.assertRaises(KeyError):
            necessity.calculate_necessity_features(pd.DataFrame({"業界": []}))


if __name__ == "__main__":
    unittest.main()
