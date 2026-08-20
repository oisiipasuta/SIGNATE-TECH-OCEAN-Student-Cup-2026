import unittest

import pandas as pd

from calc_features.dx_outlook_dictionary import (
    DX_OUTLOOK_DICTIONARY_FEATURE_COLUMNS,
    DX_OUTLOOK_DICTIONARY_V1,
    calculate_dx_outlook_dictionary_features,
    find_dx_outlook_dictionary_matches,
    normalize_dx_outlook_dictionary_text,
    profile_dx_outlook_dictionary,
)


class DXOutlookDictionaryV1Test(unittest.TestCase):
    def test_v1_has_exactly_seven_expected_categories(self):
        self.assertEqual(
            tuple(DX_OUTLOOK_DICTIONARY_V1),
            (
                "EDU",
                "EXTERNAL",
                "EXPAND",
                "CAUTIOUS",
                "MAINTAIN",
                "SUPPRESS",
                "NEED",
            ),
        )

    def test_normalization_handles_width_case_and_whitespace(self):
        self.assertEqual(
            normalize_dx_outlook_dictionary_text(" ＤＸ 教育 と e ラーニング "),
            "dx教育とeラーニング",
        )

    def test_matcher_returns_cross_axis_matches(self):
        matches = find_dx_outlook_dictionary_matches(
            "外部セミナーによるDX教育を段階的に拡充する。"
        )

        self.assertIn("DX教育", matches["EDU"])
        self.assertIn("外部セミナー", matches["EXTERNAL"])
        self.assertIn("拡充", matches["EXPAND"])
        self.assertIn("段階的", matches["CAUTIOUS"])

    def test_profiler_uses_only_texts_and_reports_document_frequency(self):
        profile = profile_dx_outlook_dictionary(
            ["DX教育を拡充する", "DX 教育は現状維持", None]
        )
        dx_education = profile.loc[profile["expression"] == "DX教育"].iloc[0]

        self.assertEqual(int(dx_education["document_frequency"]), 2)
        self.assertAlmostEqual(float(dx_education["document_rate"]), 2 / 3)
        self.assertEqual(int(dx_education["total_occurrences"]), 2)

    def test_categories_do_not_contain_duplicate_expressions(self):
        for category, expressions in DX_OUTLOOK_DICTIONARY_V1.items():
            with self.subTest(category=category):
                self.assertEqual(len(expressions), len(set(expressions)))

    def test_v1_mapping_is_runtime_immutable(self):
        with self.assertRaises(TypeError):
            DX_OUTLOOK_DICTIONARY_V1["EDU"] = ()

    def test_dictionary_features_have_two_columns_per_category(self):
        data = pd.DataFrame(
            {
                "今後のDX展望": [
                    "外部セミナーでDX教育を段階的に拡充する。",
                    None,
                ]
            },
            index=[10, 20],
        )

        features = calculate_dx_outlook_dictionary_features(data)

        self.assertEqual(features.shape, (2, 14))
        self.assertListEqual(features.columns.tolist(), DX_OUTLOOK_DICTIONARY_FEATURE_COLUMNS)
        self.assertListEqual(features.index.tolist(), [10, 20])
        self.assertEqual(features.loc[10, "dx_dict_v1_edu_matched_expressions"], 1)
        self.assertEqual(features.loc[10, "dx_dict_v1_external_total_occurrences"], 1)
        self.assertEqual(features.loc[10, "dx_dict_v1_expand_total_occurrences"], 2)
        self.assertEqual(features.loc[10, "dx_dict_v1_cautious_total_occurrences"], 1)
        self.assertEqual(int(features.loc[20].sum()), 0)

    def test_dictionary_features_keep_nested_expression_matches(self):
        features = calculate_dx_outlook_dictionary_features(["基礎研修を行う"])

        self.assertEqual(features.loc[0, "dx_dict_v1_edu_matched_expressions"], 2)
        self.assertEqual(features.loc[0, "dx_dict_v1_edu_total_occurrences"], 2)

    def test_dictionary_features_require_text_column(self):
        with self.assertRaisesRegex(KeyError, "今後のDX展望"):
            calculate_dx_outlook_dictionary_features(pd.DataFrame({"別の列": ["DX教育"]}))


if __name__ == "__main__":
    unittest.main()
