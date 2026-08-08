import pickle
import unittest

import pandas as pd

from calc_features.dx_outlook import DXOutlookTfidfSVD


class DXOutlookTfidfSVDTest(unittest.TestCase):
    def test_tokenizer_keeps_only_requested_parts_of_speech(self):
        transformer = DXOutlookTfidfSVD(n_components=5)

        tokens = transformer._tokenize(
            "全社で新しいシステムを積極的に導入する予定です。"
        )

        self.assertIn("全社", tokens)  # 名詞
        self.assertIn("新しい", tokens)  # 形容詞
        self.assertIn("する", tokens)  # 動詞
        self.assertNotIn("で", tokens)  # 助詞
        self.assertNotIn("です", tokens)  # 助動詞

    def test_tokenizer_accepts_custom_parts_of_speech(self):
        transformer = DXOutlookTfidfSVD(
            n_components=2,
            target_parts_of_speech=("名詞",),
        )

        tokens = transformer._tokenize("新しいシステムを積極的に導入する。")

        self.assertIn("システム", tokens)
        self.assertNotIn("新しい", tokens)
        self.assertNotIn("する", tokens)

    def test_empty_parts_of_speech_is_rejected(self):
        transformer = DXOutlookTfidfSVD(
            n_components=2,
            target_parts_of_speech=(),
        )

        with self.assertRaisesRegex(ValueError, "1品詞以上"):
            transformer.fit(pd.DataFrame({"今後のDX展望": ["DXを進める"]}))

    def test_fit_transform_preserves_index_and_dimension(self):
        train = pd.DataFrame(
            {
                "今後のDX展望": [
                    "全社で基幹システムを刷新する",
                    "営業部門へAIを導入する",
                    "業務データを積極的に活用する",
                    "来年度からクラウド移行を進める",
                    "経営層がDX人材を育成する",
                    "工場の作業を自動化する",
                ]
            },
            index=[10, 20, 30, 40, 50, 60],
        )
        valid = pd.DataFrame(
            {"今後のDX展望": ["新しい技術を導入する", None]},
            index=[100, 200],
        )
        transformer = DXOutlookTfidfSVD(n_components=5)

        train_features = transformer.fit_transform(train)
        valid_features = transformer.transform(valid)

        self.assertEqual(train_features.shape, (6, 5))
        self.assertEqual(valid_features.shape, (2, 5))
        self.assertListEqual(train_features.index.tolist(), train.index.tolist())
        self.assertListEqual(valid_features.index.tolist(), valid.index.tolist())
        self.assertEqual(train_features.isna().sum().sum(), 0)

    def test_fitted_transformer_can_be_pickled(self):
        train = pd.DataFrame(
            {
                "今後のDX展望": [
                    "システムを導入する",
                    "データを分析する",
                    "業務を改善する",
                ]
            }
        )
        transformer = DXOutlookTfidfSVD(n_components=2).fit(train)

        restored = pickle.loads(pickle.dumps(transformer))
        result = restored.transform(train.iloc[:1])

        self.assertEqual(result.shape, (1, 2))


if __name__ == "__main__":
    unittest.main()
