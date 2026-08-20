import pickle
import unittest

import pandas as pd

from calc_features.dx_outlook import (
    DXOutlookMultiNgramTfidfSVD,
    DXOutlookTfidfSVD,
    calculate_dx_outlook_features,
)


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

    def test_word_ngram_range_is_applied_after_mecab_tokenization(self):
        train = pd.DataFrame(
            {
                "今後のDX展望": [
                    "システムを導入する",
                    "システムを刷新する",
                    "データを分析する",
                ]
            }
        )
        transformer = DXOutlookTfidfSVD(
            n_components=2,
            target_parts_of_speech=("名詞", "動詞"),
            ngram_range=(1, 2),
        ).fit(train)

        vocabulary = set(transformer.vectorizer_.get_feature_names_out())

        self.assertIn("システム", vocabulary)
        self.assertIn("システム 導入", vocabulary)
        self.assertIn("導入 する", vocabulary)

    def test_ngram_range_can_use_function_words_when_requested(self):
        train = pd.DataFrame(
            {"今後のDX展望": ["システムを導入する", "データを活用する"]}
        )
        transformer = DXOutlookTfidfSVD(
            n_components=1,
            target_parts_of_speech=("名詞", "助詞"),
            ngram_range=(2, 2),
        ).fit(train)

        vocabulary = set(transformer.vectorizer_.get_feature_names_out())

        self.assertIn("システム を", vocabulary)

    def test_invalid_ngram_range_is_rejected(self):
        transformer = DXOutlookTfidfSVD(n_components=2, ngram_range=(2, 1))

        with self.assertRaisesRegex(ValueError, "ngram_range"):
            transformer.fit(pd.DataFrame({"今後のDX展望": ["DXを進める"]}))

    def test_calculate_function_exposes_pos_and_ngram_settings(self):
        train = pd.DataFrame(
            {
                "今後のDX展望": [
                    "システムを導入する",
                    "データを活用する",
                    "業務を改善する",
                ]
            }
        )

        result = calculate_dx_outlook_features(
            train,
            n_components=2,
            target_parts_of_speech=("名詞", "動詞"),
            ngram_range=(1, 2),
        )

        self.assertEqual(result.shape, (3, 2))

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


class DXOutlookMultiNgramTfidfSVDTest(unittest.TestCase):
    def test_channels_are_independently_reduced_and_combined(self):
        train = pd.DataFrame(
            {
                "今後のDX展望": [
                    "システム導入を進める",
                    "システム導入を継続する",
                    "データ分析を進める",
                    "データ分析を継続する",
                ]
            },
            index=[10, 20, 30, 40],
        )
        transformer = DXOutlookMultiNgramTfidfSVD(
            n_components=2,
            target_parts_of_speech=("名詞",),
            min_df=2,
        ).fit(train)

        result = transformer.transform(train)

        self.assertEqual(result.shape, (4, 4))
        self.assertListEqual(result.index.tolist(), train.index.tolist())
        self.assertListEqual(
            result.columns.tolist(),
            [
                "unigram_svd_01",
                "unigram_svd_02",
                "bigram_svd_01",
                "bigram_svd_02",
            ],
        )
        self.assertEqual(set(transformer.vocabulary_counts_), {"unigram", "bigram"})

    def test_min_df_is_applied_within_each_channel(self):
        train = pd.DataFrame(
            {
                "今後のDX展望": [
                    "システムを導入する",
                    "システムを導入する",
                    "データを分析する",
                ]
            }
        )
        transformer = DXOutlookMultiNgramTfidfSVD(
            n_components=1,
            target_parts_of_speech=("名詞",),
            channels=(("unigram", (1, 1)),),
            min_df=2,
        ).fit(train)

        vocabulary = set(
            transformer.transformers_["unigram"].vectorizer_.get_feature_names_out()
        )

        self.assertIn("システム", vocabulary)
        self.assertNotIn("データ", vocabulary)


if __name__ == "__main__":
    unittest.main()
