"""「今後のDX展望」を MeCab + TF-IDF + TruncatedSVD で数値化する。"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.validation import check_is_fitted

try:
    import MeCab
except ImportError:  # pragma: no cover - エラー文は _create_tagger で分かりやすくする。
    MeCab = None  # type: ignore[assignment]

try:
    import unidic_lite
except ImportError:  # pragma: no cover - 同上。
    unidic_lite = None  # type: ignore[assignment]


DX_OUTLOOK_COLUMN = "今後のDX展望"
DX_OUTLOOK_SVD_COMPONENTS = (5, 10, 30)
DX_OUTLOOK_NGRAM_CHANNELS = (
    ("unigram", (1, 1)),
    ("bigram", (2, 2)),
)

# 「形容詞・形状詞」は UniDic では別々の品詞として返されるため、両方を含める。
TARGET_PARTS_OF_SPEECH = frozenset(
    {
        "名詞",
        "動詞",
        "形容詞",
        "形状詞",
        "副詞",
    }
)

# 品詞アブレーションなどで指定ミスを防ぐため、UniDicの大分類を公開する。
# 既定では内容語だけを使うが、助詞などを含むn-gramも実験できるようにする。
SUPPORTED_PARTS_OF_SPEECH = frozenset(
    {
        "名詞",
        "代名詞",
        "動詞",
        "形容詞",
        "形状詞",
        "連体詞",
        "副詞",
        "接続詞",
        "感動詞",
        "助動詞",
        "助詞",
        "接頭辞",
        "接尾辞",
        "記号",
        "補助記号",
        "空白",
    }
)


def get_dx_outlook_feature_columns(n_components: int) -> list[str]:
    """SVD後の特徴量名を返す。"""
    if n_components <= 0:
        raise ValueError("n_components は1以上にしてください。")
    return [f"dx_outlook_svd_{n_components}_{i + 1:02d}" for i in range(n_components)]


# 従来の定数名は、既定の30次元で互換性を保つ。
DX_OUTLOOK_FEATURE_COLUMNS = get_dx_outlook_feature_columns(30)


class DXOutlookTfidfSVD(BaseEstimator, TransformerMixin):
    """DX展望文を、指定品詞のTF-IDFからSVD特徴量へ変換する。

    CVで利用するときは各foldの学習データだけで ``fit`` し、検証データには
    ``transform`` のみを行う。欠損値は空文字列として扱う。
    """

    def __init__(
        self,
        n_components: int = 30,
        *,
        text_column: str = DX_OUTLOOK_COLUMN,
        target_parts_of_speech: frozenset[str] | tuple[str, ...] = TARGET_PARTS_OF_SPEECH,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int | float = 1,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.text_column = text_column
        self.target_parts_of_speech = target_parts_of_speech
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state

    def _create_tagger(self) -> Any:
        if MeCab is None or unidic_lite is None:
            raise ImportError(
                "MeCabによる形態素解析には mecab-python3 と unidic-lite が必要です。"
                " `pip install -r requirements.txt` を実行してください。"
            )

        # OSに既存のmecabrcがなくても、同梱したUniDicを確実に利用する。
        args = f'-r "{os.devnull}" -d "{unidic_lite.DICDIR}"'
        tagger = MeCab.Tagger(args)
        tagger.parse("")
        return tagger

    def __getstate__(self) -> dict[str, Any]:
        """MeCabのC拡張オブジェクトを除外し、学習済み変換器を保存可能にする。"""
        state = self.__dict__.copy()
        state["_tagger"] = None
        return state

    def _tokenize(self, text: str) -> list[str]:
        if not hasattr(self, "_tagger") or self._tagger is None:
            self._tagger = self._create_tagger()

        tokens: list[str] = []
        node = self._tagger.parseToNode(text)
        while node is not None:
            if node.surface:
                part_of_speech = node.feature.split(",", maxsplit=1)[0]
                if part_of_speech in self.target_parts_of_speech:
                    tokens.append(node.surface)
            node = node.next
        return tokens

    def _extract_texts(
        self,
        X: pd.DataFrame | pd.Series | Iterable[str],
    ) -> tuple[list[str], pd.Index]:
        if isinstance(X, pd.DataFrame):
            if self.text_column not in X.columns:
                raise KeyError(f"入力DataFrameに `{self.text_column}` 列がありません。")
            values = X[self.text_column]
            index = X.index
        elif isinstance(X, pd.Series):
            values = X
            index = X.index
        else:
            values = pd.Series(list(X), dtype="string")
            index = values.index

        texts = values.fillna("").astype(str).tolist()
        return texts, index

    def fit(
        self,
        X: pd.DataFrame | pd.Series | Iterable[str],
        y: Any = None,
    ) -> "DXOutlookTfidfSVD":
        del y
        if not isinstance(self.n_components, int) or self.n_components <= 0:
            raise ValueError("n_components は1以上の整数にしてください。")
        selected_parts = frozenset(self.target_parts_of_speech)
        if not selected_parts:
            raise ValueError("target_parts_of_speech は1品詞以上指定してください。")
        unsupported = selected_parts - SUPPORTED_PARTS_OF_SPEECH
        if unsupported:
            raise ValueError(f"未対応の品詞が指定されました: {sorted(unsupported)}")
        if (
            not isinstance(self.ngram_range, tuple)
            or len(self.ngram_range) != 2
            or not all(isinstance(value, int) for value in self.ngram_range)
            or self.ngram_range[0] <= 0
            or self.ngram_range[0] > self.ngram_range[1]
        ):
            raise ValueError(
                "ngram_range は (最小n, 最大n) の形式で、"
                "1 <= 最小n <= 最大n となる整数を指定してください。"
            )

        texts, _ = self._extract_texts(X)
        self._tagger = None
        self.vectorizer_ = TfidfVectorizer(
            tokenizer=self._tokenize,
            token_pattern=None,
            lowercase=False,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
        )
        tfidf = self.vectorizer_.fit_transform(texts)

        if self.n_components > tfidf.shape[1]:
            raise ValueError(
                f"n_components={self.n_components} はTF-IDF語彙数"
                f" ({tfidf.shape[1]}) 以下にしてください。"
            )

        self.svd_ = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.svd_.fit(tfidf)
        self.feature_names_out_ = get_dx_outlook_feature_columns(self.n_components)
        return self

    def transform(
        self,
        X: pd.DataFrame | pd.Series | Iterable[str],
    ) -> pd.DataFrame:
        check_is_fitted(self, attributes=["vectorizer_", "svd_", "feature_names_out_"])
        texts, index = self._extract_texts(X)
        reduced = self.svd_.transform(self.vectorizer_.transform(texts))
        return pd.DataFrame(reduced, columns=self.feature_names_out_, index=index)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


class DXOutlookMultiNgramTfidfSVD(BaseEstimator, TransformerMixin):
    """複数の単語n-gramを別々にTF-IDF・SVD化して結合する。

    各チャネルは同じ学習データだけで独立にfitされる。``min_df`` は各
    チャネルの文書頻度に適用されるため、CVではfoldごとに新しい変換器を
    作成し、学習foldだけを ``fit`` する。
    """

    def __init__(
        self,
        n_components: int = 30,
        *,
        text_column: str = DX_OUTLOOK_COLUMN,
        target_parts_of_speech: frozenset[str] | tuple[str, ...] = TARGET_PARTS_OF_SPEECH,
        channels: tuple[tuple[str, tuple[int, int]], ...] = DX_OUTLOOK_NGRAM_CHANNELS,
        min_df: int | float = 1,
        max_features: int | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.text_column = text_column
        self.target_parts_of_speech = target_parts_of_speech
        self.channels = channels
        self.min_df = min_df
        self.max_features = max_features
        self.random_state = random_state

    def _validate_channels(self) -> None:
        if not isinstance(self.channels, tuple) or not self.channels:
            raise ValueError("channels は1チャネル以上指定してください。")
        if any(not isinstance(channel, tuple) or len(channel) != 2 for channel in self.channels):
            raise ValueError("各チャネルは (名前, ngram_range) で指定してください。")
        names = [channel[0] for channel in self.channels]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("チャネル名は空でない文字列にしてください。")
        if len(names) != len(set(names)):
            raise ValueError("チャネル名は重複しないようにしてください。")

    def fit(
        self,
        X: pd.DataFrame | pd.Series | Iterable[str],
        y: Any = None,
    ) -> "DXOutlookMultiNgramTfidfSVD":
        del y
        self._validate_channels()
        self.transformers_: dict[str, DXOutlookTfidfSVD] = {}
        self.vocabulary_counts_: dict[str, int] = {}
        feature_names: list[str] = []

        for channel_name, ngram_range in self.channels:
            transformer = DXOutlookTfidfSVD(
                n_components=self.n_components,
                text_column=self.text_column,
                target_parts_of_speech=self.target_parts_of_speech,
                ngram_range=ngram_range,
                min_df=self.min_df,
                max_features=self.max_features,
                random_state=self.random_state,
            ).fit(X)
            self.transformers_[channel_name] = transformer
            self.vocabulary_counts_[channel_name] = len(
                transformer.vectorizer_.get_feature_names_out()
            )
            feature_names.extend(
                f"{channel_name}_svd_{index + 1:02d}"
                for index in range(self.n_components)
            )

        self.feature_names_out_ = feature_names
        return self

    def transform(
        self,
        X: pd.DataFrame | pd.Series | Iterable[str],
    ) -> pd.DataFrame:
        check_is_fitted(
            self,
            attributes=["transformers_", "vocabulary_counts_", "feature_names_out_"],
        )
        frames: list[pd.DataFrame] = []
        for channel_name, _ in self.channels:
            transformed = self.transformers_[channel_name].transform(X).copy()
            transformed.columns = [
                f"{channel_name}_svd_{index + 1:02d}"
                for index in range(self.n_components)
            ]
            frames.append(transformed)
        return pd.concat(frames, axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> list[str]:
        del input_features
        check_is_fitted(self, attributes=["feature_names_out_"])
        return self.feature_names_out_.copy()


def calculate_dx_outlook_features(
    df: pd.DataFrame,
    n_components: int = 30,
    *,
    target_parts_of_speech: frozenset[str] | tuple[str, ...] = TARGET_PARTS_OF_SPEECH,
    ngram_range: tuple[int, int] = (1, 1),
    **transformer_kwargs: Any,
) -> pd.DataFrame:
    """1つの学習DataFrameにfitし、DX展望のSVD特徴量を返す。

    検証・テストデータを変換する場合は、データリーク防止のためこの補助関数を
    個別に呼ばず、``DXOutlookTfidfSVD`` の ``fit`` / ``transform`` を使う。
    """
    transformer = DXOutlookTfidfSVD(
        n_components=n_components,
        target_parts_of_speech=target_parts_of_speech,
        ngram_range=ngram_range,
        **transformer_kwargs,
    )
    return transformer.fit_transform(df)


__all__ = [
    "DX_OUTLOOK_COLUMN",
    "DX_OUTLOOK_FEATURE_COLUMNS",
    "DX_OUTLOOK_NGRAM_CHANNELS",
    "DX_OUTLOOK_SVD_COMPONENTS",
    "TARGET_PARTS_OF_SPEECH",
    "SUPPORTED_PARTS_OF_SPEECH",
    "DXOutlookMultiNgramTfidfSVD",
    "DXOutlookTfidfSVD",
    "calculate_dx_outlook_features",
    "get_dx_outlook_feature_columns",
]
