# `calc_features` 特徴量仕様書

## 概要

`calc_features` は、企業の財務情報、企業属性、アンケート回答、組織図などから、DX 商材の購買予測に利用する特徴量を生成するパッケージです。

実装状況は次の3段階に分かれます。

- **実装済み**: 入力データから値を計算する
- **仮実装**: 出力列は作られるが、現在は全行で欠損値を返す
- **未実装**: 関数を呼ぶと `NotImplementedError` が発生する

## 共通仕様

- 計算結果の行数と index は、原則として入力 `DataFrame` と同じです。
- 数値として解釈できない値は `NaN` / `pd.NA` に変換されます。
- アンケートの5段階評価は、数値化後に **1～5** の値だけを有効とし、それ以外を欠損値にします。
- 必須列が不足している場合は `KeyError` または `ValueError` が発生します。
- `calculate_*_features()` は計算した特徴量だけを返します。
- `add_*_features()` は元の `DataFrame` を変更せず、特徴量を追加したコピーを返します。

## 1. 実行能力 (`calculate.py`)

企業の収益性、財務余力、IT 投資、組織体制、企業規模から9特徴量を計算します。すべて実装済みです。

### 必須入力列

`売上`、`営業利益`、`営業CF`、`自己資本`、`総資産`、`短期借入金`、`長期借入金`、`無形固定資産変動(ソフトウェア関連)`、`従業員数`、`組織図`、`アンケート５`

### 特徴量の定義

| 特徴量 | 計算方法・定義 | 補足 |
|---|---|---|
| `営業利益率` | `営業利益 / 売上` | 売上が0の場合は欠損 |
| `営業CFマージン` | `営業CF / 売上` | 売上が0の場合は欠損 |
| `自己資本比率` | `自己資本 / 総資産` | 総資産が0の場合は欠損 |
| `借入金比率` | `(短期借入金 + 長期借入金) / 総資産` | 総資産が0の場合は欠損 |
| `ソフトウェア投資比率` | `-無形固定資産変動(ソフトウェア関連) / 売上` | 元データのキャッシュアウトが負値のため、符号を反転して投資額として扱う |
| `IT部門有無` | `組織図` に IT/DX 担当組織名があれば1、なければ0 | NFKC 正規化後に正規表現で判定。欠損も0 |

→今後、IT部門の有無は 0 or 1 ではなく、IT部門の数などの特徴量も試す。

| `セキュリティ整備度` | `アンケート５` の値 | 1～5以外は欠損 |
| `log_売上` | `log(1 + 売上)` | 売上が負の場合は欠損。自然対数を使用 |

→LightGBMでは対数変換不要

| `log_従業員数` | `log(1 + 従業員数)` | 従業員数が負の場合は欠損。自然対数を使用 |

→LightGBMでは対数変換不要

`IT部門有無` は、`情報システム`、`情報技術`、`社内システム`、`IT`、`ICT`、`DX`、`デジタル` などの担当名と、`本部`、`部門`、`センター`、`部`、`室`、`課`、`チーム`、`グループ` の組み合わせを検出します。例えば `情報システム部` は1、単なる事業部名の `製品システム開発部` は0です。

## 2. 推進意欲 (`motivation.py`)

DX への取り組み意欲を、アンケート回答から3特徴量として作成します。すべて実装済みです。

| 特徴量 | 入力列 | 計算方法 |
|---|---|---|
| `DX戦略明確度` | `アンケート１` | 回答を数値化し、1～5のみ採用 |
| `情報収集度` | `アンケート１１` | 回答を数値化し、1～5のみ採用 |
| `セミナー参加度` | `アンケート９` | 回答を数値化し、1～5のみ採用 |

## 3. 導入障壁・充足済み度 (`adoption_barriers.py`)

DX 導入の障壁と、既存施策への満足・成果実感を表す6特徴量を作成します。

### 必須入力列

`アンケート４`、`営業利益`、`営業CF`、`アンケート７`、`アンケート８`

### 特徴量の定義

| 特徴量 | 計算方法・定義 | 状態 |
|---|---|---|
| `DX抵抗感` | `アンケート４` を数値化し、1～5のみ採用 | 実装済み |
| `赤字・CF不足フラグ` | `営業利益 < 0` または `営業CF < 0` なら1、それ以外は0 | 実装済み |

→営業利益やCFを 0 or 1 に変換しないパターンも試す。

| `人材不足フラグ` | 将来は `今後のDX展望` から人材不足・採用難・スキル不足を抽出予定 | 仮実装（全行欠損） |
| `予算制約フラグ` | 将来は `今後のDX展望` から予算不足・費用負担・投資余力を抽出予定 | 仮実装（全行欠損） |
| `現行ツール満足度` | `アンケート７` を数値化し、1～5のみ採用 | 実装済み |
| `DX成果実感度` | `アンケート８` を数値化し、1～5のみ採用 | 実装済み |

`赤字・CF不足フラグ` は、営業利益または営業CFの一方だけでも負なら1です。観測できた値がすべて0以上なら0、両方とも欠損または不正値なら欠損になります。

## 4. 必要性 (`necessity.py`)

企業規模や拠点数など、DX 商材を必要とする度合いに関係する6特徴量を作成します。

### 必須入力列

`業界`、`従業員数`、`事業所数`、`工場数`、`店舗数`

### 特徴量の定義

| 特徴量 | 計算方法・定義 | 状態 |
|---|---|---|
| `業界` | 入力の `業界` をそのまま使用 | 実装済み |

→極端にデータ数が少ない業界が存在する可能性があるので、その場合はその他などにまとめてもいいかも

| `従業員規模` | `従業員数` を数値化。対数変換はしない | 実装済み |
| `拠点総数` | `事業所数 + 工場数 + 店舗数` | 実装済み |

→業界ごとに店舗や事業や工場数の割合が異なる可能性はある。業界との交互作用をみて場合によっては合計ではなく、各３つの特徴量をそれぞれ用いてもよい

| `組織部門数` | 将来は `組織図` から重複を除いた部門数を抽出予定 | 仮実装（全行欠損） |
| `組織階層数` | 将来は `組織図` の区切りやインデントから最大階層数を抽出予定 | 仮実装（全行欠損） |
| `業務種類数` | 将来は `企業概要` から該当する業務カテゴリ数を算出予定 | 仮実装（全行欠損） |

`拠点総数` は、一部の内訳が欠損でも残りの値を合計します。3項目すべてが欠損の場合だけ欠損になります。

`業務種類数` で予定されているカテゴリは、生産管理、在庫管理、物流、店舗運営、顧客管理、保守・点検、バックオフィス、データ分析の8種類です。

## 5. 購買タイミング (`purchase_timing.py`)

現状への不満やツール導入状態から、DX 商材を購入するタイミングに関係する6特徴量を作成します。

### 必須入力列

`アンケート２`、`アンケート６`、`アンケート７`、`アンケート８`

### 特徴量の定義

| 特徴量 | 計算方法・定義 | 状態 |
|---|---|---|
| `DX全体不満度` | `6 - アンケート２` | 実装済み。元回答は1～5のみ有効 |
| `DX成果不足度` | `6 - アンケート８` | 実装済み。元回答は1～5のみ有効 |
| `現行ツール状態` | `アンケート６` と `アンケート７` から4分類 | 実装済み |
| `現場課題数` | 将来は `今後のDX展望` から課題カテゴリ数を算出予定 | 仮実装（全行欠損） |
| `システム刷新フラグ` | 将来は `今後のDX展望` から更新・刷新・更改・入替予定を抽出予定 | 仮実装（全行欠損） |
| `導入時期フラグ` | 将来は `今後のDX展望` から来期・年度内・年月などを抽出予定 | 仮実装（全行欠損） |

`現行ツール状態` の分類は次のとおりです。

| 条件 | 出力値 |
|---|---|
| `アンケート６ == 2` | `ツール未導入` |
| `アンケート６ == 1` かつ `アンケート７` が1～2 | `ツール導入済み・不満` |
| `アンケート６ == 1` かつ `アンケート７` が3 | `ツール導入済み・普通` |
| `アンケート６ == 1` かつ `アンケート７` が4～5 | `ツール導入済み・満足` |
| 上記以外 | 欠損 |

`現場課題数` で予定されているカテゴリは、老朽化・レガシー、手作業・紙・Excel、データ分断、属人化、人手不足、非効率・二重入力、セキュリティ課題の7種類です。

## 6. 今後のDX展望 (`dx_outlook.py`)

`今後のDX展望` の自由記述を次の順に数値特徴量へ変換します。

1. MeCab（辞書は `unidic-lite`）で日本語を形態素解析
2. 名詞、動詞、形容詞・形状詞、副詞だけを抽出
3. 抽出語のTF-IDFを計算
4. TruncatedSVDで5、10、30次元のいずれかへ圧縮

既定値は30次元です。出力列名は、たとえば10次元なら
`dx_outlook_svd_10_01` ～ `dx_outlook_svd_10_10` です。欠損文は空文字列として
扱います。利用前にリポジトリ直下で `pip install -r requirements.txt` を実行して
`mecab-python3` と `unidic-lite` を導入してください。

学習データだけを手早く変換する場合は次の補助関数を利用できます。

```python
from calc_features import calculate_dx_outlook_features

train_dx = calculate_dx_outlook_features(train, n_components=10)
```

検証・テストデータがある場合は、TF-IDFの語彙・IDFとSVDにデータリークが
起きないよう、学習データだけで `fit` してください。

```python
from calc_features import DXOutlookTfidfSVD

transformer = DXOutlookTfidfSVD(n_components=10)
train_dx = transformer.fit_transform(train)
valid_dx = transformer.transform(valid)
test_dx = transformer.transform(test)
```

クロスバリデーションでは、各foldで新しい `DXOutlookTfidfSVD` を作成し、
そのfoldの学習部分だけに `fit_transform`、検証部分に `transform` を適用します。
TF-IDFの語彙数より大きい次元数は指定できません。

抽出品詞は `target_parts_of_speech` で変更できます。既定値は従来どおり
名詞・動詞・形容詞・形状詞・副詞です。たとえば名詞だけを使う場合は、
`DXOutlookTfidfSVD(target_parts_of_speech=("名詞",))` と指定します。

## 使い方

### 3%未満の業界統合を含む特徴量セットを計算する

```python
import pandas as pd

from calc_features import AllFeaturesV1Transformer

train = pd.read_csv("data/train.csv")
valid = pd.read_csv("data/valid.csv")

transformer = AllFeaturesV1Transformer()
train_features = transformer.fit_transform(train)
valid_features = transformer.transform(valid)
```

`AllFeaturesV1Transformer` は、`dx_outlook.py` を除く5モジュールを一括計算し、
実験3で除外した全行欠損特徴量8列と重複候補3列を除外します。業界は`fit`データ
内の出現率が3%未満なら`その他`へ統合し、validation・testにも同じ保持ルールを
適用します。出力は19列です。

学習データ単体を計算する場合は`all_features_v1(train)`も利用できます。
validation・testへデータごとにこの簡易関数を呼ぶと、それぞれの頻度で判定される
ため、分割評価や予測では上記のtransformerを使用してください。

### v1とDX展望のSVD第2成分を結合する

```python
from calc_features import AllFeaturesV2Transformer

transformer = AllFeaturesV2Transformer()
train_features = transformer.fit_transform(train)
valid_features = transformer.transform(valid)
test_features = transformer.transform(test)
```

`AllFeaturesV2Transformer` は `all_features_v1` の19列に、`今後のDX展望`を
名詞・動詞だけでTF-IDF化し、30次元SVDへ圧縮した第2成分
`dx_outlook_svd_30_02` を加えた20列を返します。TF-IDFとSVDは `fit` に渡した
データだけで学習されます。v1部分の3%業界統合も同じ学習データで決まり、
validation・testには学習済みルールが適用されます。クロスバリデーションでは
foldごとに新しいtransformerを作成してください。

学習データ単体を手早く計算する場合は `all_features_v2(train)` も利用できます。
検証・テストデータがある場合にデータごとにこの簡易関数を呼ぶと変換条件が
異なるため、上記のtransformerを使用してください。

### Python から特徴量だけを計算する

```python
import pandas as pd

from calc_features.calculate import calculate_execution_features
from calc_features.motivation import calculate_motivation_features
from calc_features.adoption_barriers import calculate_adoption_barrier_features
from calc_features.necessity import calculate_necessity_features
from calc_features.purchase_timing import calculate_purchase_timing_features

df = pd.read_csv("data/train.csv")

execution = calculate_execution_features(df)
motivation = calculate_motivation_features(df)
barriers = calculate_adoption_barrier_features(df)
necessity = calculate_necessity_features(df)
timing = calculate_purchase_timing_features(df)

features = pd.concat(
    [execution, motivation, barriers, necessity, timing],
    axis=1,
)
```

`必要性` の出力に含まれる `業界` は入力列と重なるため、元データと全特徴量を連結する場合は重複列に注意してください。また、仮実装の特徴量は全行欠損になるため、学習に使用する前に除外するか、適切な値で補完してください。

### 元データへ特徴量を追加する

```python
import pandas as pd

from calc_features import add_motivation_features, add_adoption_barrier_features
from calc_features.necessity import add_necessity_features
from calc_features.purchase_timing import add_purchase_timing_features

df = pd.read_csv("data/train.csv")
df = add_motivation_features(df)
df = add_adoption_barrier_features(df)
df = add_necessity_features(df)
df = add_purchase_timing_features(df)
```

パッケージ直下の `calc_features` からは、`all_features_v1` と、`motivation.py`、
`adoption_barriers.py` の公開関数・定数を直接 import できます。それ以外は上記の
ように各モジュールから import してください。

### コマンドラインから実行能力特徴量をCSVへ出力する

リポジトリのルートディレクトリで次を実行します。

```bash
python -m calc_features.calculate data/train.csv calc_features/train_features.csv
```

出力CSVには、入力に存在する場合は `企業ID` と `購入フラグ` が先頭に入り、その後に実行能力の9特徴量が出力されます。出力先の親フォルダが存在しない場合は自動的に作成され、CSVは UTF-8 BOM 付きで保存されます。

## ファイル構成

| ファイル | 役割 |
|---|---|
| `calculate.py` | 実行能力の9特徴量、CSV用CLI |
| `motivation.py` | 推進意欲の3特徴量 |
| `adoption_barriers.py` | 導入障壁・充足済み度の6特徴量 |
| `necessity.py` | 必要性の6特徴量 |
| `purchase_timing.py` | 購買タイミングの6特徴量 |
| `dx_outlook.py` | MeCab + TF-IDF + TruncatedSVDによる今後のDX展望特徴量 |
| `all_features_v1.py` | 実験3・4採用後の19特徴量 |
| `all_features_v2.py` | v1の19列とDX展望SVD第2成分を結合した20特徴量 |
| `__init__.py` | パッケージ直下に公開する関数・定数の定義 |
| `test_*.py` | 一部特徴量の単体テスト |
