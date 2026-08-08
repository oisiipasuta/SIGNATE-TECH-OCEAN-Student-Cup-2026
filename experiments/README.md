# 実験ディレクトリ

実験は対象とする特徴量ごとに分け、各ディレクトリ内で `exp01.py` から連番にする。

## `exp_base`

DX展望のテキスト本文を除く `calc_features` の特徴量を、LightGBMベースライン上で検証する実験。

| 現在のID | 旧ID | 内容 |
| --- | --- | --- |
| `exp01` | `exp1` | LightGBMベースライン |
| `exp02` | `exp2` | `scale_pos_weight` 比較 |
| `exp03` | `exp3` | 重複特徴量の除外 |
| `exp04` | `exp4` | 業界特徴量の比較 |
| `exp05` | `exp5` | 重要特徴量のみのモデル |
| `exp06` | `exp6` | 3軸アブレーション |

実行例: `python experiments/exp_base/exp01.py`

## `exp_dx_outlook`

「今後のDX展望」のテキストから特徴量を計算・検証する実験。

| 現在のID | 旧ID | 内容 |
| --- | --- | --- |
| `exp01` | `exp01` | 文字N-gram TF-IDFベースライン |
| `exp02` | `exp08` | 品詞別TF-IDF・SVD |
| `exp03` | `exp11` | 名詞のみ |
| `exp04` | `exp12` | 名詞＋動詞 |
| `exp05` | `exp13` | 名詞＋動詞＋形容詞・形状詞 |
| `exp06` | `exp14` | 名詞＋動詞＋形容詞・形状詞＋副詞 |

実行例: `python experiments/exp_dx_outlook/exp01.py`

各実験の画像は、それぞれのディレクトリにある `results/<実験ID>/` へ保存する。
共通テンプレートは `experiments/template.py` に置く。
