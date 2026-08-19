# 実験ディレクトリ

実験は対象とする特徴量・モデルごとにディレクトリを分け、各ディレクトリ内の `expNN.py` で管理する。
現在は4系列、全21本の実験スクリプトがある。

## ディレクトリ構成

| ディレクトリ | 実験ID | 概要 |
| --- | --- | --- |
| `exp_base/` | `exp01`〜`exp06` | DX展望のテキスト本文を除く `calc_features` 特徴量をLightGBMで検証 |
| `exp_dx_outlook/` | `exp01`〜`exp06` | 「今後のDX展望」の文字・品詞特徴量をロジスティック回帰で検証 |
| `exp_dx_outlook_lightgbm/` | `exp03`〜`exp10` | 「今後のDX展望」の品詞別TF-IDF・SVD特徴量をLightGBMで検証 |
| `exp_mix/` | `exp01` | 複数ジャンルの特徴量を結合してLightGBMで検証 |

## `exp_base`

| ID | 旧ID | 内容 |
| --- | --- | --- |
| `exp01` | `exp1` | `calc_features`（DX展望を除く）+ LightGBMベースライン |
| `exp02` | `exp2` | `scale_pos_weight` の比較 |
| `exp03` | `exp3` | 重複特徴量の除外 |
| `exp04` | `exp4` | 業界特徴量の有無・カテゴリ統合の比較 |
| `exp05` | `exp5` | 重要特徴量のみのモデル |
| `exp06` | `exp6` | 実行能力・必要性・意欲の3軸アブレーション |

特徴量の詳細は [`exp_base/FEATURES.md`](exp_base/FEATURES.md) を参照する。

実行例: `python experiments/exp_base/exp01.py`

## `exp_dx_outlook`

| ID | 旧ID | 内容 |
| --- | --- | --- |
| `exp01` | `exp01` | MeCabを使わない文字N-gram TF-IDFベースライン |
| `exp02` | `exp08` | 品詞別TF-IDF・SVD + ロジスティック回帰 |
| `exp03` | `exp11` | 品詞アブレーション（名詞のみ） |
| `exp04` | `exp12` | 品詞アブレーション（名詞 + 動詞） |
| `exp05` | `exp13` | 品詞アブレーション（名詞 + 動詞 + 形容詞・形状詞） |
| `exp06` | `exp14` | 品詞アブレーション（名詞 + 動詞 + 形容詞・形状詞 + 副詞） |

`exp03`〜`exp06` の共通処理は `_pos_ablation_common.py` にまとめている。

実行例: `python experiments/exp_dx_outlook/exp01.py`

## `exp_dx_outlook_lightgbm`

| ID | 内容 |
| --- | --- |
| `exp03` | LightGBM品詞アブレーション（名詞のみ） |
| `exp04` | LightGBM品詞アブレーション（名詞 + 動詞） |
| `exp05` | LightGBM品詞アブレーション（名詞 + 動詞 + 形容詞・形状詞） |
| `exp06` | LightGBM品詞アブレーション（名詞 + 動詞 + 形容詞・形状詞 + 副詞） |
| `exp07` | SVD第2成分のみ |
| `exp08` | SVD第2成分を除外 |
| `exp09` | SVD全30成分の再現ベースライン |
| `exp10` | SVD成分数をinner CV内で選ぶリーク防止ネステッド比較 |

この系列には現在 `exp01.py` と `exp02.py` はない。全実験の共通処理は `_common.py` にまとめている。

実行例: `python experiments/exp_dx_outlook_lightgbm/exp03.py`

## `exp_mix`

| ID | 内容 |
| --- | --- |
| `exp01` | `all_features_v2`（`all_features_v1` 19列 + DX展望のSVD第2成分）による混合特徴量ベースライン |

補足は [`exp_mix/README.md`](exp_mix/README.md) を参照する。

実行例: `python experiments/exp_mix/exp01.py`

## 共通ファイルと出力

- 新しい実験のひな形は `template.py` に置く。
- 各実験の画像は `results/<実験ID>/feature_importance.png` と `results/<実験ID>/f1_scores.png` に保存する。
- 現在存在する21本の実験について、上記2種類の画像がすべて保存されている。
- `__pycache__/` と `*.pyc` はPython実行時に生成されるキャッシュであり、実験本体ではない。
