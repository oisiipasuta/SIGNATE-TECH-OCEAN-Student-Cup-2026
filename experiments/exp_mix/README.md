# exp_mix

複数ジャンルの特徴量を結合して評価する本実験を配置するディレクトリです。

最初の特徴量セットとして、`calc_features.all_features_v2` が
`all_features_v1` の19列と「今後のDX展望」のSVD第2成分を結合します。

## 実験結果

| ID | 特徴量セット | nested OOF F1 | 最終閾値 |
| --- | --- | ---: | ---: |
| `exp02` | all_features_v2（20列） | 0.7634 | 0.2830 |
| `exp03` | all_features_v3（v2 + アンケート3・6・10、23列） | 0.7744 | 0.3260 |
| `exp04` | all_features_v4（v3 + 汎用組織図15特徴、38列） | **0.7989** | 0.2770 |
| `exp05` | all_features_v3 + 組織図Top-2特徴（25列） | **0.8063** | 0.2470 |
| `exp06` | all_features_v3 + DX変革組織有無（24列） | 0.7804 | 0.2460 |
| `exp07` | all_features_v3 + 平均分岐数（24列） | 0.7538 | 0.2420 |

`exp03`は`exp_ai/exp09`を再実装したもので、特徴量計算を
`calc_features.all_features_v3.AllFeaturesV3Transformer`へ切り出しています。
業界統合、TF-IDF、SVD、補完、One-Hot Encoding、閾値選択はすべて学習fold内で
行います。

`exp04`は`exp_tree_of_corp/exp02`を`AllFeaturesV4Transformer`で再実装し、
特徴量数、各foldの閾値・F1、nested OOF F1を完全再現しました。

`exp05`は`exp_tree_of_corp/exp08`の累積Top-k比較で最良だった
`DX変革組織有無`と`平均分岐数`だけを`all_features_v3`へ追加し、同実験の
Top-2結果を完全再現しました。

`exp06`はTop-2から`平均分岐数`を除き、`DX変革組織有無`だけを追加した
Top-1構成です。`exp_tree_of_corp/exp08`のTop-1結果を完全再現しました。

`exp07`はTop-2から`DX変革組織有無`を除き、`平均分岐数`だけを追加した
単独効果のアブレーションです。nested OOF F1は0.7538で、`all_features_v3`
単独の0.7744を下回りました。`exp05`の改善は平均分岐数単独では再現しません。
