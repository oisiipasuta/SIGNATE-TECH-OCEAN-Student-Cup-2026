# データ配置

コンペデータそのものはGitHubへコミットしません。各自でコンペサイトから取得し、このディレクトリに次のように配置してください。

```text
data/
├── train.csv
├── test.csv
├── description.csv
└── sample_submit.csv
```

- `train.csv`: 学習用データ
- `test.csv`: 予測対象データ
- `description.csv`: 変数の説明
- `sample_submit.csv`: 提出ファイルのサンプル

配布時のファイル名が異なる場合、ファイルをリネームするか、Notebook上部の`TRAIN_FILENAME`、`TEST_FILENAME`、`DESCRIPTION_FILENAME`、`SAMPLE_SUBMIT_FILENAME`を実際の名前に変更してください。

CSVおよびZIPファイルは`.gitignore`でGit管理対象外にしています。`data/README.md`だけは配置手順としてGitで管理します。
