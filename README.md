# SIGNATE TECH OCEAN Student Cup 2026

SIGNATEのデータコンペにチームで取り組むための、Notebook中心の実験リポジトリです。

## プロジェクトの目的

各実験で使用した特徴量、特徴量処理、モデル、F1スコアを比較し、チーム内で再現可能な形で管理することを目的とします。実験の詳細はNotebook、比較に必要な要約は`experiment_log.csv`に残します。

共通の初期設定は、5分割の`StratifiedKFold`、乱数seed 42、判定閾値0.5です。これらを変更した実験ではNotebookと実験ログに値を記録し、共通設定自体を変える場合はこのREADMEも更新してください。

## ディレクトリ構成

```text
SIGNATE-TECH-OCEAN-Student-Cup-2026/
├── data/
│   └── README.md                  # データの取得・配置方法
├── experiments/
│   └── exp001_template.ipynb     # 新規実験用Notebookテンプレート
├── experiment_log.csv            # 実験結果の比較表
├── requirements.txt              # Python依存ライブラリ
├── .gitignore                    # データやローカル環境の除外設定
└── README.md                     # 本ファイル
```

コンペのCSVファイルはGitHubへコミットしません。

## 環境構築

リポジトリのルートで仮想環境を作成します。

```bash
python -m venv .venv
```

Windows（PowerShell）で有効化する場合：

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows（コマンドプロンプト）で有効化する場合：

```bat
.venv\Scripts\activate.bat
```

macOS / Linuxで有効化する場合：

```bash
source .venv/bin/activate
```

依存ライブラリをインストールします。

```bash
python -m pip install -r requirements.txt
```

Jupyterを起動する場合：

```bash
jupyter notebook
```

## データ配置

コンペサイトから各自でデータを取得し、`data/`に配置してください。詳しいファイル名と注意事項は[data/README.md](data/README.md)を参照してください。

## 新しい実験の始め方

1. `experiments/template.py`をコピーする。
2. コピーを`exp002_実験名.py`のように変更する。
3. Notebook上部に使用特徴量、特徴量処理、モデル、CV設定を記録する。
4. Notebookを上から順に実行する。
5. `experiment_log.csv`に結果を1行追加する。
6. 実験ブランチからPull Requestを作成する。

`experiment_log.csv`の`features`と`feature_pipeline`には比較しやすい短い名前を記載し、詳細は対応するNotebookに残してください。

## GitHubの最小運用ルール

- `main`ブランチを直接編集しない。
- 実験ごとにブランチを作成する。
- ブランチ名は`exp/expXXX-short-name`形式とする。
- 1つのNotebookを1つの実験単位とする。
- Pull Requestで実験内容と結果を共有する。
- 共通CV設定を変更した場合はREADMEにも記録する。

## Pull Requestに記載する内容

以下をPull Request本文のテンプレートとして利用してください。

```markdown
## 実験ID

expXXX

## 仮説

検証した仮説を書く。

## 使用特徴量

使用した特徴量を書く。

## 特徴量処理

行った処理を書く。

## 使用モデル

モデル名と主要パラメータを書く。

## 結果

- fold平均F1：
- OOF F1：
- 閾値：
- 比較対象との差分：

## 結論

採用／不採用と、その理由を書く。
```
