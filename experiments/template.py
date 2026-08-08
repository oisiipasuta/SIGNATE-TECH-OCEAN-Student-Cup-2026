'''
実験ID: exp1
実験名: ベースラインモデルの作成
著者: oisiipasuta

実験概要:
LightGBMを用いたベースラインモデルの作成を行う。

使用する特徴量:
- 売上高
- 従業員数

前処理方法:
- 数値特徴量: 標準化
- カテゴリ特徴量: One-Hotエンコーディング

使用するモデル:
- LightGBM

結果：
threshold=0.5でのOOF F1スコアを計算する。
その結果はF1スコア=0.XXXXとなった。
'''







import numpy as np
import pandas as pd

from pathlib import Path

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ==================================================
# 1. 実験設定
# ==================================================

EXPERIMENT_ID = "exp1"
EXPERIMENT_NAME = "ベースラインモデルの作成"
AUTHOR = "oisiipasuta"

TARGET_COLUMN = "購入フラグ"

NUMERIC_FEATURES = [
    "売上",
    "従業員数",
]

CATEGORICAL_FEATURES = [

]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MODEL_NAME = "LightGBM"

MODEL_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "max_depth": 3,
    "num_leaves": 7,
    "random_state": 42,
}

N_SPLITS = 5
RANDOM_STATE = 42
THRESHOLD = 0.5

base_dir = Path(__file__).resolve().parent.parent
train_dir = base_dir / "data" / "train.csv"
test_dir = base_dir / "data" / "test.csv"
submit_dir = base_dir / "data" / "submit.csv"


# ==================================================
# 2. データ読み込み
# ==================================================
    
train = pd.read_csv(base_dir / "data" / "train.csv")

X = train[FEATURE_COLUMNS].copy()
y = train[TARGET_COLUMN].copy()


# ==================================================
# 3. 特徴量処理
# ==================================================
'''
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ]
)
'''



# ==================================================
# 4. モデル
# ==================================================

model = LGBMClassifier(**MODEL_PARAMS)

'''
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)
'''



# ==================================================
# 5. クロスバリデーション
# ==================================================

cv = StratifiedKFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_probabilities = np.zeros(len(train))
fold_scores = []

for fold, (train_index, valid_index) in enumerate(cv.split(X, y)):
    X_train = X.iloc[train_index]
    X_valid = X.iloc[valid_index]

    y_train = y.iloc[train_index]
    y_valid = y.iloc[valid_index]

    model.fit(X_train, y_train)

    valid_probability = model.predict_proba(X_valid)[:, 1]
    valid_prediction = (valid_probability >= THRESHOLD).astype(int)

    score = f1_score(y_valid, valid_prediction)

    fold_scores.append(score)
    oof_probabilities[valid_index] = valid_probability

    print(f"Fold {fold}: F1 = {score:.4f}")


# ==================================================
# 6. 実験結果
# ==================================================

oof_prediction = (oof_probabilities >= THRESHOLD).astype(int)
oof_f1 = f1_score(y, oof_prediction)

print()
print("=" * 50)
print(f"Experiment: {EXPERIMENT_ID} {EXPERIMENT_NAME}")
print(f"Features: {FEATURE_COLUMNS}")
print(f"Model: {MODEL_NAME}")
print(f"Threshold: {THRESHOLD}")
print(f"Fold F1 mean: {np.mean(fold_scores):.4f}")
print(f"Fold F1 std: {np.std(fold_scores):.4f}")
print(f"OOF F1: {oof_f1:.4f}")
print("=" * 50)

#特徴量重要度の表示および図示
feature_importance = model.feature_importances_
print("Feature Importance:")
for feature, importance in zip(FEATURE_COLUMNS, feature_importance):
    print(f"  {feature}: {importance:.4f}")