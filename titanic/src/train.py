from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

RARE_TITLES = {
    "Capt", "Col", "Don", "Dona", "Dr", "Jonkheer", "Lady", "Major",
    "Rev", "Sir", "the Countess",
}


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    title = df["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    df["Title"] = title.where(~title.isin(RARE_TITLES), "Rare")
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["Deck"] = df["Cabin"].str[0].fillna("U")
    df["HasCabin"] = df["Cabin"].notna().astype(int)
    return df


def fill_missing(train: pd.DataFrame, test: pd.DataFrame):
    # Imputation values come from train only to avoid leakage.
    age_by_title = train.groupby("Title")["Age"].median()
    fare_by_class = train.groupby("Pclass")["Fare"].median()
    embarked_mode = train["Embarked"].mode()[0]
    for df in (train, test):
        df["Age"] = df["Age"].fillna(df["Title"].map(age_by_title)).fillna(train["Age"].median())
        df["Fare"] = df["Fare"].fillna(df["Pclass"].map(fare_by_class))
        df["Embarked"] = df["Embarked"].fillna(embarked_mode)
    return train, test


NUMERIC = ["Age", "Fare", "FamilySize", "IsAlone", "HasCabin", "SibSp", "Parch"]
CATEGORICAL = ["Pclass", "Sex", "Embarked", "Title", "Deck"]


def preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def main():
    train = add_features(pd.read_csv(DATA / "train.csv"))
    test = add_features(pd.read_csv(DATA / "test.csv"))
    train, test = fill_missing(train, test)

    X, y = train[NUMERIC + CATEGORICAL], train["Survived"]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(
            n_estimators=500, min_samples_leaf=3, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = {}
    for name, model in models.items():
        pipe = make_pipeline(preprocessor(), model)
        s = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        scores[name] = s.mean()
        print(f"{name:20s} acc = {s.mean():.4f} (+/- {s.std():.4f})")

    best = max(scores, key=scores.get)
    print(f"\nBest model: {best}")
    final = make_pipeline(preprocessor(), models[best]).fit(X, y)

    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": final.predict(test[NUMERIC + CATEGORICAL]),
    })
    out = ROOT / "submission.csv"
    submission.to_csv(out, index=False)
    print(f"Wrote {out} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
