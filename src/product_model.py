from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed/merged_dataset.csv"
MODEL_OUT = ROOT / "models/product_model.pkl"

DROP = ["transaction_id", "customer_id", "purchase_date", "product_category"]


def load_xy(path=DATA):
    df = pd.read_csv(path)
    y = df["product_category"]
    groups = df["customer_id"]
    X = pd.get_dummies(df.drop(columns=DROP), drop_first=True)
    return X, y, groups


def evaluate(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)
    acc = accuracy_score(y_te, pred)
    f1 = f1_score(y_te, pred, average="macro")
    loss = log_loss(y_te, proba, labels=model.classes_)
    print(f"{name:22s} acc={acc:.3f}  macro F1={f1:.3f}  log loss={loss:.3f}")
    return model, acc, f1, loss


def train():
    X, y, groups = load_xy()
    # customer_avg_amount and customer_txn_count are built from every transaction a
    # customer made, so splitting a customer across train and test would let the model
    # see their other purchases through those features. splitting on the customer
    # instead of the row keeps each customer wholly on one side and removes that.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(X, y, groups))
    X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    print(f"train {len(X_tr)} rows, test {len(X_te)} rows, {X.shape[1]} features")
    print(f"customers: {groups.iloc[tr].nunique()} train, {groups.iloc[te].nunique()} test, "
          f"{len(set(groups.iloc[tr]) & set(groups.iloc[te]))} shared\n")

    # "prior" predicts the majority class like most_frequent does, but reports the
    # class frequencies as probabilities instead of a hard 1.0, so its log loss is
    # comparable to the real models rather than an artefact of predicting 0
    evaluate("baseline (prior)", DummyClassifier(strategy="prior"),
             X_tr, X_te, y_tr, y_te)

    scaler = StandardScaler().fit(X_tr)
    evaluate("logistic regression", LogisticRegression(max_iter=2000),
             scaler.transform(X_tr), scaler.transform(X_te), y_tr, y_te)

    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf, acc, f1, loss = evaluate("random forest", rf, X_tr, X_te, y_tr, y_te)

    # one small test split moves a lot with the seed, so cross validation gives a
    # fairer read. grouped by customer for the same reason the holdout is.
    cv = cross_val_score(RandomForestClassifier(n_estimators=50, random_state=42),
                         X, y, groups=groups, cv=GroupKFold(n_splits=5),
                         scoring="accuracy")
    print(f"\nrandom forest grouped 5-fold accuracy: {cv.mean():.3f} +/- {cv.std():.3f}")
    print(f"majority class share:                  {y.value_counts(normalize=True).max():.3f}")

    print(f"\n{classification_report(y_te, rf.predict(X_te), zero_division=0)}")

    MODEL_OUT.parent.mkdir(exist_ok=True)
    joblib.dump({"model": rf, "columns": list(X.columns)}, MODEL_OUT)
    print(f"wrote {MODEL_OUT}")
    return rf


if __name__ == "__main__":
    train()
