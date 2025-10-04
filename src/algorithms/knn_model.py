# knn_model.py (fast)
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)


def _parse_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["_year"] = dt.dt.year
    df["_month"] = dt.dt.month
    df["_day"] = dt.dt.day
    df["_dayofweek"] = dt.dt.dayofweek
    df["_dayofyear"] = dt.dt.dayofyear
    tm = pd.to_datetime(df["Time"], format="%H:%M", errors="coerce")
    df["_hour"] = tm.dt.hour
    df["_minute"] = tm.dt.minute
    df["_minutes_since_midnight"] = df["_hour"] * 60 + df["_minute"]
    df["_is_weekend"] = df["_dayofweek"].isin([5, 6]).astype(int)
    df["_count"] = pd.to_numeric(df["Count"], errors="coerce")
    features = df[
        [
            "_year",
            "_month",
            "_day",
            "_dayofweek",
            "_dayofyear",
            "_hour",
            "_minute",
            "_minutes_since_midnight",
            "_is_weekend",
            "_count",
        ]
    ]
    features = features.fillna(features.median(numeric_only=True))
    return features


def build_feature_matrix(df: pd.DataFrame, target_col: str = "Event"):
    X = _parse_datetime_features(df)
    y = df[target_col].astype(int).values
    feature_names = X.columns.tolist()
    return X.values, y, feature_names


def make_pipeline(n_neighbors=5, metric="minkowski", p=2):
    feat = FunctionTransformer(lambda df: build_feature_matrix(df)[0], validate=False)
    pipe = Pipeline(
        steps=[
            ("features", feat),
            ("scale", StandardScaler(with_mean=True)),
            ("knn", KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric, p=p)),
        ]
    )
    return pipe


def evaluate_knn(
    csv_path: str, cv_splits: int = 3, output_prefix: str = "data/knn"
):
    df = pd.read_csv(csv_path)
    X, y, feature_names = build_feature_matrix(df)

    param_grid = {
        "knn__n_neighbors": [5, 11],
        "knn__metric": ["minkowski"],
        "knn__p": [2],
    }
    base_pipe = make_pipeline()
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    grid = GridSearchCV(
        base_pipe, param_grid, scoring="f1", cv=cv, n_jobs=1, refit=True
    )
    grid.fit(df, y)
    best_pipe = grid.best_estimator_
    best_params = grid.best_params_

    proba = cross_val_predict(
        best_pipe, df, y, cv=cv, method="predict_proba", n_jobs=1
    )[:, 1]
    fpr, tpr, thresholds = roc_curve(y, proba)
    auc_val = roc_auc_score(y, proba)

    y_pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()

    # Resultados do modelo KNN
    results = {
        "modelo": "KNN",
        "divisões_cv": cv_splits,
        "melhores_parametros": best_params,
        "metricas_ponto_no_conjunto": {
            "acurácia": acc,
            "precisão": prec,
            "revocação": rec,
            "f1": f1,
            "roc_auc": auc_val,
        },
        "matriz_de_confusão": cm,
        "curva_roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "limiares": thresholds.tolist(),
        },
        "nomes_das_variáveis": feature_names,
    }
    # Salva os resultados em arquivo JSON
    with open(f"{output_prefix}_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results


if __name__ == "__main__":
    out = evaluate_knn("data/Dodgers_processed.csv", cv_splits=10)
    print("Melhores parâmetros:", out["melhores_parametros"])
