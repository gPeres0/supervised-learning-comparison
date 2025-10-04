
# sgd_logreg.py (fast)
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix


import pandas as pd
import numpy as np

def _parse_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    dt = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
    if dt.isna().mean() > 0.5:
        dt = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['_year'] = dt.dt.year
    df['_month'] = dt.dt.month
    df['_day'] = dt.dt.day
    df['_dayofweek'] = dt.dt.dayofweek
    df['_dayofyear'] = dt.dt.dayofyear
    tm = pd.to_datetime(df['Time'], format='%H:%M', errors='coerce')
    df['_hour'] = tm.dt.hour
    df['_minute'] = tm.dt.minute
    df['_minutes_since_midnight'] = df['_hour'] * 60 + df['_minute']
    df['_is_weekend'] = df['_dayofweek'].isin([5, 6]).astype(int)
    df['_count'] = pd.to_numeric(df['Count'], errors='coerce')
    features = df[['_year','_month','_day','_dayofweek','_dayofyear','_hour','_minute','_minutes_since_midnight','_is_weekend','_count']]
    features = features.fillna(features.median(numeric_only=True))
    return features

def build_feature_matrix(df: pd.DataFrame, target_col: str='Event'):
    X = _parse_datetime_features(df)
    y = df[target_col].astype(int).values
    feature_names = X.columns.tolist()
    return X.values, y, feature_names


def make_pipeline(random_state: int=42):
    feat = FunctionTransformer(lambda df: build_feature_matrix(df)[0], validate=False)
    pipe = Pipeline(steps=[
        ('features', feat),
        ('scale', StandardScaler(with_mean=True)),
        ('clf', SGDClassifier(loss='log_loss', penalty='l2', alpha=1e-4, max_iter=1000, tol=1e-3, class_weight='balanced', random_state=random_state))
    ])
    return pipe

def evaluate_sgd(csv_path: str, cv_splits: int=3, random_state: int=42, output_prefix: str='/mnt/data/sgd'):
    df = pd.read_csv(csv_path)
    X, y, feature_names = build_feature_matrix(df)
    pipe = make_pipeline(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    scores = cross_val_predict(pipe, df, y, cv=cv, method='decision_function', n_jobs=1)
    fpr, tpr, thresholds = roc_curve(y, scores)
    auc_val = roc_auc_score(y, scores)

    y_pred = (scores >= 0.0).astype(int)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    cm = confusion_matrix(y, y_pred).tolist()

    acc_cv = cross_val_score(pipe, df, y, cv=cv, scoring='accuracy', n_jobs=1).mean()
    prec_cv = cross_val_score(pipe, df, y, cv=cv, scoring='precision', n_jobs=1).mean()
    rec_cv = cross_val_score(pipe, df, y, cv=cv, scoring='recall', n_jobs=1).mean()
    f1_cv = cross_val_score(pipe, df, y, cv=cv, scoring='f1', n_jobs=1).mean()

    # Resultados do modelo Regressão Logística  
    results = {
        'modelo': 'Regressão Logística',
        'divisões_cv': cv_splits,
        'metricas_ponto_no_conjunto': {'acurácia': acc, 'precisão': prec, 'revocação': rec, 'f1': f1, 'roc_auc': auc_val},
        'metricas_media_cv': {'acurácia': float(acc_cv), 'precisão': float(prec_cv), 'revocação': float(rec_cv), 'f1': float(f1_cv), 'roc_auc': float(auc_val)},
        'matriz_de_confusão': cm,
        'curva_roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'limiares': thresholds.tolist()},
        'nomes_das_variáveis': feature_names,
    }
    # Salva os resultados em arquivo JSON
    with open(f'{output_prefix}_results.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results

if __name__ == '__main__':
    out = evaluate_sgd('data/Dodgers_processed.csv', cv_splits=10, random_state=42)
    print(json.dumps(out['metricas_media_cv'], indent=2, ensure_ascii=False))
