import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from algorithms.sgd_logreg import evaluate_sgd
from algorithms.knn_model import evaluate_knn

if __name__ == '__main__':
    sgd = evaluate_sgd('data/Dodgers_processed.csv', cv_splits=10, random_state=42, output_prefix='data/sgd')
    knn = evaluate_knn('data/Dodgers_processed.csv', cv_splits=10, output_prefix='data/knn')
    print("Métricas médias de validação cruzada do SGD (Regressão Logística):", sgd['metricas_media_cv'])
    print("Melhores parâmetros do KNN:", knn['melhores_parametros'])
