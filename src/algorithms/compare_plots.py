
# compare_plots.py
import json
import matplotlib.pyplot as plt
import numpy as np

def _load(path):
    with open(path, 'r') as f:
        return json.load(f)

def plot_bar_compare(sgd_res, knn_res, outpath='data/metrics_comparison.png'):
    nomes = ['acurácia','precisão','revocação','f1','roc_auc']
    sgd_vals = [sgd_res['metricas_ponto_no_conjunto'][k] for k in nomes]
    knn_vals = [knn_res['metricas_ponto_no_conjunto'][k] for k in nomes]
    x = np.arange(len(nomes))
    largura = 0.35
    plt.figure()
    plt.bar(x - largura/2, sgd_vals, largura, label='SGD (Regressão Logística)')
    plt.bar(x + largura/2, knn_vals, largura, label='KNN')
    plt.xticks(x, nomes)
    plt.ylabel('Pontuação')
    plt.title('Métricas Comparativas (limiar=0.5)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def plot_roc_both(sgd_res, knn_res, outpath='data/roc_both.png'):
    plt.figure()
    plt.plot(sgd_res['curva_roc']['fpr'], sgd_res['curva_roc']['tpr'], label=f"SGD (AUC={sgd_res['metricas_ponto_no_conjunto']['roc_auc']:.3f})")
    plt.plot(knn_res['curva_roc']['fpr'], knn_res['curva_roc']['tpr'], label=f"KNN (AUC={knn_res['metricas_ponto_no_conjunto']['roc_auc']:.3f})")
    plt.plot([0,1], [0,1], linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos (Revocação)')
    plt.title('Curvas ROC')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

if __name__ == '__main__':
    sgd_res = _load('data/sgd_results.json')
    knn_res = _load('data/knn_results.json')
    plot_bar_compare(sgd_res, knn_res)
    plot_roc_both(sgd_res, knn_res)
    print('Gráficos salvos: data/metrics_comparison.png, data/roc_both.png')
