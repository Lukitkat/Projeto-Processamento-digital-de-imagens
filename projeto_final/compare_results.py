"""
compare_results.py — Compara SVM, CNN e ViT e gera gráficos
Execute APÓS rodar: svm_mnist.py, cnn_mnist.py, vit_mnist.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 1. Carrega resultados ────────────────────────────────────────────────────
res_dir = 'resultados'
os.makedirs(res_dir, exist_ok=True)

def load_json(fname):
    path = os.path.join(res_dir, fname)
    if not os.path.exists(path):
        print(f"[AVISO] {fname} não encontrado. Execute o script correspondente primeiro.")
        return None
    with open(path) as f:
        return json.load(f)

svm_data = load_json('svm_results.json')
cnn_data = load_json('cnn_results.json')
vit_data = load_json('vit_results.json')

# ── 2. Tabela Comparativa (melhores configurações) ───────────────────────────
print("\n" + "="*65)
print(" Comparação Final — Melhor Configuração de cada Método (Teste)")
print("="*65)
print(f"{'Método':>10} | {'Acurácia':>10} {'Precisão':>10} {'Recall':>10} {'F1':>10}")
print("-"*65)

summary = []
for data, name in [(svm_data, 'SVM+HOG'), (cnn_data, 'CNN'), (vit_data, 'ViT')]:
    if data is None:
        continue
    t = data['test']
    print(f"{name:>10} | {t['acc']:>10.4f} {t['prec']:>10.4f} {t['rec']:>10.4f} {t['f1']:>10.4f}")
    summary.append({'method': name, **t})

# ── 3. Gráfico 1: Comparação F1 por método ──────────────────────────────────
if summary:
    methods = [s['method'] for s in summary]
    metrics = ['acc', 'prec', 'rec', 'f1']
    labels  = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    colors  = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    x = np.arange(len(methods))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        vals = [s[metric] for s in summary]
        ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)

    ax.set_ylabel('Valor')
    ax.set_title('Comparação de Métricas por Método (Melhor Hiperparâmetro)')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(methods, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'comparacao_metodos.png'), dpi=150)
    plt.close()
    print(f"\n[INFO] Gráfico salvo: {res_dir}/comparacao_metodos.png")

# ── 4. Gráfico 2: SVM — F1 por C e gamma ────────────────────────────────────
if svm_data:
    grid = svm_data['grid_results']
    C_vals     = sorted(set(r['C'] for r in grid))
    gamma_vals = sorted(set(r['gamma'] for r in grid))
    colors_g   = ['#1f77b4', '#ff7f0e', '#2ca02c']

    fig, ax = plt.subplots(figsize=(9, 5))
    for gcolor, gval in zip(colors_g, gamma_vals):
        y_vals = [r['val_f1'] for r in grid if r['gamma'] == gval]
        ax.plot([str(c) for c in C_vals], y_vals, marker='o', label=f'gamma={gval}', color=gcolor)

    ax.set_xlabel('C (regularização)')
    ax.set_ylabel('F1-Score (Validação)')
    ax.set_title('SVM+HOG — F1-Score por C e gamma')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'svm_hiperparametros.png'), dpi=150)
    plt.close()
    print(f"[INFO] Gráfico salvo: {res_dir}/svm_hiperparametros.png")

# ── 5. Gráfico 3: CNN — curva de val_acc por época ──────────────────────────
if cnn_data:
    grid = cnn_data['grid_results']
    lr_vals = sorted(set(r['lr'] for r in grid))
    ep_vals = sorted(set(r['epochs'] for r in grid))

    fig, axes = plt.subplots(1, len(lr_vals), figsize=(14, 5), sharey=True)
    cmap = plt.cm.get_cmap('tab10')

    for ax, lr in zip(axes, lr_vals):
        for j, ep in enumerate(ep_vals):
            match = [r for r in grid if r['lr'] == lr and r['epochs'] == ep]
            if match:
                history = match[0]['val_history']
                ax.plot(range(1, len(history)+1), history,
                        marker='o', label=f'{ep} épocas', color=cmap(j))
        ax.set_title(f'lr={lr}')
        ax.set_xlabel('Época')
        ax.set_ylabel('Acurácia (Validação)')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle('CNN — Acurácia de Validação por learning rate e epochs', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'cnn_hiperparametros.png'), dpi=150)
    plt.close()
    print(f"[INFO] Gráfico salvo: {res_dir}/cnn_hiperparametros.png")

# ── 6. Gráfico 4: ViT — F1 por configuração ──────────────────────────────────
if vit_data:
    grid = vit_data['grid_results']
    labels_v = [f"lr={r['lr']}\nep={r['epochs']}" for r in grid]
    f1_vals  = [r['test_f1'] for r in grid]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels_v, f1_vals, color='#9467bd', alpha=0.85)
    for bar, val in zip(bars, f1_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_ylabel('F1-Score (Teste)')
    ax.set_title('ViT (fine-tuning) — F1-Score por configuração')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(res_dir, 'vit_hiperparametros.png'), dpi=150)
    plt.close()
    print(f"[INFO] Gráfico salvo: {res_dir}/vit_hiperparametros.png")

print("\n[INFO] compare_results.py concluído.")
