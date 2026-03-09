"""
svm_mnist.py — OCR de Dígitos com SVM + HOG (dataset MNIST)
Baseado em: ocrsvm.py (aula 24/02)

Hiperparâmetros variados:
  - C: [0.1, 1, 10, 100]
  - gamma: ['scale', 0.001, 0.01]
"""

import cv2 as cv
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# ── 1. Carregamento do Dataset ──────────────────────────────────────────────
print("[INFO] Carregando MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
X_raw, y_raw = mnist.data.astype(np.uint8), mnist.target.astype(int)

# Usa apenas 10 000 amostras para viabilizar SVM em tempo razoável
N = 10000
X_raw, y_raw = X_raw[:N], y_raw[:N]

# Divide treino (70%), validação (15%), teste (15%)
X_temp, X_test, y_temp, y_test = train_test_split(
    X_raw, y_raw, test_size=0.15, random_state=42, stratify=y_raw
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp  # 0.176 * 0.85 ≈ 0.15
)

print(f"  Treino: {len(X_train)} | Validação: {len(X_val)} | Teste: {len(X_test)}")

# ── 2. Extração de Descritores HOG ───────────────────────────────────────────
# Cada imagem MNIST é 28x28 → redimensionamos para 20x20 (igual à aula)
def extract_hog(X):
    hog = cv.HOGDescriptor((20, 20), (10, 10), (5, 5), (10, 10), 9)
    descriptors = []
    for sample in X:
        img = sample.reshape(28, 28)
        img_resized = cv.resize(img, (20, 20))
        desc = hog.compute(img_resized)
        descriptors.append(desc)
    return np.array(descriptors, dtype=np.float32)

print("[INFO] Extraindo descritores HOG...")
X_train_hog = extract_hog(X_train)
X_val_hog   = extract_hog(X_val)
X_test_hog  = extract_hog(X_test)

# ── 3. Grid de Hiperparâmetros ───────────────────────────────────────────────
C_values     = [0.1, 1, 10, 100]
gamma_values = ['scale', 0.001, 0.01]

results = []

print("\n[INFO] Treinando SVM com diferentes hiperparâmetros...\n")
print(f"{'C':>8} {'gamma':>8} | {'Val Acc':>8} {'Val P':>8} {'Val R':>8} {'Val F1':>8}")
print("-" * 60)

best_val_f1 = -1
best_cfg = None

for C in C_values:
    for gamma in gamma_values:
        svm = cv.ml.SVM_create()
        svm.setKernel(cv.ml.SVM_RBF)
        svm.setType(cv.ml.SVM_C_SVC)
        svm.setC(C)
        if gamma == 'scale':
            # scale = 1 / (n_features * X.var())
            g_val = float(1.0 / (X_train_hog.shape[1] * X_train_hog.var()))
            svm.setGamma(g_val)
            gamma_label = 'scale'
        else:
            svm.setGamma(gamma)
            gamma_label = str(gamma)

        labels_col = y_train.reshape(-1, 1).astype(np.int32)
        svm.train(X_train_hog, cv.ml.ROW_SAMPLE, labels_col)

        _, val_pred = svm.predict(X_val_hog)
        val_pred = val_pred.ravel().astype(int)

        acc  = accuracy_score(y_val, val_pred)
        prec = precision_score(y_val, val_pred, average='macro', zero_division=0)
        rec  = recall_score(y_val, val_pred, average='macro', zero_division=0)
        f1   = f1_score(y_val, val_pred, average='macro', zero_division=0)

        print(f"{C:>8} {gamma_label:>8} | {acc:>8.4f} {prec:>8.4f} {rec:>8.4f} {f1:>8.4f}")

        entry = {'C': C, 'gamma': gamma_label, 'val_acc': acc, 'val_prec': prec,
                 'val_rec': rec, 'val_f1': f1}
        results.append(entry)

        if f1 > best_val_f1:
            best_val_f1 = f1
            best_cfg = (C, gamma, gamma_label)
            best_model = svm

# ── 4. Avaliação Final no Conjunto de Teste ──────────────────────────────────
print(f"\n[INFO] Melhor configuração: C={best_cfg[0]}, gamma={best_cfg[2]}")
_, test_pred = best_model.predict(X_test_hog)
test_pred = test_pred.ravel().astype(int)

test_acc  = accuracy_score(y_test, test_pred)
test_prec = precision_score(y_test, test_pred, average='macro', zero_division=0)
test_rec  = recall_score(y_test, test_pred, average='macro', zero_division=0)
test_f1   = f1_score(y_test, test_pred, average='macro', zero_division=0)

print(f"\n=== Resultado Final no Teste ===")
print(f"  Acurácia : {test_acc:.4f}")
print(f"  Precisão : {test_prec:.4f}")
print(f"  Recall   : {test_rec:.4f}")
print(f"  F1-Score : {test_f1:.4f}")

# ── 5. Salva resultados ──────────────────────────────────────────────────────
final = {
    'method': 'SVM+HOG',
    'grid_results': results,
    'best': {'C': best_cfg[0], 'gamma': best_cfg[2]},
    'test': {'acc': test_acc, 'prec': test_prec, 'rec': test_rec, 'f1': test_f1}
}

os.makedirs('resultados', exist_ok=True)
with open('resultados/svm_results.json', 'w') as f:
    json.dump(final, f, indent=2)
print("\n[INFO] Resultados salvos em resultados/svm_results.json")
