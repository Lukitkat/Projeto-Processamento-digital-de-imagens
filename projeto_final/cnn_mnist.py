"""
cnn_mnist.py — OCR de Dígitos com CNN (dataset MNIST)
Baseado em: cnnocr.py (aula 27/02)

Hiperparâmetros variados:
  - learning_rate: [0.1, 0.01, 0.001]
  - epochs: [5, 10, 20]
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import json
import os

# ── 1. Dataset MNIST via torchvision ────────────────────────────────────────
print("[INFO] Carregando MNIST via torchvision...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

full_train = datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_set   = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Divide treino (70%) e validação (15%) — teste já vem separado (15% ~10 000/70 000)
val_size   = int(0.15 * len(full_train))
train_size = len(full_train) - val_size
train_set, val_set = random_split(full_train, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))

print(f"  Treino: {train_size} | Validação: {val_size} | Teste: {len(test_set)}")


# ── 2. Arquitetura CNN (igual ao cnnocr.py) ──────────────────────────────────
# MNIST: imagens 1×28×28  →  MaxPool(2): 1×14×14 → MaxPool(2): 1×7×7
def build_model(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(1, 16, kernel_size=3, padding=1),   # → 16×28×28
        nn.ReLU(),
        nn.MaxPool2d(2),                               # → 16×14×14
        nn.Conv2d(16, 32, kernel_size=3, padding=1),  # → 32×14×14
        nn.ReLU(),
        nn.MaxPool2d(2),                               # → 32×7×7
        nn.Flatten(),
        nn.Linear(32 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )


# ── 3. Função de Treinamento ─────────────────────────────────────────────────
def train_and_eval(lr, epochs, batch_size=64):
    device = torch.device('cpu')
    model  = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

    history = []

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

        # Validação
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                output = model(x_batch)
                _, predicted = torch.max(output, 1)
                total   += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_acc = correct / total
        history.append(val_acc)
        print(f"  Época {epoch+1:>2}/{epochs} | val_acc={val_acc:.4f}")

    # Avaliação final no teste
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            output = model(x_batch)
            _, predicted = torch.max(output, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    test_acc  = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        'lr': lr, 'epochs': epochs,
        'val_history': history,
        'test_acc': test_acc, 'test_prec': test_prec,
        'test_rec': test_rec, 'test_f1': test_f1
    }


# ── 4. Grid de Hiperparâmetros ───────────────────────────────────────────────
lr_values    = [0.1, 0.01, 0.001]
epoch_values = [5, 10, 20]

results = []
print("\n[INFO] Treinando CNN com diferentes hiperparâmetros...\n")

best_test_f1 = -1
best_cfg = None

for lr in lr_values:
    for epochs in epoch_values:
        print(f"--- lr={lr}, epochs={epochs} ---")
        res = train_and_eval(lr, epochs)
        results.append(res)
        print(f"    Teste → Acc={res['test_acc']:.4f} | P={res['test_prec']:.4f} "
              f"| R={res['test_rec']:.4f} | F1={res['test_f1']:.4f}\n")

        if res['test_f1'] > best_test_f1:
            best_test_f1 = res['test_f1']
            best_cfg = res

# ── 5. Resumo ────────────────────────────────────────────────────────────────
print("\n=== Resumo (todas as configurações) ===")
print(f"{'lr':>8} {'epochs':>8} | {'Test Acc':>10} {'Test F1':>10}")
print("-" * 44)
for r in results:
    print(f"{r['lr']:>8} {r['epochs']:>8} | {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")

print(f"\n[INFO] Melhor: lr={best_cfg['lr']}, epochs={best_cfg['epochs']} → F1={best_test_f1:.4f}")

# ── 6. Salva resultados ──────────────────────────────────────────────────────
output = {'method': 'CNN', 'grid_results': results,
          'best': {'lr': best_cfg['lr'], 'epochs': best_cfg['epochs']}}

os.makedirs('resultados', exist_ok=True)
with open('resultados/cnn_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("[INFO] Resultados salvos em resultados/cnn_results.json")
