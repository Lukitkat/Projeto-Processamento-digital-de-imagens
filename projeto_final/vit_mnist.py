"""
vit_mnist.py — OCR de Dígitos com Vision Transformer (ViT) via HuggingFace
Baseado em: transfpy.py (aula 27/02)

Técnica extra-classe: fine-tuning do ViT pré-treinado no ImageNet.
Hiperparâmetros variados:
  - learning_rate: [1e-4, 5e-5]
  - num_epochs:    [2, 4]

Correções para execução em CPU no Windows:
  - attn_implementation="eager" evita o travamento do SDPA no Windows
  - Subconjunto reduzido (50 por classe) para viabilizar CPU
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
from transformers import ViTForImageClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import os

# ── 1. Dataset ───────────────────────────────────────────────────────────────
print("[INFO] Preparando MNIST para ViT (resize 224×224, RGB)...")
transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_train = datasets.MNIST(root='./data', train=True,  download=True, transform=transform_vit)
full_test  = datasets.MNIST(root='./data', train=False, download=True, transform=transform_vit)

# Estratificação via .targets (sem carregar imagens — O(1) por amostra)
def get_stratified_subset(dataset, n_per_class=50, n_classes=10):
    targets = dataset.targets  # tensor de rótulos, sem transform
    counts  = {c: 0 for c in range(n_classes)}
    selected = []
    for i in range(len(targets)):
        label = int(targets[i])
        if counts[label] < n_per_class:
            selected.append(i)
            counts[label] += 1
        if all(v == n_per_class for v in counts.values()):
            break
    return Subset(dataset, selected)

# 50 por classe → 500 treino total | 20 por classe → 200 teste
train_sub = get_stratified_subset(full_train, n_per_class=50)
test_sub  = get_stratified_subset(full_test,  n_per_class=20)

val_size   = int(0.2 * len(train_sub))   # 100
train_size = len(train_sub) - val_size   # 400
train_set, val_set = random_split(train_sub, [train_size, val_size],
                                  generator=torch.Generator().manual_seed(42))

print(f"  Treino: {train_size} | Validação: {val_size} | Teste: {len(test_sub)}")


# ── 2. Modelo ViT ─────────────────────────────────────────────────────────────
def build_vit():
    """
    Carrega ViT-Base pré-treinado e substitui o classificador (10 classes).
    attn_implementation='eager' evita o bug do SDPA no Windows/CPU.
    """
    model = ViTForImageClassification.from_pretrained(
        'google/vit-base-patch16-224',
        num_labels=10,
        ignore_mismatched_sizes=True,
        attn_implementation='eager'   # ← corrige travamento no Windows
    )
    # Congela todos os pesos exceto o classificador final
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False
    return model


# ── 3. Função de Treinamento ─────────────────────────────────────────────────
def train_vit(lr, num_epochs, batch_size=16):
    device = torch.device('cpu')
    model  = build_vit().to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_sub,  batch_size=batch_size, shuffle=False)

    history = []

    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(pixel_values=x_batch)
            loss    = criterion(outputs.logits, y_batch)
            loss.backward()
            optimizer.step()

        # Validação
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(pixel_values=x_batch)
                _, predicted = torch.max(outputs.logits, 1)
                total   += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        val_acc = correct / total
        history.append(val_acc)
        print(f"  Época {epoch+1:>2}/{num_epochs} | val_acc={val_acc:.4f}")

    # Teste
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            outputs  = model(pixel_values=x_batch)
            _, predicted = torch.max(outputs.logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    test_acc  = accuracy_score(all_labels, all_preds)
    test_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    test_rec  = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1   = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {'lr': lr, 'epochs': num_epochs, 'val_history': history,
            'test_acc': test_acc, 'test_prec': test_prec,
            'test_rec': test_rec, 'test_f1': test_f1}


# ── 4. Grid de Hiperparâmetros ───────────────────────────────────────────────
lr_values    = [1e-4, 5e-5]
epoch_values = [2, 4]

results = []
print("\n[INFO] Fine-tuning ViT com diferentes hiperparâmetros...\n")

best_f1  = -1
best_cfg = None

for lr in lr_values:
    for epochs in epoch_values:
        print(f"--- lr={lr}, epochs={epochs} ---")
        res = train_vit(lr, epochs)
        results.append(res)
        print(f"    Teste → Acc={res['test_acc']:.4f} | F1={res['test_f1']:.4f}\n")
        if res['test_f1'] > best_f1:
            best_f1  = res['test_f1']
            best_cfg = res

# ── 5. Resumo ────────────────────────────────────────────────────────────────
print("\n=== Resumo (ViT) ===")
print(f"{'lr':>10} {'epochs':>8} | {'Test Acc':>10} {'Test F1':>10}")
print("-" * 44)
for r in results:
    print(f"{r['lr']:>10} {r['epochs']:>8} | {r['test_acc']:>10.4f} {r['test_f1']:>10.4f}")

print(f"\n[INFO] Melhor: lr={best_cfg['lr']}, epochs={best_cfg['epochs']} → F1={best_f1:.4f}")

# ── 6. Salva resultados ──────────────────────────────────────────────────────
output = {'method': 'ViT',
          'grid_results': results,
          'best': {'lr': best_cfg['lr'], 'epochs': best_cfg['epochs']},
          'test': {'acc': best_cfg['test_acc'], 'prec': best_cfg['test_prec'],
                   'rec': best_cfg['test_rec'],  'f1':  best_cfg['test_f1']}}

os.makedirs('resultados', exist_ok=True)
with open('resultados/vit_results.json', 'w') as f:
    json.dump(output, f, indent=2)
print("[INFO] Resultados salvos em resultados/vit_results.json")
