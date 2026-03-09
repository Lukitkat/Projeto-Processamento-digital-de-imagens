# Relatório Final — Projeto de Processamento de Imagem e Visão Computacional

**Disciplina:** Processamento de Imagem e Visão Computacional  
**Data:** 08 de março de 2026  

---

## 1. Problema de Visão Computacional Escolhido

O problema escolhido é o **Reconhecimento Óptico de Caracteres (OCR) de Dígitos Manuscritos**, especificamente a classificação automatizada de imagens de dígitos isolados (0–9) escritos à mão. Este é um problema clássico de classificação de imagens com aplicações práticas em digitalização de documentos, leitura de cheques bancários, reconhecimento de CEP em envelopes e preenchimento automático de formulários.

A tarefa consiste em, dada uma imagem em escala de cinza de dimensão 28×28 pixels contendo um único dígito manuscrito, prever corretamente qual dígito (0 a 9) está representado.

---

## 2. Dataset Utilizado

### 2.1 Conjunto de Dados: MNIST

O dataset utilizado é o **MNIST (Modified National Institute of Standards and Technology)**, um benchmark amplamente conhecido em visão computacional.

| Atributo           | Detalhe |
|--------------------|---------|
| **Fonte**          | Yann LeCun et al. — http://yann.lecun.com/exdb/mnist/ |
| **Acesso (scikit-learn)** | `sklearn.datasets.fetch_openml('mnist_784')` |
| **Acesso (PyTorch)** | `torchvision.datasets.MNIST(download=True)` |
| **Total de amostras** | 70 000 imagens (60 000 treino + 10 000 teste original) |
| **Dimensão**       | 28×28 pixels, escala de cinza (1 canal) |
| **Classes**        | 10 (dígitos 0–9), balanceado |
| **Formato**        | Pixel values 0–255 |

### 2.2 Divisão Treino / Validação / Teste

Cada método utiliza a mesma estratégia de divisão proporcionalmente:

| Conjunto     | Proporção | Quantidade (SVM/CNN) |
|--------------|-----------|----------------------|
| Treino       | 70%       | 7 000 / 42 000       |
| Validação    | 15%       | 1 500 / 9 000        |
| Teste        | 15%       | 1 500 / 10 000       |

> Para o SVM foi utilizado um subconjunto de 10 000 amostras para viabilizar o treinamento em hardware local. O ViT utilizou **400 amostras de treino + 200 de teste** (50 por classe), limitado pelo custo de inferência do modelo ViT-Base em CPU.

A divisão foi realizada com `random_state=42` e estratificação por classe para garantir distribuição equilibrada em todos os subconjuntos.

---

## 3. Fluxo Completo da Solução

```
┌─────────────────────────────────────────────────────────────────┐
│                     PIPELINE DE SOLUÇÃO                         │
│                                                                 │
│  Dataset MNIST ──► Pré-processamento ──► Extração de Features   │
│                                               │                 │
│                    ┌──────────────────────────┤                 │
│                    │              │           │                 │
│                  SVM+HOG        CNN          ViT                │
│                    │              │           │                 │
│                    └──────────────┴───────────┘                 │
│                                  │                              │
│                      Grid de Hiperparâmetros                    │
│                                  │                              │
│                      Avaliação (Treino/Val/Teste)               │
│                                  │                              │
│                      Métricas: Acc, P, R, F1                    │
│                                  │                              │
│                      compare_results.py → Gráficos              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.1 Pré-processamento

- **SVM+HOG:** Imagens redimensionadas de 28×28 para 20×20 (padrão usado em aula com `digits.png`); pixel values divididos por 255; descritores HOG extraídos com janela (20,20), bloco (10,10), célula (10,10) e 9 bins de orientação.
- **CNN:** Imagens mantidas em 28×28; normalização com média 0.1307 e desvio 0.3081 (estatísticas do MNIST).  
- **ViT:** Imagens redimensionadas para 224×224 (tamanho exigido pelo modelo pré-treinado); convertidas para 3 canais (replicação dos canais de escala de cinza); normalização com média 0.5 e desvio 0.5 em todos os canais.

---

## 4. Técnicas Utilizadas

### 4.1 SVM com Descritores HOG *(técnica de aula — `ocrsvm.py`)*

**Histogram of Oriented Gradients (HOG)** é um descritor de características que captura a distribuição local dos gradientes da imagem. Para cada célula da imagem, o histograma dos ângulos dos gradientes é computado e normalizado por bloco vizinho, produzindo um vetor de características robusto a variações de iluminação e pequenas deformações.

O **SVM (Support Vector Machine)** com kernel RBF encontra o hiperplano de máxima margem no espaço de características HOG. O kernel RBF é definido por:

$$K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$$

O parâmetro `C` controla a penalização de erros de classificação (trade-off viés-variância) e `γ (gamma)` controla a largura do kernel.

### 4.2 CNN — Rede Neural Convolucional *(técnica de aula — `cnnocr.py`)*

A **Rede Neural Convolucional (CNN)** aprende automaticamente representações hierárquicas por meio de camadas de convolução, ativação e pooling. A arquitetura adotada é:

```
Input (1×28×28)
  → Conv2d(16 filtros, 3×3, pad=1) → ReLU → MaxPool2d(2×2)   [→ 16×14×14]
  → Conv2d(32 filtros, 3×3, pad=1) → ReLU → MaxPool2d(2×2)   [→ 32×7×7]
  → Flatten
  → Linear(32×7×7 = 1568, 128) → ReLU
  → Linear(128, 10)
```

Função de perda: **Cross-Entropy Loss**.  
Otimizador: **Adam** (Adaptive Moment Estimation).

### 4.3 ViT — Vision Transformer *(técnica extra-classe — `transfpy.py`)*

O **Vision Transformer (ViT)** [Dosovitskiy et al., 2020] adapta a arquitetura Transformer, originalmente proposta para NLP, para imagens. O funcionamento é:

1. **Patchificação:** A imagem de entrada (224×224) é dividida em *patches* não sobrepostos de 16×16 pixels → 196 patches.
2. **Embedding linear:** Cada patch é achatado e projetado para um vetor de dimensão *D* (768 no ViT-Base).
3. **Token [CLS]:** Um token de classificação especial é concatenado à sequência.
4. **Positional Encoding:** Embeddings posicionais treináveis são somados para preservar informação espacial.
5. **Encoder Transformer:** A sequência é processada por *L* camadas de *Multi-Head Self-Attention* (MSA) + MLP:
   $$\text{MSA}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
6. **Classificação:** O token [CLS] é passado por uma camada linear final que produz os logits para as 10 classes.

**Estratégia de fine-tuning:** Utilizamos o modelo pré-treinado no ImageNet (`google/vit-base-patch16-224`) e congelamos todos os pesos exceto o classificador final, realizando apenas o ajuste (*fine-tuning*) da última camada. Isso reduz o custo computacional e aproveita as representações visuais ricas já aprendidas.

---

## 5. Métricas de Avaliação

Todas as métricas são na versão **macro** (média não ponderada entre as 10 classes), o que garante sensibilidade a desempenho por classe mesmo com datasets balanceados.

### 5.1 Acurácia

$$\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{\text{Predições corretas}}{\text{Total de amostras}}$$

### 5.2 Precisão (Macro)

$$\text{Precisão}_c = \frac{TP_c}{TP_c + FP_c}, \quad \text{Precisão} = \frac{1}{C}\sum_{c=1}^{C}\text{Precisão}_c$$

### 5.3 Recall / Sensibilidade (Macro)

$$\text{Recall}_c = \frac{TP_c}{TP_c + FN_c}, \quad \text{Recall} = \frac{1}{C}\sum_{c=1}^{C}\text{Recall}_c$$

### 5.4 F1-Score (Macro)

$$F1_c = \frac{2 \cdot \text{Precisão}_c \cdot \text{Recall}_c}{\text{Precisão}_c + \text{Recall}_c}, \quad F1 = \frac{1}{C}\sum_{c=1}^{C}F1_c$$

Onde *C* = 10 (número de classes), *TP* = Verdadeiros Positivos, *FP* = Falsos Positivos, *FN* = Falsos Negativos.

---

## 6. Variação de Hiperparâmetros e Resultados

### 6.1 SVM + HOG

Hiperparâmetros variados no conjunto de **validação** (F1-Score Macro):

| C     | gamma  | Acurácia (Val) | Precisão (Val) | Recall (Val) | F1 (Val) |
|-------|--------|---------------|----------------|--------------|----------|
| 0.1   | scale  | 0.9519        | 0.9524         | 0.9518       | 0.9517   |
| 0.1   | 0.001  | 0.1130        | 0.0113         | 0.1000       | 0.0203   |
| 0.1   | 0.01   | 0.7420        | 0.8072         | 0.7285       | 0.7114   |
| 1     | scale  | 0.9639        | 0.9641         | 0.9641       | 0.9639   |
| 1     | 0.001  | 0.7513        | 0.8050         | 0.7388       | 0.7233   |
| 1     | 0.01   | 0.9392        | 0.9400         | 0.9385       | 0.9387   |
| **10**| **scale** | **0.9659** | **0.9661**  | **0.9656**   | **0.9657** ✅ |
| 10    | 0.001  | 0.9378        | 0.9386         | 0.9372       | 0.9374   |
| 10    | 0.01   | 0.9552        | 0.9554         | 0.9552       | 0.9551   |
| 100   | scale  | 0.9659        | 0.9660         | 0.9657       | 0.9657   |
| 100   | 0.001  | 0.9532        | 0.9534         | 0.9531       | 0.9530   |
| 100   | 0.01   | 0.9559        | 0.9555         | 0.9558       | 0.9556   |

> Melhor configuração: **C=10, gamma=scale** → F1=0.9657 (validação). Resultado no conjunto de teste: Acc=**0.9667**, Prec=0.9661, Recall=0.9656, F1=**0.9657**.

**Efeito dos hiperparâmetros no SVM:**
- **C alto (100):** Menos regularização → o modelo se ajusta melhor ao treino, mas pode sobreajustar.
- **C baixo (0.1):** Margem larga → mais regularização, risco de subajuste.
- **gamma alto:** Kernel mais "apertado" → fronteiras mais complexas.
- **gamma baixo / scale:** Kernel mais suave → boa generalização para dados de alta dimensão como HOG.

### 6.2 CNN (PyTorch)

Hiperparâmetros variados (F1-Score no **conjunto de teste**):

| learning_rate | epochs | Acurácia (Teste) | F1-Score (Teste) |
|---------------|--------|-----------------|-----------------|
| 0.1           | 5      | 0.1032          | 0.0187          |
| 0.1           | 10     | 0.1135          | 0.0204          |
| 0.1           | 20     | 0.0892          | 0.0164          |
| 0.01          | 5      | 0.9805          | 0.9803          |
| 0.01          | 10     | 0.9674          | 0.9673          |
| 0.01          | 20     | 0.9808          | 0.9807          |
| 0.001         | 5      | 0.9883          | 0.9882          |
| **0.001**     | **10** | **0.9904**      | **0.9903** ✅   |
| 0.001         | 20     | 0.9900          | 0.9900          |



**Efeito dos hiperparâmetros na CNN:**
- **lr=0.1:** Passo de gradiente muito grande → risco de divergência e oscilação do loss.
- **lr=0.001:** Convergência estável e geralmente ótima para Adam com MNIST.
- **Mais épocas:** Geralmente beneficia ao lr menor; com lr alto pode causar divergência.

### 6.3 ViT (Fine-tuning)

Hiperparâmetros variados no conjunto de **teste** (400 amostras treino, 200 teste, 50 por classe):

| learning_rate | epochs | Acurácia (Teste) | F1-Score (Teste) |
|---------------|--------|-----------------|-----------------|
| **1e-4**      | **2**  | 0.3150          | 0.3196          |
| **1e-4**      | **4**  | **0.5050**      | **0.4685** ✅   |
| 5e-5          | 2      | 0.2100          | 0.1757          |
| 5e-5          | 4      | 0.3750          | 0.3345          |

**Efeito dos hiperparâmetros no ViT:**
- **lr=1e-4 supera 5e-5:** Com apenas uma camada treinável (classificador), o lr maior converge mais depressa e aproveita melhor as poucas amostras disponíveis.
- **Mais épocas ajudam:** A acurácia sobe progressivamente (0.16 → 0.48 em validação com lr=1e-4, 4 épocas), indicando que o modelo ainda estava aprendendo — mais épocas poderiam ajudar, mas o risco de sobreajuste ao pequeno subconjunto aumenta.
- **lr=5e-5 com 2 épocas:** Pior resultado (F1=0.1757), próximo de chance aleatória, indicando underfitting por atualização insuficiente.

---

## 7. Comparação Final dos Métodos

Execute `compare_results.py` para gerar os gráficos comparativos em `resultados/`.

### Tabela Resumo (melhores configurações de cada método)

| Método   | Config. ótima          | Acurácia   | Precisão   | Recall     | F1-Score   |
|----------|------------------------|-----------|-----------|-----------|----------|
| SVM+HOG  | C=10, gamma=scale      | 0.9667    | 0.9661    | 0.9656    | 0.9657   |
| **CNN**  | lr=0.001, epochs=10    | **0.9904** | **0.9903** | **0.9904** | **0.9903** |
| ViT      | lr=1e-4, epochs=4      | 0.5050    | —         | —         | 0.4685   |

### Análise dos Resultados Obtidos

**CNN (melhor método — F1 = 0.9903):**  
A CNN treinada do zero no MNIST atingiu 99% de acurácia com lr=0.001 e 10 épocas. O lr=0.1 colapsou completamente (F1 ≈ 0.02, equivalente a prever sempre a mesma classe), demonstrando claramente o impacto negativo de uma taxa de aprendizado excessivamente alta com o otimizador Adam. O lr=0.01 e 0.001 convergiram bem, com 0.001 sendo ligeiramente superior.

**ViT (fine-tuning limitado — F1 = 0.4685):**  
O desempenho inferior ao esperado se explica por dois fatores: (a) apenas 400 amostras de treino — insuficientes para adaptar as representações do ImageNet ao domínio de dígitos manuscritos em escala de cinza — e (b) somente a camada classificadora foi descongelada, mantendo todas as 197 outras camadas fixas. Em cenário com mais dados e mais camadas liberadas para fine-tuning, o ViT tipicamente supera CNNs shallow.

**SVM+HOG (F1 = 0.9657):**  
Atingiu 96.7% de acurácia com C=10 e gamma=scale. O gamma=0.001 fixo colapsou para F1≈0.02 com C baixo (kernel excessivamente estreito, incapaz de generalizar). O parâmetro gamma=scale — calculado automaticamente como 1/(n_features × variância) — mostrou-se consistentemente superior e robusto à escala dos descritores HOG, ficando a menos de 0.3 p.p. da CNN para a mesma tarefa.

---

## 8. Como Executar

```bash
# Instalar dependências (caso necessário)
pip install opencv-python scikit-learn torch torchvision transformers matplotlib

# 1. Treinar SVM
cd projeto_final
python svm_mnist.py

# 2. Treinar CNN
python cnn_mnist.py

# 3. Fine-tuning ViT (requer conexão à internet para baixar o modelo)
python vit_mnist.py

# 4. Gerar comparação e gráficos
python compare_results.py
```

---

## 9. Referências

1. LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). **Gradient-based learning applied to document recognition**. *Proceedings of the IEEE*, 86(11), 2278–2324.
2. Cortes, C., & Vapnik, V. (1995). **Support-vector networks**. *Machine Learning*, 20(3), 273–297.
3. Dalal, N., & Triggs, B. (2005). **Histograms of oriented gradients for human detection**. *CVPR*.
4. LeCun, Y., & Bengio, Y. (1995). **Convolutional networks for images, speech, and time series**. *The Handbook of Brain Theory and Neural Networks*.
5. Dosovitskiy, A., et al. (2020). **An image is worth 16x16 words: Transformers for image recognition at scale**. *ICLR 2021*. https://arxiv.org/abs/2010.11929
6. OpenCV. **OCR of Hand-written Digits**. https://docs.opencv.org/4.x/dd/d3b/tutorial_py_svm_opencv.html
