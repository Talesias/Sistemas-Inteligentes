# Problema 2 — Regressão Logística para Previsão de Doenças Cardíacas

## 1. Dados e pré-processamento

- **Dataset:** UCI Heart Disease, base de Cleveland (`dados/processed.cleveland.data`, arquivo original sem cabeçalho): **303 pacientes e 13 variáveis clínicas**. Os nomes das colunas vêm da documentação da UCI.
- **Binarização do alvo:** o diagnóstico original vai de 0 a 4 (164, 55, 36, 35 e 13 pacientes). Foi binarizado em **0 = sem doença** (164 pacientes) e **1 = com doença** (139 pacientes, valores 1 a 4). As classes ficam **bem balanceadas** (54,1% / 45,9%), então a acurácia é uma métrica confiável neste problema.
- **Valores ausentes:** há 6 pacientes com valor ausente (marcados com `?`): 4 em `ca` e 2 em `thal`. Foram imputados com a **moda** (valor mais frequente), calculada **apenas no treino** (`ca` = 0 e `thal` = 3), pois são variáveis de contagem/categoria. Descartar as 6 linhas seria uma alternativa, mas custaria 2% de uma base já pequena.
- **Divisão treino/teste:** o enunciado não fixa a proporção; usei **80% / 20%** com `random_state=42` e **estratificação** pelo alvo: **242 pacientes de treino e 61 de teste**, ambos com 45,9% de doentes.
- **Padronização:** `StandardScaler` (z-score) das 13 variáveis independentes, **ajustado apenas no treino**. A divisão vem antes de imputar e padronizar, para que nada do teste influencie o treino (*data leakage*).

## 2. Tarefa 1 — Treinamento e convergência

Modelo: `LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)`, exatamente com os parâmetros pedidos (regularização L2 com C = 1,0, padrão do scikit-learn).

> **Número de iterações até a convergência (`n_iter_`): 5.** O limite era `max_iter = 1000`, então o modelo convergiu com folga.

Observação: o `liblinear` não usa o gradiente descendente dos slides. Para regressão logística com L2, ele usa um método de segunda ordem (Newton com região de confiança), que converge em poucas iterações, por isso o número é tão pequeno.

Coeficientes (variáveis padronizadas; `exp(coef)` é a razão de chances para +1 desvio padrão da variável, mantendo as demais fixas):

| Variável | Coeficiente | Razão de chances |
|---|---:|---:|
| ca (nº de vasos obstruídos) | +1,109 | 3,03 |
| thal (cintilografia de tálio) | +0,678 | 1,97 |
| sex (1 = masculino) | +0,656 | 1,93 |
| cp (tipo de dor no peito) | +0,543 | 1,72 |
| exang (angina por exercício) | +0,384 | 1,47 |
| slope | +0,354 | 1,42 |
| thalach (FC máxima) | −0,348 | 0,71 |
| trestbps | +0,314 | 1,37 |
| fbs | −0,221 | 0,80 |
| restecg | +0,218 | 1,24 |
| chol | +0,215 | 1,24 |
| oldpeak | +0,150 | 1,16 |
| age | −0,104 | 0,90 |


O número de vasos coronários com obstrução (`ca`) é o fator de maior peso, seguido de `thal`, `sex` e `cp`. Quanto maior a frequência cardíaca máxima atingida (`thalach`), menor o risco. A idade tem coeficiente próximo de zero e negativo, o que não deve ser lido como "idade protege": isoladamente ela se correlaciona positivamente com a doença (r = +0,22), mas também com `ca` (+0,36), `trestbps` (+0,28) e `thalach` (−0,39), e esse efeito é absorvido por essas variáveis no modelo múltiplo. Os coeficientes de `cp`, `restecg`, `slope` e `thal` devem ser lidos com cautela: são códigos de categorias tratados como números (ver análise C2).

## 3. Tarefa 2 — Avaliação no conjunto de teste

Limiar de decisão padrão de 0,5; classe positiva = com doença.

| Métrica | Valor |
|---|---:|
| **Acurácia** | **0,8689** |
| **Precisão** | **0,8125** |
| **Recall** | **0,9286** |
| **F1-score** | **0,8667** |
| **AUC-ROC** | **0,9513** |

Matriz de confusão (teste, 61 pacientes):

| | Previsto: sem doença | Previsto: com doença |
|---|---:|---:|
| **Real: sem doença** | VN = 27 | FP = 6 |
| **Real: com doença** | FN = 2 | VP = 26 |

![Matriz de confusão](figuras/t2_matriz_confusao.png)
![Curva ROC](figuras/t2_curva_roc.png)

**Interpretação.**
- O modelo acerta 86,9% dos pacientes do teste. A **AUC de 0,951** é "excelente" na escala dos slides (0,9 a 1,0): em 95% dos pares paciente doente/saudável, o modelo atribui probabilidade maior ao doente.
- O **recall de 0,929** é o ponto forte: dos 28 pacientes doentes, 26 foram detectados e só 2 passaram despercebidos. Isso é desejável em triagem médica, em que o falso negativo (doente não detectado) é o erro mais grave. O custo é uma precisão menor (0,813): 6 pacientes saudáveis receberam alarme falso.
- As fórmulas das métricas foram conferidas manualmente no código contra a matriz de confusão.

### Cautela sobre a incerteza do resultado

O teste tem apenas **61 pacientes**: um único erro muda a acurácia em 1,6 ponto percentual. Para ter uma ideia mais estável, fiz uma validação cruzada estratificada de 5 folds **só no treino** (análise complementar C1):

| Métrica | Média ± desvio padrão (CV 5-fold) |
|---|---:|
| Acurácia | 0,826 ± 0,017 |
| Precisão | 0,844 ± 0,034 |
| Recall | 0,765 ± 0,055 |
| F1-score | 0,801 ± 0,026 |
| AUC | 0,896 ± 0,015 |

Os valores de teste (acurácia 0,869; AUC 0,951) ficam acima da média da validação cruzada (0,826; 0,896). A diferença é compatível com a incerteza de um teste tão pequeno e sugere que o desempenho esperado do modelo é um pouco menor que o do teste: **cerca de 83% de acurácia e AUC próxima de 0,90**. O resultado oficial da tarefa é o do conjunto de teste, mas essa ressalva deve acompanhá-lo.

## 4. Análises complementares

**C2 — One-hot nas variáveis nominais.** `cp`, `restecg`, `slope` e `thal` são códigos de categorias sem ordem numérica real (por exemplo, `thal` = 3, 6, 7). O enunciado pede padronizar as variáveis independentes, e a versão principal segue isso. Como verificação, treinei também uma variante com one-hot nessas quatro variáveis (e padronização das demais): convergiu em 6 iterações e obteve acurácia 0,869, precisão 0,813, recall 0,929, F1 0,867 e **AUC 0,958**. O resultado é praticamente idêntico, então a escolha de codificação não altera as conclusões.

**C3 — Efeito do limiar de decisão** (ilustrativo; usa o teste só para mostrar o compromisso, e um limiar real deveria ser escolhido com dados de validação):

| Limiar | Acurácia | Precisão | Recall | FP | FN |
|---:|---:|---:|---:|---:|---:|
| 0,3 | 0,803 | 0,700 | 1,000 | 12 | 0 |
| 0,4 | 0,869 | 0,812 | 0,929 | 6 | 2 |
| 0,5 | 0,869 | 0,812 | 0,929 | 6 | 2 |
| 0,6 | 0,869 | 0,833 | 0,893 | 5 | 3 |
| 0,7 | 0,902 | 0,893 | 0,893 | 3 | 3 |

Baixar o limiar para 0,3 detecta todos os doentes (recall = 1,0), ao custo de dobrar os falsos positivos (12). Em um cenário de triagem, em que o falso negativo é mais caro, um limiar abaixo de 0,5 seria uma opção a considerar.

