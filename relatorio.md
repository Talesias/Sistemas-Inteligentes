# Problema 3 — Classificação de Qualidade de Vinhos Utilizando k-NN


## 1. Tarefa 1 — Preparação dos dados

- **Dataset:** `winequality-red.csv` (UCI Wine Quality): **1.599 vinhos tintos**, 11 atributos físico-químicos e a nota `quality`. Não há valores ausentes. Há **240 linhas duplicadas** (15%), que foram mantidas (ver seções 3 e 4 sobre o efeito delas).
- **Recodificação de `quality`** em três classes: nota ≤ 5 → **ruim**, 6–7 → **médio**, ≥ 8 → **bom**.

| Classe | Amostras | Proporção |
|---|---:|---:|
| ruim | 744 | 46,5% |
| médio | 837 | 52,3% |
| bom | **18** | **1,1%** |


**Atenção ao desbalanceamento.** A classe "bom" tem só 18 vinhos, e um classificador que sempre responde "médio" já acerta 52,3%. Isso afeta toda a análise (ver seção 4).

- **Normalização:** Z-score (`StandardScaler`, média 0 e desvio 1). Ele foi colocado dentro de um `Pipeline`, de modo que a média e o desvio são calculados **apenas com os dados de treino** de cada etapa (cada fold da validação cruzada e o ajuste final). Normalizar o dataset inteiro antes de dividir faria informação do teste vazar para o treino.

**Por que a normalização é obrigatória para o k-NN?** O k-NN decide pelos vizinhos mais próximos segundo a distância euclidiana, que soma os quadrados das diferenças de todos os atributos. Sem normalização, os atributos de maior escala dominam essa soma, não porque sejam mais informativos, mas porque seus números são maiores. Neste dataset, os desvios padrão vão de 32,9 (`total sulfur dioxide`) a 0,0019 (`density`), uma razão de **17.430 vezes**. Sem normalizar, só o `total sulfur dioxide` responde por **91,1%** da variância total e o `free sulfur dioxide` por 8,4%; os outros nove atributos somam menos de 0,5%. Na prática, o modelo ignoraria acidez, álcool, sulfatos, pH e os demais. O experimento confirma (validação cruzada agrupada, 5 folds):

| k | Com Z-score | Sem normalização |
|---:|---:|---:|
| 1 | 0,653 | 0,585 |
| 7 | 0,694 | 0,593 |
| 101 | **0,722** | 0,619 |

## 2. Tarefa 2 — Divisão treino/teste estratificada (80% / 20%)

`train_test_split(test_size=0.20, random_state=42, stratify=y)` → **1.279 amostras de treino e 320 de teste**, com as proporções das classes preservadas:

| Classe | Treino | Teste | % treino | % teste |
|---|---:|---:|---:|---:|
| ruim | 595 | 149 | 46,52 | 46,56 |
| médio | 670 | 167 | 52,38 | 52,19 |
| bom | 14 | 4 | 1,09 | 1,25 |

A classe "bom" tem **4 amostras no teste**; qualquer métrica sobre ela é estatisticamente frágil.

## 3. Tarefa 3 — Treinamento e escolha de k

Testei k = 1, 3, 5, 7, 11 (os valores sugeridos) e também 15, 21, 31, 51, 101, 201, 401 e 801, para mostrar o que acontece com k muito alto. Validação cruzada de 5 folds **sobre o conjunto de treino**, com acurácia como critério.

**Cuidado com as duplicatas.** Numa validação cruzada comum, uma linha duplicada pode cair no fold de validação e a cópia no de treino; o k-NN a encontra a distância zero e "acerta por memorização", o que infla a acurácia de k pequeno. Por isso calculei duas curvas sobre os mesmos 1.279 exemplos de treino:

- **(A) comum** (`StratifiedKFold`), como pede o enunciado, mostrada para comparação;
- **(B) agrupada** (`StratifiedGroupKFold`), em que linhas idênticas ficam sempre no mesmo fold. **A escolha de k usa esta curva**, que não tem vazamento.

| k | Acurácia de validação (B) | Treino (B) | Validação comum (A) |
|---:|---:|---:|---:|
| 1 | 0,653 ± 0,021 | 1,000 | 0,726 |
| 3 | 0,673 ± 0,038 | 0,854 | 0,709 |
| 5 | 0,690 ± 0,025 | 0,799 | 0,713 |
| 7 | 0,694 ± 0,030 | 0,786 | 0,699 |
| 11 | 0,701 ± 0,043 | 0,765 | 0,714 |
| 15 | 0,701 ± 0,053 | 0,762 | 0,722 |
| 21 | 0,698 ± 0,047 | 0,743 | 0,712 |
| 31 | 0,696 ± 0,036 | 0,729 | 0,714 |
| 51 | 0,715 ± 0,021 | 0,732 | 0,725 |
| **101** | **0,722 ± 0,032** | 0,727 | 0,727 |
| 201 | 0,706 ± 0,025 | 0,712 | 0,702 |
| 401 | 0,707 ± 0,030 | 0,706 | 0,705 |
| 801 | 0,647 ± 0,026 | 0,645 | 0,639 |


**k escolhido: 101** (maior acurácia média de validação, 0,722). Entre os cinco valores sugeridos no enunciado, o melhor é k = 11 (0,701).

**Ressalvas sobre essa escolha.** As diferenças entre os valores de k de 7 a 401 ficam dentro de um desvio padrão (≈ 0,03) do melhor, então esses valores são **praticamente equivalentes**; k = 101 é o primeiro lugar pelo critério definido, mas não uma vitória estatisticamente clara. O que a curva mostra com clareza é o formato geral: acurácia baixa em k = 1, subida até um patamar, e queda em k muito alto.

**O que acontece com k muito baixo e muito alto?**
- **k muito baixo (k = 1) → sobreajuste (variância alta).** A acurácia de treino é 1,000 (cada ponto é o seu próprio vizinho), mas a de validação é só 0,653: o modelo memoriza o treino, inclusive o ruído, e a fronteira fica muito irregular. Isso pesa mais aqui porque a nota de qualidade é subjetiva (sommeliers) e os rótulos são ruidosos. Na curva (A), k = 1 parece bom (0,726) apenas por causa das duplicatas: sem elas, cai 7 pontos percentuais.
- **k muito alto (k = 801) → subajuste (viés alto).** A vizinhança passa a incluir vinhos de todas as classes, o modelo suaviza demais, e a acurácia cai para 0,647. Treino (0,645) e validação (0,647) praticamente se igualam: o modelo perdeu a capacidade de capturar o padrão. No limite (k = número de amostras), o modelo sempre prevê a classe majoritária e acerta só 52,3%.
- **Valores intermediários** equilibram viés e variância e dão o melhor desempenho (≈ 0,70–0,72).

Mesmo no melhor ponto, a acurácia de validação (≈ 72%) supera a referência de 52,3% por cerca de 20 pontos, mas não é alta: classificar a qualidade a partir só da química é difícil, e as classes "ruim" e "médio" se sobrepõem bastante.

## 4. Tarefa 4 — Avaliação no conjunto de teste (k = 101)

**Acurácia no teste: 0,7344.**

Matriz de confusão (linhas = real, colunas = previsto):

| | Prev. ruim | Prev. médio | Prev. bom |
|---|---:|---:|---:|
| **Real ruim** (149) | 111 | 38 | 0 |
| **Real médio** (167) | 43 | 124 | 0 |
| **Real bom** (4) | 0 | 4 | 0 |


| Classe | Precisão | Recall | F1-score | Amostras |
|---|---:|---:|---:|---:|
| ruim | 0,7208 | 0,7450 | 0,7327 | 149 |
| médio | 0,7470 | 0,7425 | 0,7447 | 167 |
| bom | 0,0000 | 0,0000 | 0,0000 | 4 |
| média macro | 0,4893 | 0,4958 | 0,4925 | 320 |
| média ponderada | 0,7254 | 0,7344 | 0,7298 | 320 |

**Entre quais classes o modelo erra com mais frequência?** Entre **"ruim" e "médio"**: dos 85 erros, **81 (95,3%)** são confusões entre essas duas classes (43 "médio" previstos como "ruim" e 38 "ruim" previstos como "médio"). É coerente com a natureza do problema: as notas 5 e 6, que ficam na fronteira entre as duas classes, concentram a maior parte dos vinhos (681 e 638), e vinhos de notas vizinhas têm perfis químicos muito parecidos.

**A classe "bom" nunca é prevista.** Os 4 vinhos "bons" do teste foram classificados como "médio" (recall = 0). Com k = 101 e apenas 14 vinhos "bons" no treino, essa classe nunca consegue ganhar a votação em nenhuma vizinhança. A acurácia (0,73) esconde esse problema: a F1 macro (0,49), que trata as três classes com o mesmo peso, o revela, enquanto a ponderada (0,73) o mascara, como mostram os slides de métricas. Com só 4 amostras de teste não dá para medir o desempenho nessa classe com precisão, mas a limitação é real. Algumas saídas possíveis (não testadas neste projeto): juntar "bom" a "médio", reamostrar as classes ou usar pesos por classe.

**Efeito das duplicatas no teste.** 71 das 320 amostras de teste (22,2%) têm uma cópia exata no treino. A acurácia nelas é 0,789, contra **0,719** nas amostras sem cópia. Portanto, a estimativa de desempenho realista para vinhos novos é de cerca de **72%**, um pouco abaixo do 73,4% reportado.

