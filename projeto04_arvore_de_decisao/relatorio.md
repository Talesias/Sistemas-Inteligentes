# Problema 4 — Classificação de Risco de Crédito Utilizando Árvores de Decisão

**Convenção:** a classe positiva é o **mau pagador** (evento que o banco quer detectar). O rótulo original 1 (bom) virou 0 e o 2 (mau) virou 1.

## 1. Tarefa 1 — Preparação dos dados

- **Dataset:** `german.data` (UCI Statlog German Credit): **1.000 solicitações**, 20 atributos e a classe. Sem valores ausentes e sem linhas duplicadas.
- **Atributos numéricos (7):** `duracao_meses`, `valor_credito`, `taxa_parcela`, `tempo_residencia`, `idade`, `num_creditos`, `num_dependentes`.
- **Atributos categóricos (13), codificados como A11, A12, …:** `status_conta`, `historico_credito`, `finalidade`, `poupanca`, `tempo_emprego`, `estado_civil_sexo`, `outros_devedores`, `propriedade`, `outros_planos`, `moradia`, `tipo_emprego`, `telefone`, `trab_estrangeiro`. O significado dos códigos foi transcrito do `german.doc` oficial e está no dicionário `DESCRICAO` do código.
- **Codificação:** **sim, é necessária.** O `DecisionTreeClassifier` do scikit-learn só aceita números, e os códigos A11, A12… são rótulos de categoria, não quantidades. Usei **one-hot encoding** nos 13 atributos categóricos (e não ordinal), porque vários não têm uma ordem real (por exemplo, `finalidade`) e, mesmo nos que têm, existe uma categoria "desconhecido/sem conta" (A14, A65) que quebraria a ordem. O resultado são 54 colunas binárias + 7 numéricas = **61 variáveis de entrada**. O efeito colateral é que cada corte da árvore testa uma única categoria por vez.
- **Balanceamento:** 700 bons (70%) e 300 maus (30%).



**Impacto do desbalanceamento sobre a acurácia.** Um classificador que **aprova todos os clientes** acerta 70% sem aprender nada, mas tem recall de 0% para o mau pagador e, pela matriz de custo do `german.doc` (aprovar um mau custa 5; recusar um bom custa 1), custo de 1.500 no dataset inteiro. Portanto, **70% é a acurácia mínima a superar** e a acurácia sozinha não mede o que importa aqui. Devem-se olhar o recall e a precisão da classe "mau" e o custo.

## 2. Tarefa 2 — Por que a normalização não é necessária para a árvore

Cada nó da árvore testa uma condição do tipo `x_j ≤ limiar` e escolhe o par (atributo, limiar) que mais reduz a impureza. A normalização (z-score ou min-max) é uma **transformação monótona crescente**: ela muda os valores de `x_j`, mas não a **ordem** das amostras ao longo do atributo. Logo, as mesmas partições continuam possíveis, com o mesmo ganho de pureza; só o valor numérico do limiar muda. Além disso, a árvore **nunca calcula distâncias entre atributos**, então a diferença de escala (por exemplo, `valor_credito` na casa dos milhares contra `taxa_parcela` de 1 a 4) não distorce nada. É o contrário do k-NN do Problema 3.

**Verificação empírica:** treinei duas árvores (profundidade 5, gini), uma com as variáveis numéricas padronizadas e outra sem. Resultado: **mesma acurácia no teste (0,645), previsões idênticas em 100% das amostras, mesmo número de nós (53) e os mesmos atributos nos cortes.**

## 3. Tarefa 3 — Divisão treino/teste estratificada (80% / 20%)

`train_test_split(test_size=0.20, random_state=42, stratify=y)` → **800 de treino e 200 de teste**, ambos com 70% de bons e 30% de maus (560/240 no treino e 140/60 no teste).

## 4. Tarefa 4 — Controle de complexidade e critério de divisão

Validação cruzada estratificada de 5 folds **sobre o treino**, para profundidades máximas 2, 3, 5, 7, 10 e sem limite (a árvore livre chega à profundidade real 18, com 149 folhas). Resultados com o critério **Gini**:

| Profundidade | Acurácia treino | Acurácia validação | Desvio | Recall "mau" (val.) |
|---|---:|---:|---:|---:|
| 2 | 0,708 | 0,689 | 0,014 | 0,133 |
| 3 | 0,745 | 0,698 | 0,035 | 0,367 |
| 5 | 0,804 | 0,698 | 0,028 | 0,442 |
| **7** | 0,878 | **0,711** | 0,027 | **0,529** |
| 10 | 0,958 | 0,686 | 0,027 | 0,513 |
| sem limite | 1,000 | 0,679 | 0,029 | 0,488 |


**Profundidade escolhida: 7** (maior acurácia média de validação, 0,711).

**O que acontece nas profundidades muito baixas e muito altas?**
- **Muito baixa (2): subajuste (*underfitting*, viés alto).** Treino (0,708) e validação (0,689) são baixos e quase iguais: a árvore é simples demais para capturar o padrão. Na prática, ela quase só prevê "bom": o recall do mau pagador é de apenas 13%.
- **Muito alta (sem limite, profundidade 18): sobreajuste (*overfitting*, variância alta).** A acurácia de treino chega a **1,000** (a árvore decora cada cliente), mas a de validação **cai para 0,679**: a árvore aprendeu o ruído do treino e não generaliza. O distanciamento entre as curvas de treino e validação cresce com a profundidade.
- **Intermediária (7):** melhor equilíbrio entre viés e variância, com o melhor recall do mau pagador (0,53).

**Ressalva importante.** A acurácia de validação do melhor modelo (0,711) é só **1 ponto acima da referência de 70%** ("aprovar todos"), e as diferenças entre as profundidades 3 a 7 ficam dentro de um desvio padrão (≈ 0,03). A acurácia, por si só, quase não distingue os modelos aqui; o ganho real da árvore aparece no recall do mau pagador (0,53 contra 0 da referência).

**Gini x entropia.** Repeti o experimento com `criterion="entropy"`:

| Profundidade | Val. Gini | Val. Entropia | Diferença |
|---|---:|---:|---:|
| 2 | 0,689 | 0,694 | +0,005 |
| 3 | 0,698 | 0,699 | +0,001 |
| 5 | 0,698 | 0,719 | +0,021 |
| 7 | 0,711 | 0,720 | +0,009 |
| 10 | 0,686 | 0,690 | +0,004 |
| sem limite | 0,679 | 0,690 | +0,011 |

**Não houve diferença relevante.** A maior diferença (0,021, na profundidade 5) é menor que o desvio padrão da validação (≈ 0,027), e **ambos os critérios escolhem a profundidade 7**. A entropia fica ligeiramente acima em todas as profundidades, mas isso está dentro do ruído. Mantive o Gini (padrão) como critério do modelo principal.

## 5. Tarefa 5 — Interpretação do modelo (Gini, profundidade 7)

A árvore final tem profundidade 7, **59 folhas e 117 nós**. 


**Os três atributos mais determinantes** (importância somada das dummies de cada atributo original):

| Atributo | Importância |
|---|---:|
| **Conta corrente (`status_conta`)** | **18,8%** |
| **Duração do empréstimo (`duracao_meses`)** | **14,1%** |
| **Valor do crédito (`valor_credito`)** | **11,4%** |
| Poupança | 8,0% |
| Propriedade | 7,2% |
| Finalidade | 7,1% |

Os três juntos respondem por **44,3%** da importância total. A coluna individual mais importante é "sem conta corrente" (A14), com 18,6%. No primeiro corte da árvore, quem **não** tem conta corrente (316 clientes de treino) tem só 12% de maus pagadores, contra 42% entre quem tem conta (484 clientes), e este foi o corte que mais separou as classes.


**Regras do caminho da raiz até a folha com maior número de maus pagadores** (folha 67, com 44 maus entre 97 clientes de treino):

> **SE** o cliente **tem conta corrente** (não é "sem conta corrente", A14)
> **E** a duração do empréstimo está **entre 15,5 e 47,5 meses**
> **E** a poupança é **< 100 DM** (A61)
> **E** o valor do crédito está **entre 2.321 e 8.015,5 DM**
> **ENTÃO** a árvore prevê **bom pagador**, mas é uma zona de risco: nessa folha, 44 dos 97 clientes (45,4%) eram maus pagadores.

Observação: como essa folha é grande e ainda tem maioria de bons (53 contra 44), ela prevê "bom"; é a folha que concentra **mais maus pagadores em número absoluto**, não a de maior proporção. Para completar, a folha que **prevê "mau"** com mais maus pagadores (folha 65, 24 maus entre 38 = 63,2%) tem a regra: tem conta corrente **E** duração entre 15,5 e 47,5 meses **E** poupança < 100 DM **E** valor ≤ 2.321 DM **E** sem outros planos de financiamento (A143). Ambas as regras foram conferidas diretamente nos dados (97/44 e 38/24).

## 6. Tarefa 6 — Avaliação no conjunto de teste

Acurácia: **0,680**. Matriz de confusão (200 clientes, linhas = real):

| | Previsto: bom | Previsto: mau |
|---|---:|---:|
| **Real: bom** (140) | 110 | FP = 30 |
| **Real: mau** (60) | **FN = 34** | 26 |


| Classe | Precisão | Recall | F1-score | Amostras |
|---|---:|---:|---:|---:|
| bom | 0,7639 | 0,7857 | 0,7746 | 140 |
| mau | 0,4643 | 0,4333 | 0,4483 | 60 |
| média macro | 0,6141 | 0,6095 | 0,6115 | 200 |
| média ponderada | 0,6740 | 0,6800 | 0,6767 | 200 |

**Atenção:** a acurácia do teste (0,680) fica **abaixo da referência de 70%** de "aprovar todos". Ela não deve ser lida isoladamente: o modelo detecta 43% dos maus pagadores (a referência detecta 0%), e o custo pela matriz do `german.doc` cai de 300 (aprovar todos) para **200**. Como o teste tem só 200 clientes (60 maus), os números têm incerteza considerável.

**Qual tipo de erro é mais frequente e por que é o mais grave?** O erro mais frequente é o **falso negativo do mau pagador: aprovar um cliente que é mau pagador (34 casos, contra 30 falsos positivos)**, e o modelo deixa passar 57% dos maus. É o mais grave porque aprovar um inadimplente gera **prejuízo direto** (perda do principal), enquanto recusar um bom pagador (falso positivo) significa apenas perda de receita e de oportunidade. A própria matriz de custo do conjunto de dados (`german.doc`) atribui custo **5** ao primeiro erro e **1** ao segundo. Pelo custo, os 34 FN pesam 170 e os 30 FP pesam 30: 85% do custo vem dos maus pagadores aprovados.

**Que ajuste faria no treinamento para reduzi-lo?** Usar **pesos por classe** (`class_weight`), de modo que a árvore penalize mais os erros na classe "mau". Comparei três opções por validação cruzada **no treino** (sem olhar o teste), usando o custo da matriz 5:1:

| `class_weight` | Acurácia | Recall "mau" | Precisão "mau" | Custo por cliente |
|---|---:|---:|---:|---:|
| nenhum (padrão) | 0,711 | 0,529 | 0,518 | 0,854 |
| `balanced` | 0,661 | 0,663 | 0,456 | 0,744 |
| **custo 5:1** (`{bom: 1, mau: 5}`) | 0,641 | **0,763** | 0,443 | **0,644** |

A opção de menor custo foi `custo 5:1`. Resultado **no teste**:

| Modelo | Acurácia | Recall "mau" | Precisão "mau" | FN | FP | Custo total |
|---|---:|---:|---:|---:|---:|---:|
| Padrão | 0,680 | 0,433 | 0,464 | 34 | 30 | 200 |
| **`class_weight` 5:1** | 0,585 | **0,767** | 0,400 | **14** | 69 | **139** |

O ajuste reduz os maus pagadores aprovados de **34 para 14** (o recall sobe de 43% para 77%) e o custo total cai **31%** (de 200 para 139). O preço é **recusar mais bons pagadores** (FP de 30 para 69), com queda da acurácia para 0,585. Isso é um compromisso esperado e coerente com a assimetria de custos; se a política do banco fosse outra (por exemplo, custo 2:1), o peso ideal seria diferente. Outras alternativas, não testadas neste projeto: reduzir o limiar de decisão sobre `predict_proba` ou reajustar a profundidade já com os pesos.
