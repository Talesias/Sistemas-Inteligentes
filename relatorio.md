# Problema 5 — Segmentação de Clientes Utilizando k-Means

**Dados:** `Wholesale customers data.csv` (UCI Wholesale Customers): **440 clientes**, sem valores ausentes nem duplicatas; 298 do canal Horeca e 142 do Retail; regiões Lisboa (77), Porto (47) e Outras (316). O agrupamento usa **somente os seis gastos** (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen). `Channel` e `Region` foram separados no início e só voltam na validação externa da Tarefa 5.

## 1. Tarefa 1 — Preparação dos dados

### Estatísticas descritivas (escala original)

| Categoria | Média | Mediana | Desvio padrão | Mínimo | Máximo | Assimetria |
|---|---:|---:|---:|---:|---:|---:|
| Fresh | 12.000 | 8.504 | 12.647 | 3 | 112.151 | 2,56 |
| Milk | 5.796 | 3.627 | 7.380 | 55 | 73.498 | 4,05 |
| Grocery | 7.951 | 4.756 | 9.503 | 3 | 92.780 | 3,59 |
| Frozen | 3.072 | 1.526 | 4.855 | 25 | 60.869 | 5,91 |
| Detergents_Paper | 2.881 | 817 | 4.768 | 3 | 40.827 | 3,63 |
| Delicassen | 1.525 | 966 | 2.820 | 3 | 47.943 | 11,15 |

Em todas as categorias a média é bem maior que a mediana e a assimetria é fortemente positiva: distribuições com cauda longa à direita, poucos clientes de gasto muito alto e um máximo que vai de 13 a 50 vezes a mediana (em Fresh, os valores vão de 3 a 112.151).


### Transformação logarítmica e padronização

Apliquei `x' = log(1 + x)` e, em seguida, Z-score (`StandardScaler`, ajustado com os 440 clientes, já que não há conjunto de teste).

| Categoria | Assimetria antes → depois | Outliers (regra de Tukey) antes → depois |
|---|---:|---:|
| Fresh | 2,56 → −1,58 | 20 → 16 |
| Milk | 4,05 → −0,22 | 28 → 4 |
| Grocery | 3,59 → −0,67 | 24 → 2 |
| Frozen | 5,91 → −0,35 | 43 → 10 |
| Detergents_Paper | 3,63 → −0,24 | 30 → 2 |
| Delicassen | 11,15 → −1,09 | 27 → 14 |
| **Total** | | **172 → 48** |

**(a) Por que a transformação logarítmica é recomendável.** Os gastos são positivos, têm cauda longa e variam em ordens de grandeza. O k-Means usa a média (o centroide) e a soma de **quadrados** das distâncias; poucos clientes de gasto extremo puxam os centroides e dominam a inércia. O logaritmo comprime a cauda: diminui a assimetria (em Milk, Grocery, Frozen e Detergents_Paper ela cai para valores entre −0,7 e −0,2) e reduz os pontos atípicos de 172 para 48. Em Fresh e Delicassen a assimetria muda de sinal e continua moderada (−1,58 e −1,09), porque esses gastos têm valores mínimos muito pequenos (3 unidades), que viram a cauda esquerda. A necessidade da transformação aparece no teste prático: **sem o log** (só Z-score nos dados brutos) o k-Means com k = 3 produz grupos de **393, 45 e 2 clientes** (os grupos de 45 e 2 são quase só outliers) e concorda muito pouco com o agrupamento final (ARI = 0,084).

**(b) Por que a padronização é determinante.** O k-Means atribui cada ponto ao centroide mais próximo pela **distância euclidiana**, que soma os quadrados das diferenças de todas as variáveis. Com variáveis em escalas diferentes, a de maior variância domina a distância. Participação de cada categoria na variância total (= no peso da distância):

| Categoria | Dados brutos | Só log | Log + Z-score |
|---|---:|---:|---:|
| Fresh | 44,6% | 20,0% | 16,7% |
| Milk | 15,2% | 10,8% | 16,7% |
| Grocery | 25,2% | 11,4% | 16,7% |
| Frozen | 6,6% | 15,2% | 16,7% |
| Detergents_Paper | 6,3% | 27,0% | 16,7% |
| Delicassen | 2,2% | 15,5% | 16,7% |

Nos dados brutos, Fresh e Grocery sozinhas respondem por 70% da distância e Delicassen por apenas 2%. O log melhora, mas **não iguala** as variâncias (Detergents_Paper passa a pesar 27%); só o Z-score dá o mesmo peso (1/6) a cada categoria. As duas etapas são complementares: o log corrige a forma da distribuição e o Z-score corrige a escala.

### Matriz de correlação entre as seis categorias

| (escala original) | Fresh | Milk | Grocery | Frozen | Det_Paper | Delicassen |
|---|---:|---:|---:|---:|---:|---:|
| Fresh | 1,00 | 0,10 | −0,01 | 0,35 | −0,10 | 0,25 |
| Milk | 0,10 | 1,00 | 0,73 | 0,12 | 0,66 | 0,41 |
| Grocery | −0,01 | 0,73 | 1,00 | −0,04 | **0,93** | 0,21 |
| Frozen | 0,35 | 0,12 | −0,04 | 1,00 | −0,13 | 0,39 |
| Detergents_Paper | −0,10 | 0,66 | 0,93 | −0,13 | 1,00 | 0,07 |
| Delicassen | 0,25 | 0,41 | 0,21 | 0,39 | 0,07 | 1,00 |


**Par Grocery–Detergents_Paper:** correlação de **0,925** (0,797 após o log), a mais alta da matriz. Quem compra muita mercearia também compra muitos detergentes e papel, comportamento típico de varejo. As duas variáveis carregam informação quase redundante e, na distância euclidiana, acabam contando em dobro. Junto com Milk (correlação de 0,66 a 0,73 com ambas), formam um bloco que age como um único "eixo do varejo": ele domina a primeira componente principal (PC1, 44% da variância, cargas ≈ 0,55 nas três). Isso ajuda a explicar por que o agrupamento separa tão bem o perfil de varejo dos demais.

## 2. Tarefa 2 — Ausência de divisão treino/teste

**Por que não se separa um conjunto de teste.** No aprendizado **supervisionado**, cada exemplo tem um rótulo conhecido, e o objetivo é estimar o **erro de generalização**: o quanto o modelo acerta rótulos de dados que nunca viu. O teste serve para medir isso sem viés. No aprendizado **não supervisionado** não há rótulo a prever nem "erro" a medir. O objetivo é descrever a estrutura dos próprios dados, aqui, segmentar *estes* 440 clientes. Não existe um valor correto ao qual comparar o resultado, então reservar clientes para teste não responderia a nenhuma pergunta e só reduziria os dados (440 já é pouco).

**A dificuldade central da validação.** Sem gabarito, como afirmar que uma partição é "boa"? Todo critério tem limites:
- Os critérios **internos** (inércia, silhueta) medem compacidade e separação segundo a própria geometria do algoritmo; a inércia sempre melhora com mais grupos (ver Tarefa 3), e a silhueta favorece grupos esféricos e bem separados.
- O k-Means sempre devolve k grupos, **mesmo que os dados não tenham grupos naturais**: ele particiona um contínuo do mesmo jeito.
- A **validação externa** (ARI contra `Channel`) exige um rótulo de referência, que nem sempre existe e que pode não ser a "verdade" que o agrupamento deveria encontrar.
- Por fim, "bom" depende do uso: um segmento só é útil se a área comercial souber o que fazer com ele.

Por isso a escolha final combina critérios internos, validação externa e interpretabilidade, como feito nas tarefas 3 e 5.

## 3. Tarefa 3 — Escolha do número de grupos

k-Means (`init='k-means++'`, `n_init=10`, `random_state=42`) para k = 2 a 10:

| k | Inércia | Queda relativa Δ% | Silhueta média |
|---:|---:|---:|---:|
| 2 | 1.844,1 | — | **0,290** |
| **3** | 1.553,4 | 15,8% | **0,259** |
| 4 | 1.386,9 | 10,7% | 0,189 |
| 5 | 1.266,1 | 8,7% | 0,192 |
| 6 | 1.174,8 | 7,2% | 0,201 |
| 7 | 1.086,4 | 7,5% | 0,196 |
| 8 | 1.028,5 | 5,3% | 0,185 |
| 9 | 973,1 | 5,4% | 0,197 |
| 10 | 932,0 | 4,2% | 0,191 |

**Escolha: k = 3.**
- **Cotovelo:** a curva de inércia é suave, **sem cotovelo nítido**. A maior queda relativa ocorre ao passar de 2 para 3 grupos (15,8%) e depois ela diminui devagar (10,7%, 8,7%, 7,2%…). Uma medida objetiva (distância à reta que liga o primeiro e o último ponto) tem máximo em k = 5, mas quase igual para k = 3 a 5 (0,14; 0,18; 0,18), ou seja, não discrimina bem.
- **Silhueta:** o máximo é em **k = 2 (0,290)**, seguido de **k = 3 (0,259)**. De k = 4 em diante ela cai para cerca de **0,19 a 0,20**. Isso desencoraja k ≥ 4. Mesmo o melhor valor (0,29) indica estrutura **fraca**: os dados formam mais um contínuo do que grupos bem separados (ver PCA, Tarefa 5).
- **Interpretabilidade comercial:** k = 2 se aproxima bastante da divisão Horeca × Retail (ARI 0,51), que a empresa já conhece (o canal está cadastrado), então acrescenta pouca informação nova. k = 3 revela **três perfis distintos e acionáveis** (seção 6), inclusive um grupo de grandes compradores multicategoria que o canal não captura. k = 4 divide o perfil de restaurante por volume, mas com silhueta de 0,19.
- **Resumo:** k = 3 é o melhor compromisso: silhueta ainda alta (segunda melhor), ganho de inércia relevante em relação a k = 2 e segmentos com ação comercial clara. A alternativa estatisticamente mais forte é k = 2; o ARI de k = 2 também é calculado na Tarefa 5.

**Por que a inércia decresce monotonicamente com k?** Qualquer solução com k centroides pode ser estendida para k + 1 acrescentando um centroide extra sem aumentar a distância de nenhum ponto ao centroide mais próximo; logo o menor valor possível de inércia com k + 1 centroides é, no máximo, o de k. Nas execuções, a inércia foi estritamente decrescente. **Por isso ela não serve sozinha** como critério: sempre prefere mais grupos, sem dizer se o ganho vale a pena. É preciso procurar o ponto de retornos decrescentes (cotovelo) e combinar com outros critérios.

**Caso limite k = n (440).** Cada cliente vira o seu próprio centroide e a **inércia é 0** (verificado: 0,000000). É a partição "perfeita" para o critério e completamente inútil: não resume nada. Nesse caso a silhueta nem é definida (o `scikit-learn` exige 2 ≤ k ≤ n − 1); é o equivalente do sobreajuste em aprendizado supervisionado.

## 4. Tarefa 4 — Inicialização e convergência

10 execuções de cada tipo, com `n_init=1` e sementes 0 a 9. Resultados para o k adotado (k = 3) e, de forma complementar, para k = 8 (com mais grupos há mais mínimos locais):

| | k = 3 `random` | k = 3 `k-means++` | k = 8 `random` | k = 8 `k-means++` |
|---|---:|---:|---:|---:|
| Inércia mínima | 1.553,39 | 1.553,34 | 1.025,93 | 1.024,93 |
| Inércia média | 1.569,21 | 1.564,79 | 1.060,71 | 1.059,66 |
| Inércia máxima | 1.626,55 | 1.582,23 | 1.122,48 | 1.091,55 |
| **Desvio padrão** | **24,11** | **14,55** | **29,67** | **16,49** |
| Amplitude (máx − mín) | 73,16 | 28,89 | 96,55 | 66,61 |
| Execuções a ≤ 0,1% do melhor | 6 | 6 | 1 | 1 |
| Execuções > 1% acima do melhor | 4 | 4 | 8 | 9 |
| Iterações até convergir (média) | 15,4 | 16,2 | 15,2 | 14,9 |


**Comparação da dispersão.** Os dois métodos convergem para **mínimos locais diferentes** (a inércia final varia de uma execução para outra). Com `k-means++` a dispersão é menor: o desvio padrão cai cerca de **40%** em k = 3 (14,6 contra 24,1) e **44%** em k = 8 (16,5 contra 29,7), e o pior resultado também melhora (1.582 contra 1.627 em k = 3; 1.092 contra 1.122 em k = 8). Já a **média quase não muda** (diferença de 0,1% a 0,3%), o número de execuções que chegam perto do melhor é parecido e o número de iterações também (15 a 16), então não houve ganho de velocidade neste conjunto. Como são só 10 execuções por método, o desvio padrão é uma estimativa aproximada. Mesmo com `k-means++`, parte das execuções ficou presa em mínimos piores (4 de 10 em k = 3), por isso o procedimento padrão é **repetir** o algoritmo (`n_init > 1`) e ficar com a menor inércia, como foi feito nas tarefas 3 e 5 (`n_init=10`).

**Qual problema o `k-means++` busca mitigar e como.** O k-Means só encontra um **mínimo local**, que depende dos centroides iniciais. Na inicialização aleatória, é possível sortear dois ou mais centroides iniciais na mesma região densa e nenhum em outra, e o algoritmo converge para uma partição ruim. O `k-means++` escolhe o primeiro centroide ao acaso e cada novo centroide com probabilidade **proporcional ao quadrado da distância ao centroide já escolhido mais próximo** (`D(x)²`). Pontos distantes dos centroides existentes têm mais chance de ser escolhidos, e os centroides iniciais ficam **espalhados**, evitando os piores pontos de partida. Ele não garante o mínimo global, apenas reduz a chance e a gravidade das partidas ruins.

## 5. Tarefa 5 — Interpretação e validação dos grupos (k = 3)

### Centroides e nomes comerciais

Os centroides foram revertidos para a escala original desfazendo a padronização e, depois, o logaritmo (`x = exp(z·σ + μ) − 1`). Como o centroide está em escala logarítmica, o valor revertido corresponde ao gasto **típico** (parecido com uma média geométrica), que fica abaixo da média aritmética quando há cauda longa. Por isso mostro também as médias aritméticas.

**Centroides revertidos (gasto anual típico):**

| Grupo | Clientes | Fresh | Milk | Grocery | Frozen | Det_Paper | Delicassen |
|---|---:|---:|---:|---:|---:|---:|---:|
| **G0** | 80 | 1.095 | 5.635 | 9.851 | 344 | 3.392 | 318 |
| **G1** | 147 | 11.287 | 7.972 | 10.520 | 2.168 | 2.762 | 1.906 |
| **G2** | 213 | 7.869 | 1.529 | 1.985 | 1.976 | 246 | 602 |

**Médias aritméticas por grupo:**

| Grupo | Fresh | Milk | Grocery | Frozen | Det_Paper | Delicassen |
|---|---:|---:|---:|---:|---:|---:|
| G0 | 2.899 | 7.136 | 12.570 | 607 | 5.554 | 783 |
| G1 | 17.043 | 10.560 | 13.334 | 4.133 | 4.987 | 2.847 |
| G2 | 11.939 | 2.006 | 2.502 | 3.266 | 424 | 891 |

**Centroides em escala Z** (desvios padrão acima ou abaixo da média geral, base para os nomes):

| Grupo | Fresh | Milk | Grocery | Frozen | Det_Paper | Delicassen |
|---|---:|---:|---:|---:|---:|---:|
| G0 | −1,18 | +0,48 | +0,68 | −1,14 | +0,78 | −0,70 |
| G1 | +0,41 | +0,80 | +0,74 | +0,30 | +0,66 | +0,68 |
| G2 | +0,16 | −0,73 | −0,76 | +0,22 | −0,75 | −0,21 |


| Grupo | Nome comercial | O que o distingue | Uso comercial |
|---|---|---|---|
| **G0** (80) | **Mercearia, laticínios e limpeza (varejo de bairro)** | Muito abaixo da média em Fresh e Frozen (−1,2 desvio) e acima em Grocery, Detergents_Paper e Milk | Pacotes de mercearia e limpeza; entrega sem cadeia de frio |
| **G1** (147) | **Grandes compradores multicategoria** | Acima da média nas seis categorias (+0,3 a +0,8); maiores gastos médios em cinco das seis (só em Detergents_Paper o G0 é maior) | Contas-chave: limite de crédito maior, atendimento dedicado, contratos de fornecimento completo |
| **G2** (213) | **Perfil restaurante: frescos e congelados, pouca mercearia e limpeza** | Fresh e Frozen na média ou um pouco acima; Milk, Grocery e Detergents_Paper bem abaixo (−0,75) | Entregas frequentes de perecíveis, cadeia de frio, rotas e promoções de frescos e congelados |

### Validação externa

**Grupo × Canal** (contagens; ARI = **0,350**):

| Grupo | Horeca | Retail | % Horeca no grupo |
|---|---:|---:|---:|
| G0 | 32 | 48 | 40,0% |
| G1 | 56 | 91 | 38,1% |
| G2 | **210** | 3 | **98,6%** |

**Grupo × Região:**

| Grupo | Lisboa | Porto | Outras |
|---|---:|---:|---:|
| G0 | 14 (17,5%) | 10 (12,5%) | 56 (70,0%) |
| G1 | 23 (15,6%) | 11 (7,5%) | 113 (76,9%) |
| G2 | 40 (18,8%) | 26 (12,2%) | 147 (69,0%) |
| *Base total* | *17,5%* | *10,7%* | *71,8%* |

**ARI entre os grupos e `Channel`: 0,350** (para referência, com k = 2 o ARI é **0,514**; nesse caso o grupo 0 reúne 244 Horeca e 8 Retail, e o grupo 1 reúne 54 Horeca e 134 Retail). **ARI entre os grupos e `Region`: −0,006.**

**O algoritmo, sem jamais ter visto o canal de venda, foi capaz de reconstruí-lo?** **Em parte.** A associação entre grupos e canal é forte (qui-quadrado = 180, p ≈ 8·10⁻⁴⁰; V de Cramér = 0,64): o grupo G2 é praticamente puro (98,6% Horeca, e reúne 70% de todos os Horeca), e 98% dos clientes Retail (139 de 142) caem nos grupos G0 e G1. Mas a reconstrução está longe de ser perfeita (ARI 0,35 com k = 3 e 0,51 com k = 2, onde 1 seria idêntico): **88 clientes Horeca (30%) compram como varejo ou como grandes compradores** e ficam em G0 e G1. Dois motivos: os canais se sobrepõem no perfil de gasto, e o ARI com k = 3 é penalizado por comparar três grupos com duas classes. O canal é, assim, um bom resumo do comportamento de compra, mas não o único fator.

**Qual conjunto de categorias mais separa Horeca de Retail?** O trio **Detergents_Paper, Grocery e Milk**. Medido pela diferença padronizada de médias (d de Cohen, nos gastos em log + Z-score; positivo = Retail gasta mais):

| Categoria | d de Cohen | Mediana Horeca | Mediana Retail |
|---|---:|---:|---:|
| Detergents_Paper | **+2,26** | 386 | 5.614 |
| Grocery | **+1,89** | 2.684 | 12.390 |
| Milk | **+1,51** | 2.157 | 7.812 |
| Frozen | −0,54 | 2.058 | 1.081 |
| Fresh | −0,37 | 9.582 | 5.994 |
| Delicassen | +0,28 | 821 | 1.350 |

O Retail gasta, na mediana, cerca de 14 vezes mais em Detergents_Paper, 4,6 vezes mais em Grocery e 3,6 vezes mais em Milk. Já Horeca gasta mais em Fresh e Frozen, com separação bem menor. Delicassen quase não distingue os canais.

**A região geográfica se reflete nos agrupamentos?** **Não.** O ARI é praticamente zero (−0,006), o teste qui-quadrado não rejeita a independência (p = 0,49) e o V de Cramér é de apenas 0,06. A composição regional de cada grupo é quase igual à da base inteira (cerca de 70% "Outras", 17% Lisboa, 11% Porto). O perfil de consumo não depende da região.

### Projeção PCA e limitações do k-Means

Projeção PCA dos dados padronizados em 2 dimensões (PC1 = 44,1% e PC2 = 27,2% da variância; total 71,3%). PC1 é dominado por Milk, Grocery e Detergents_Paper (cargas 0,54 a 0,57), o "eixo do varejo". PC2 é dominado por Fresh, Frozen e Delicassen (0,53 a 0,59).


O gráfico mostra uma **nuvem contínua, sem vazios entre os grupos**: os agrupamentos são "fatias" dessa nuvem. Duas limitações do k-Means aparecem:

1. **Tendência a grupos de geometria esférica e de tamanhos semelhantes.** O k-Means separa os grupos por fronteiras retas (equidistantes entre dois centroides) e representa cada grupo por um "bolo" compacto em torno da média. Como a nuvem é contínua, a fronteira entre os grupos é, em boa parte, arbitrária: ela corta uma região densa, e o canal real (painel da direita) mostra os dois canais sobrepostos no centro. A silhueta média é de apenas 0,26, e o grupo G0 é o pior ajustado (silhueta 0,12, distância média ao centroide de 2,07 contra 1,57 e 1,65 dos outros); 23 clientes (5,2%) têm silhueta negativa, ou seja, provavelmente estariam melhor no grupo vizinho. Neste conjunto, os tamanhos acabaram desiguais (80, 147 e 213), mas a restrição à forma compacta continua valendo: o algoritmo não consegue representar grupos alongados, curvos ou de densidades muito diferentes.
2. **Sensibilidade a outliers.** Como o centroide é uma média, pontos extremos o deslocam. No gráfico, os 5 clientes mais afastados (circulados) ficam nas bordas. Removendo os 22 clientes mais afastados (5%) e refazendo o agrupamento, o centroide do G0 se desloca 0,54 desvio padrão (os outros dois, 0,16 e 0,18) e o ARI entre as duas soluções é 0,87: a solução é razoavelmente estável, mas não insensível. O efeito é muito mais forte **sem a transformação logarítmica**: a partição vira 393, 45 e 2 clientes, ou seja, os outliers formam grupos próprios (ver Tarefa 1).

