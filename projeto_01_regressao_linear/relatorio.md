# Problema 1 — Regressão Linear para Previsão de Produção de Energia em uma Usina de Ciclo Combinado

## 1. Dados e pré-processamento

- **Dataset:** UCI Combined Cycle Power Plant (CCPP), arquivo `dados/Folds5x2_pp.csv`: **9.568 amostras**, 4 variáveis independentes (AT, V, AP, RH) e a variável alvo PE (MW).
- **Duplicatas:** há 41 linhas idênticas (0,4%). Foram mantidas: podem ser leituras repetidas dos sensores e não distorcem o resultado de forma relevante.
- **Divisão treino/teste:** 80% / 20% com `train_test_split(random_state=42)` → **7.654 amostras de treino e 1.914 de teste**.
- **Padronização:** `StandardScaler` (z-score), **ajustado apenas no treino**. A divisão acontece antes do ajuste do padronizador e do imputador, para que nenhuma informação do teste influencie o treino (*data leakage*). A padronização coloca as variáveis na mesma escala (AP ≈ 1000 mbar contra AT ≈ 20 °C) e permite comparar a magnitude dos coeficientes.
- **Modelo:** `Pipeline(imputador → padronizador → LinearRegression)`.

As correlações das tarefas 1 e 2 são análises exploratórias (não ajustam nenhum modelo) e foram calculadas sobre o dataset completo.

## 2. Tarefa 1 — Correlação entre as variáveis independentes e PE

| Variável | Pearson com PE | Spearman com PE |
|---|---:|---:|
| **AT** (temperatura) | **−0,948** | −0,944 |
| **V** (vácuo de escape) | **−0,870** | −0,884 |
| AP (pressão) | +0,518 | +0,543 |
| RH (umidade) | +0,390 | +0,390 |


**Resposta.** As variáveis de maior influência sobre PE são **AT** e **V**, ambas com correlação negativa e forte: quanto maior a temperatura ambiente e o vácuo de escape, menor a potência gerada. Isso é coerente com o enunciado (temperatura mais alta reduz a densidade do ar e a eficiência da combustão; o vácuo de escape está ligado à pressão na turbina a vapor). **AP** e **RH** têm correlação positiva e moderada. Pearson e Spearman dão resultados muito próximos, o que indica que a relação dominante é aproximadamente monotônica e linear. Os diagramas de dispersão mostram AT × PE quase linear, com leve curvatura nas pontas. Correlação indica associação, não causalidade.

## 3. Tarefa 2 — Correlação entre as variáveis independentes

| | AT | V | AP | RH |
|---|---:|---:|---:|---:|
| **AT** | 1,00 | **0,84** | −0,51 | −0,54 |
| **V** | 0,84 | 1,00 | −0,41 | −0,31 |
| **AP** | −0,51 | −0,41 | 1,00 | 0,10 |
| **RH** | −0,54 | −0,31 | 0,10 | 1,00 |

Fatores de inflação da variância (VIF): **AT = 5,98**, **V = 3,94**, AP = 1,45, RH = 1,71.

**Resposta.** Sim, existe uma relação forte: **AT × V (r = +0,84)**. AT também se correlaciona moderadamente com RH (−0,54) e AP (−0,51). Isso indica **multicolinearidade moderada**: o VIF de AT (≈ 6) passa da regra prática de 5, mas fica abaixo de 10, então a colinearidade não é severa. Na prática, os coeficientes individuais de AT e V ficam menos estáveis e mais difíceis de interpretar isoladamente, porque as duas variáveis carregam informação parecida. A capacidade de previsão do modelo quase não é afetada: ao remover V, o R² no teste cai só de 0,9315 para 0,9238 (RMSE de 4,43 para 4,67 MW).

## 4. Tarefa 3 — Modelo de regressão linear múltipla

Coeficientes do modelo treinado com os 7.654 exemplos de treino:

| Variável | Coef. padronizado | Coef. em unidade original |
|---|---:|---:|
| AT | −14,771 | −1,9770 MW/°C |
| V | −2,985 | −0,2348 MW/cmHg |
| AP | +0,344 | +0,0583 MW/mbar |
| RH | −2,304 | −0,1581 MW/% |

**Equação (unidades originais):**

`PE = 458,596 − 1,9770·AT − 0,2348·V + 0,0583·AP − 0,1581·RH`

Conferências feitas no código: (i) a equação em unidades originais reproduz exatamente as previsões do pipeline; (ii) os coeficientes obtidos pela **equação normal** `Φ = (XᵀX)⁻¹Xᵀy` coincidem com os do scikit-learn.

**Interpretação.**
- Mantendo as demais variáveis constantes, cada +1 °C em AT reduz PE em cerca de **1,98 MW**. Na escala padronizada, AT domina (|−14,77|), seguida de V (−2,98) e RH (−2,30); AP tem efeito pequeno (+0,34).
- RH tem correlação **positiva** com PE isoladamente (+0,39), mas coeficiente **negativo** no modelo múltiplo (−0,158 MW/%). A inversão de sinal é efeito da correlação entre as variáveis: RH é maior em dias mais frios (r(AT, RH) = −0,54), e dias mais frios produzem mais energia. Depois de descontar o efeito de AT, o efeito próprio de RH é ligeiramente negativo. Esse é um exemplo concreto de por que, na presença de multicolinearidade, os coeficientes isolados devem ser lidos com cuidado.

## 5. Tarefa 4 — Avaliação do desempenho

| Conjunto | R² | RMSE (MW) | MAE (MW) |
|---|---:|---:|---:|
| Treino | 0,9280 | 4,589 | 3,647 |
| **Teste** | **0,9315** | **4,428** | **3,543** |
| Referência (prever sempre a média) — teste | −0,0001 | 16,917 | 14,706 |

- **Validação cruzada 5-fold (no treino):** R² = 0,9279 ± 0,0044, com folds entre 0,921 e 0,935.
- **Contexto:** PE varia de 420,26 a 495,76 MW (amplitude de 75,5 MW; média de 454,4 MW). O RMSE de teste equivale a **5,9% da amplitude** e o MAE a **0,78% da média**.
- **Distribuição do erro (teste):** 75,1% das previsões erram até 5 MW e 98,2% erram até 10 MW. O maior erro absoluto é de 28,4 MW. A média dos resíduos é −0,32 MW (praticamente sem viés).


**O modelo é adequado para prever a produção de energia?** **Sim, para o objetivo do enunciado** (apoiar o despacho energético a partir de sensores ambientais), pelos motivos abaixo:

1. **Explica ~93% da variância de PE** (R² = 0,93) e o RMSE cai de 16,9 MW (referência da média) para 4,4 MW, uma redução de cerca de 74%.
2. **Não há sobreajuste:** os desempenhos de treino e teste são praticamente iguais (R² 0,928 e 0,932), e a validação cruzada tem desvio padrão pequeno (0,0044), então o resultado é estável.
3. **O erro é pequeno em termos práticos:** o erro médio absoluto de 3,5 MW é menos de 1% da potência média.
4. **Resíduos bem comportados:** centrados em zero e com formato aproximadamente em sino (histograma), com poucos erros grandes.

**Limitações observadas.** O gráfico de resíduos contra o previsto mostra um padrão em "U": o modelo tende a superestimar PE na faixa central e a subestimar nas pontas. Isso indica uma **leve não linearidade** que a regressão linear não captura, o que também explica o R² não chegar mais perto de 1. Por isso, o modelo é adequado como previsor, mas não é o melhor possível; modelos com termos não lineares seriam o próximo passo natural (não testado neste projeto). Além disso, o modelo só usa condições ambientais e vale para a faixa de operação do dataset; fora dela, não há garantia.

