Problema 3 – Classificac¸ao de Qualidade de Vinhos Utilizando ˜ 𝑘-NN
A ind´ustria vin´ıcola depende historicamente de avaliac¸oes humanas — caras, lentas e subjetivas. Um modelo preditivo ˜
de qualidade tem aplicac¸oes diretas no controle de lotes antes do engarrafamento, na precificac¸ ˜ ao e na certificac¸ ˜ ao de ˜
denominac¸ao de origem por ˜ org ´ aos reguladores. Na pesquisa enol ˜ ogica (ci ´ encia que estuda a produc¸ ˆ ao, conservac¸ ˜ ao e ˜
analise de vinhos), ele tamb ´ em ajuda a identificar quais vari ´ aveis qu ´ ´ımicas mais influenciam a percepc¸ao de qualidade, ˜
orientando decisoes de produc¸ ˜ ao. ˜
Plataformas como a Vivino (https://www.vivino.com/BR/pt-BR/) ja utilizam ´ machine learning com esse objetivo. No Brasil, o contexto do Vale do Sao Francisco — maior polo de vinhos tropicais do mundo, localizado aqui no ˜
Nordeste — torna o problema especialmente proximo e motivador para nossa realidade. ´
Neste projeto, voceˆ e convidado a construir um modelo de classificac¸ ´ ao de vinhos em tr ˜ es categorias de qualidade ˆ
(baixa, media e alta) a partir de um conjunto de dados qu ´ ´ımicos utilizando o 𝑘-NN (k-Nearest Neighbors)
O projeto utiliza o arquivo winequality-red.csv do dataset Wine Quality (https://archive.ics.uci.edu/
ml/datasets/Wine+Quality). O dataset foi criado por Paulo Cortez et al. (Universidade do Minho, Portugal) e
publicado em 2009. Ele re´une 1.599 amostras de vinho tinto da regiao do Vinho Verde, no noroeste de Portugal. ˜
Cada amostra foi submetida a testes laboratoriais que geraram 11 atributos f´ısico-qu´ımicos — como acidez fixa,
acidez volatil, teor de ´ acido c ´ ´ıtrico, ac¸ ´ucar residual, cloretos, dioxido de enxofre, densidade, pH, sulfatos e teor ´
alcoolico. O r ´ otulo de qualidade ( ´ quality) e um valor inteiro de 0 a 10 atribu ´ ´ıdo por sommeliers em avaliac¸oes cegas, ˜
representando a mediana de pelo menos tres avaliac¸ ˆ oes humanas. ˜
Realize as seguintes tarefas utilizando o dataset fornecido:
1. Preparac¸ao dos dados: carregue o ˜ dataset e recodifique a variavel ´ quality em tres classes: notas ˆ ≤ 5 como
ruim, 6–7 como medio ´ e ≥ 8 como bom. Em seguida, normalize todos os atributos utilizando Min-Max ou
𝑍-score. Justifique por que a normalizac¸ao˜ e uma etapa obrigat ´ oria para o ´ 𝑘-NN.
2. Divisao treino/teste: divida os dados em 80% para treino e 20% para teste, utilizando divis ˜ ao estratificada para ˜
garantir que a proporc¸ao das tr ˜ es classes seja mantida em ambos os conjuntos. ˆ
3. Treinamento e escolha do 𝑘: treine o modelo para diferentes valores de 𝑘 (ex: 1, 3, 5, 7, 11) utilizando validac¸ao˜
cruzada de 5 folds sobre o conjunto de treino. Plote a acuracia m ´ edia em func¸ ´ ao de ˜ 𝑘 e escolha o valor com
melhor desempenho. O que acontece com o modelo para valores muito baixos e muito altos de 𝑘?
4. Avaliac¸ao do modelo: utilizando o conjunto de teste, calcule a matriz de confus ˜ ao e as m ˜ etricas de acur ´ acia, ´
precisao, ˜ recall e F1-score para cada classe. Responda: Entre quais classes o modelo erra com mais frequencia?
