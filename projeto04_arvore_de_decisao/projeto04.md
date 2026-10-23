A concessao de cr ˜ edito ´ e uma das decis ´ oes mais sens ˜ ´ıveis do sistema financeiro: aprovar um cliente inadimplente gera
preju´ızo direto, mas recusar um bom pagador significa perda de receita e exclusao financeira. Diferentemente de outras ˜
aplicac¸oes de ˜ machine learning, aqui nao basta acertar, ˜ e preciso justificar a decis ´ ao. A LGPD (Lei 13.709/2018), ˜
em seu Art. 20, assegura ao titular o direito de solicitar a revisao de decis ˜ oes tomadas exclusivamente com base em ˜
tratamento automatizado, o que torna modelos interpretaveis especialmente valiosos nesse dom ´ ´ınio.
E exatamente esse o apelo da ´ arvore de decis ´ ao: o modelo aprendido pode ser lido como um conjunto de regras de ˜
produc¸ao, uma explicac¸ ˜ ao que um analista, um cliente ou um auditor consegue compreender sem conhecimento de ˜
estat´ıstica. No Brasil, iniciativas como o Cadastro Positivo, o Open Finance e programas de microcredito produtivo ´
orientado como o Crediamigo, do Banco do Nordeste, dependem de avaliac¸oes de risco em larga escala, tornando o ˜
problema proximo da nossa realidade. ´
Neste projeto, voceˆ e convidado a construir um modelo de classificac¸ ´ ao de solicitantes de cr ˜ edito em duas categorias ´
de risco (bom pagador e mau pagador) a partir de dados cadastrais e financeiros, utilizando uma Arvore de Decis ´ ao˜
(Decision Tree).
O projeto utiliza o arquivo german.data do dataset Statlog (German Credit Data) (https://archive.ics.uci.
edu/dataset/144/statlog+german+credit+data). O dataset foi doado por Hans Hofmann (Universidade de
Hamburgo, Alemanha) em 1994 e re´une 1.000 solicitac¸oes de cr ˜ edito reais de um banco alem ´ ao. Cada amostra ˜ e´
descrita por 20 atributos: 7 numericos e 13 categ ´ oricos. Entre os atributos, temos durac¸ ´ ao do empr ˜ estimo em meses, ´
valor solicitado, historico de cr ´ edito, finalidade do empr ´ estimo, saldo em conta corrente, tempo de emprego atual, ´
idade, n´umero de creditos ativos e exist ´ encia de garantias. O r ˆ otulo indica se o cliente representou, de fato, um bom ´
risco (700 amostras) ou um mau risco (300 amostras) de credito. 
Observac¸ao: ˜ os atributos categoricos do arquivo ´ german.data aparecem codificados (A11, A12, . . . ), com o
significado de cada codigo descrito no arquivo ´ german.doc. Alternativamente, a versao com r ˜ otulos leg ´ ´ıveis pode
ser obtida diretamente via:
sklearn.datasets.fetch openml(name=’credit-g’, version=1).
Realize as seguintes tarefas utilizando o dataset fornecido:
1. Preparac¸ao dos dados: carregue o ˜ dataset, identifique quais atributos sao num ˜ ericos e quais s ´ ao categ ˜ oricos. ´
Verifique se existe necessidade de codificac¸ao adequada ˜ as vari ` aveis categ ´ oricas ( ´ one-hot encoding ou ordinal encoding). Avalie o balanceamento entre as classes e comente o impacto de um conjunto desbalanceado
(70%/30%) sobre a interpretac¸ao da acur ˜ acia. ´
2. Justifique por que a normalizac¸ao dos atributos n ˜ ao˜ e necess ´ aria para uma ´ arvore de decis ´ ao. ˜
3. Divisao treino/teste: divida os dados em 80% para treino e 20% para teste, utilizando divis ˜ ao estratificada para ˜
garantir que a proporc¸ao das duas classes seja mantida em ambos os conjuntos. ˜
4. Treinamento e controle de complexidade: treine arvores com diferentes valores de profundidade m ´ axima (ex: ´
2, 3, 5, 7, 10 e sem limite), utilizando validac¸ao cruzada de 5 ˜ folds sobre o conjunto de treino. Plote, em um
mesmo grafico, a acur ´ acia m ´ edia de treino e a de validac¸ ´ ao em func¸ ˜ ao da profundidade, e escolha o valor com ˜
melhor desempenho. O que acontece com o modelo para profundidades muito baixas e muito altas? Relacione
sua resposta aos conceitos de underfitting e overfitting. Repita o experimento trocando o criterio de divis ´ ao˜
dos nos de Gini para entropia e comente se houve diferenc¸a relevante. ´
5. Interpretac¸ao do modelo: usando a ˜ arvore com a profundidade escolhida, gere a visualizac¸ ´ ao gr ˜ afica da ´ arvore ´
e a lista de importancia dos atributos. Responda: quais s ˆ ao os tr ˜ es atributos mais determinantes para a ˆ
classificac¸ao? Escreva, em linguagem natural, o conjunto de regras correspondente ao caminho que leva da ˜
raiz ate a folha com maior n´umero de amostras de mau risco. ´
6. Avaliac¸ao do modelo: utilizando o conjunto de teste, calcule a matriz de confus ˜ ao e as m ˜ etricas de acur ´ acia, ´
precisao, ˜ recall e F1-score para cada classe. Responda: qual tipo de erro o modelo comete com mais frequencia ˆ
e por que ele e o mais grave nesta aplicac¸ ´ ao? Que ajuste voc ˜ e faria no treinamento para reduzi-lo?
