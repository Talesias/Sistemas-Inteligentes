Problema 1 – Regressao Linear para Previs ˜ ao de Produc¸ ˜ ao de Energia em uma Usina de Ciclo ˜
Combinado

Uma usina termoeletrica de ciclo combinado (UTCC) ´ e uma instalac¸ ´ ao de alta efici ˜ encia que combina turbinas a g ˆ as´
e a vapor para gerar eletricidade. Ela reaproveita o calor dos gases de exaustao da turbina a g ˜ as (ciclo Brayton) para ´
produzir vapor em uma caldeira, acionando uma segunda turbina (ciclo Rankine), aumentando significativamente o
rendimento. A principal vantagem e a efici ´ encia energ ˆ etica, que pode chegar a 60%, reduzindo as emiss ´ oes de gases ˜
de efeito estufa.
Embora apresente alta eficiencia, o principal desafio t ˆ ecnico no uso de uma usina UTCC reside na integrac¸ ´ ao, ˜
sincronizac¸ao e operac¸ ˜ ao conjunta de dois ciclos termodin ˜ amicos distintos (ciclo Brayton e ciclo Rankine) para ˆ
funcionar de forma eficiente. Assim sendo, a potencia el ˆ etrica de sa ´ ´ıda (PE) e altamente sens ´ ´ıvel as condic¸ ` oes ˜
ambientais.
Neste projeto, voce deve construir um modelo matem ˆ atico de regress ´ ao linear m´ultipla para prever a produc¸ ˜ ao de ˜
energia baseando-se em sensores ambientais, permitindo a otimizac¸ao do despacho energ ˜ etico. ´
Para realizar o treinamento do nosso modelo, utilizamos o dataset UCI CCPP (dispon´ıvel em https://archive.
ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant). O dataset contem 9.568 amostras coletadas ao ´
longo de 6 anos. Cada amostra possui as seguintes variaveis: ´
• AT (Ambient Temperature): Temperatura em °C. Afeta a densidade do ar e a eficiencia da combust ˆ ao. ˜
• V (Exhaust Vacuum): Vacuo de escape em cm Hg. Relacionado ´ a press ` ao na turbina a vapor. ˜
• AP (Ambient Pressure): Pressao atmosf ˜ erica em mbar. ´
• RH (Relative Humidity): Umidade relativa em percentagem.
• PE (Net Hourly Electrical Energy Output): A variavel alvo em MW (MegaWatts). ´
Realize as seguintes tarefas utilizando o dataset fornecido:
1. Calcule a correlac¸ao entre as vari ˜ aveis independentes (AT, V, AP, RH) e a vari ´ avel dependente (PE). Quais ´
variaveis parecem ter maior influ ´ encia na produc¸ ˆ ao de energia? ˜
2. Calcule a correlac¸ao entre as vari ˜ aveis independentes (AT, V, AP, RH) entre si. Existe alguma relac¸ ´ ao forte ˜
entre elas? Se sim, isso pode indicar multicolinearidade.
3. Construa um modelo de regressao linear m´ultipla utilizando as vari ˜ aveis independentes (AT, V, AP, RH) para ´
prever a variavel dependente (PE). Divida os dados em treino (80%) e teste (20%). ´
4. Avalie o desempenho do modelo utilizando metricas como R ´ ², RMSE e MAE. O modelo e adequado para ´
prever a produc¸ao de energia? Justifique sua resposta. ˜
Realize o pre-processamento dos dados, incluindo a normalizac¸ ´ ao ou padronizac¸ ˜ ao das vari ˜ aveis independentes, ´
tratamento de valores ausentes e divisao dos dados em conjuntos de treino e teste. 
