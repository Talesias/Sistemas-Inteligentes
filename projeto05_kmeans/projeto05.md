Um distribuidor atacadista atende, ao mesmo tempo, clientes muito diferentes entre si: o restaurante que compra
hortifr´uti e congelados em alto giro, o pequeno mercado de bairro que se abastece de mercearia, latic´ınios e produtos
de limpeza, entre outros. Tratar todos eles com a mesma pol´ıtica comercial e, na pr ´ atica, desperdic¸ar margem e ´
capacidade log´ıstica.
A segmentac¸ao de clientes tem aplicac¸ ˜ oes diretas no planejamento de rotas e estoques, na definic¸ ˜ ao de campanhas ˜
promocionais dirigidas, na concessao de limites de cr ˜ edito e na identificac¸ ´ ao de clientes at ˜ ´ıpicos, que merecem
atendimento dedicado. No Brasil, o crescimento do modelo de atacarejo e a importancia dos distribuidores regionais ˆ
no abastecimento do interior do Nordest, tornam o problema especialmente concreto para a nossa realidade.
Este projeto tem uma caracter´ıstica que o distingue dos anteriores: nao existe um r ˜ otulo a ser previsto. Ningu ´ em´
classificou previamente os clientes em “perfil restaurante” ou “perfil varejo’, se esses grupos existem, eles precisam
emergir da propria estrutura dos gastos. ´ E o dom ´ ´ınio do aprendizado nao supervisionado. ˜
Neste projeto, voceˆ e convidado a identificar perfis de consumo de clientes de um distribuidor atacadista por meio do ´
algoritmo de agrupamento 𝑘-Means.
O projeto utiliza o arquivo Wholesale customers data.csv do dataset Wholesale Customers (https://archive.
ics.uci.edu/dataset/292/wholesale+customers). O dataset foi doado por Margarida G. M. S. Cardoso
(ISCTE-IUL, Lisboa, Portugal) em 2014 e re´une 440 clientes de um distribuidor atacadista portugues. Para cada cli- ˆ
ente sao registrados os gastos anuais, em unidades monet ˜ arias, em seis categorias de produto — ´ Fresh (frescos), Milk
(latic´ınios), Grocery (mercearia), Frozen (congelados), Detergents Paper (detergentes e papel) e Delicassen
(delicatessen) — alem de dois atributos categ ´ oricos: ´ Channel, indicando se o cliente pertence ao canal Horeca (hoteis, ´
restaurantes e cafes, 298 clientes) ou ´ Retail (varejo, 142 clientes), e Region, indicando sua localizac¸ao (Lisboa, Porto ˜
ou outras regioes). ˜
Observac¸ao: ˜ os atributos Channel e Region nao˜ devem ser utilizados no agrupamento. Eles serao reservados para ˜
a validac¸ao externa da tarefa 5. O agrupamento deve basear-se exclusivamente nas seis categorias de gasto. ˜
Realize as seguintes tarefas utilizando o dataset fornecido:
1. Preparac¸ao dos dados: carregue o ˜ dataset e separe Channel e Region do restante. Para cada uma das seis
categorias de gasto, calcule media, mediana, desvio padr ´ ao e valores extremos, e construa um ˜ boxplot ou
histograma. Voce observar ˆ a distribuic¸ ´ oes fortemente assim ˜ etricas ´ a direita, com poucos clientes de gasto `
muito elevado — em Fresh, por exemplo, os valores vao de dezenas a mais de cem mil unidades monet ˜ arias. ´
Aplique entao a transformac¸ ˜ ao logar ˜ ´ıtmica 𝑥
′ = log(1 + 𝑥) e, em seguida, padronize os atributos por 𝑍-score.
Justifique as duas etapas separadamente: (a) por que a transformac¸ao logar ˜ ´ıtmica e recomend ´ avel diante da ´
assimetria e da presenc¸a de outliers, e (b) por que a padronizac¸ao˜ e determinante para o ´ 𝑘-Means, relacionando
sua resposta ao uso da distancia euclidiana na atribuic¸ ˆ ao dos pontos aos centr ˜ oides. Calcule tamb ´ em a matriz ´
de correlac¸ao entre as seis categorias e comente o par ˜ Grocery–Detergents Paper.
2. Ausencia de divis ˆ ao treino/teste: diferentemente dos projetos anteriores, aqui ˜ nao˜ se separa um conjunto de
teste. Explique por que, contrastando o objetivo do aprendizado supervisionado (estimar o erro de generalizac¸ ˆ ao˜
sobre rotulos conhecidos) com o do n ´ ao supervisionado. Discuta, ent ˜ ao, qual ˜ e a dificuldade central da validac¸ ´ ao˜
em agrupamento: como afirmar que uma partic¸ao˜ e “boa” sem dispor de um gabarito? ´
3. Escolha do n´umero de grupos: execute o 𝑘-Means para 𝑘 variando de 2 a 10. Plote a inercia (soma dos ´
quadrados intra-cluster) em func¸ao de ˜ 𝑘 e aplique o metodo do cotovelo; em seguida, plote o coeficiente de ´
silhueta medio em func¸ ´ ao de ˜ 𝑘. Escolha o valor de 𝑘 justificando sua decisao˜ a luz dos dois crit ` erios e da ´
interpretabilidade comercial dos grupos resultantes — um segmento so´ e ´util se a ´ area comercial souber o que ´
fazer com ele. Responda: por que a inercia decresce monotonicamente com ´ 𝑘 e por que, portanto, ela nao pode ˜
ser usada isoladamente como criterio de escolha? O que ocorre no caso limite em que ´ 𝑘 e igual ao n´umero de ´
amostras?
4. Inicializac¸ao e converg ˜ encia: o ˆ 𝑘-Means converge para um m´ınimo local, e nao necessariamente global. ˜
Execute o algoritmo dez vezes com init=’random’ e n init=1, usando sementes distintas, e registre a
inercia final de cada execuc¸ ´ ao. Repita o procedimento com ˜ init=’k-means++’. Compare a dispersao dos ˜
resultados nos dois casos e explique qual problema a inicializac¸ao˜ k-means++ busca mitigar e de que forma.
5. Interpretac¸ao e validac¸ ˜ ao dos grupos: com o ˜ 𝑘 escolhido, recupere os centroides, reverta-os para a escala ´
original de gasto (desfazendo a padronizac¸ao e a transformac¸ ˜ ao logar ˜ ´ıtmica) e apresente-os em uma tabela
e em um grafico de barras agrupadas por categoria. Atribua a cada agrupamento um nome descritivo de ´
natureza comercial, baseado nas categorias em que ele se destaca. Em seguida, realize uma validac¸ao externa: ˜
construa tabelas de contingencia cruzando o agrupamento de cada cliente com ˆ Channel e com Region, e
calcule o ´ındice Rand ajustado (Adjusted Rand Index) entre a partic¸ao obtida e a vari ˜ avel ´ Channel. Responda:
o algoritmo, sem jamais ter visto o canal de venda, foi capaz de reconstru´ı-lo? Qual conjunto de categorias
de produto mais separa Horeca de Retail? E a regiao geogr ˜ afica — ela se reflete nos agrupamentos? Por fim, ´
projete os dados em duas dimensoes por PCA, colorindo os pontos por agrupamento, e discuta duas limitac¸ ˜ oes ˜
do 𝑘-Means evidenciadas nesse grafico: a tend ´ encia a formar grupos de geometria esf ˆ erica e de tamanhos ´
semelhantes e a sensibilidade a outliers.
