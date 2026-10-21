Problema 2 – Regressao Log ˜ ´ıstica para Previsao de Doenc¸as Card ˜ ´ıacas

Doenc¸as cardiovasculares sao a principal causa de morte no mundo, respons ˜ aveis por cerca de 18 milh ´ oes de ˜ obitos ´
por ano segundo a OMS. O diagnostico precoce ´ e determinante para o sucesso do tratamento. Existe, portanto, uma ´
necessidade real de ferramentas capazes de estimar o risco de um paciente com base em dados cl´ınicos simples,
coletados em consulta de rotina. E nesse contexto que o aprendizado de m ´ aquina pode contribuir. Um modelo de ´
classificac¸ao treinado com dados hist ˜ oricos de pacientes diagnosticados ´ e capaz de aprender quais combinac¸ ´ oes de ˜
variaveis cl ´ ´ınicas estao associadas ˜ a presenc¸a de doenc¸a card ` ´ıaca e aplicar esse aprendizado a novos pacientes.
Neste projeto, voceˆ e convidado a construir um modelo utilizando regress ´ ao log ˜ ´ıstica para prever a presenc¸a de doenc¸a
card´ıaca com base em um conjunto de dados cl´ınicos.
O projeto utiliza o dataset Heart Disease Cleveland (UCI Machine Learning Repository – dispon´ıvel em https:
//archive.ics.uci.edu/dataset/45/heart+disease), com registros de 303 pacientes e 13 variaveis cl ´ ´ınicas.
Os diagnosticos foram confirmados por cateterismo card ´ ´ıaco, o que garante rotulos confi ´ aveis. As vari ´ aveis incluem: ´
• age — idade do paciente em anos;
• sex — sexo biologico (1 = masculino, 0 = feminino); ´
• cp — tipo de dor no peito: angina t´ıpica (1), at´ıpica (2), nao anginosa (3) ou assintom ˜ atico (4); ´
• trestbps — pressao arterial em repouso em mmHg; ˜
• chol — colesterol serico total em mg/dl; ´
• fbs — glicemia em jejum acima de 120 mg/dl (1 = sim, 0 = nao); ˜
• restecg — resultado do ECG em repouso: normal (0), alterac¸ao na onda ST-T (1) ou hipertrofia ventricular ˜
(2);
• thalach — frequencia card ˆ ´ıaca maxima atingida no teste de esforc¸o; ´
• exang — angina induzida por exerc´ıcio (1 = sim, 0 = nao); ˜
• oldpeak — depressao do segmento ST durante o esforc¸o, indicador de isquemia; ˜
• slope — inclinac¸ao do segmento ST no pico do esforc¸o: ascendente (1), plano (2) ou descendente (3); ˜
• ca — n´umero de arterias coron ´ arias com obstruc¸ ´ ao vis ˜ ´ıvel na fluoroscopia (0 a 3);
• thal — cintilografia de talio: normal (3), defeito fixo / infarto pr ´ evio (6) ou defeito revers ´ ´ıvel / isquemia ativa
(7);

A sa´ıda e a vari ´ avel ´ target, que indica o diagnostico original de 0 a 4, binarizado para 0 (sem doenc¸a) e 1 (com ´
doenc¸a) antes do treinamento.
Realize as seguintes tarefas utilizando o dataset fornecido:
1. Treinar um modelo de LogisticRegression do scikit-learn utilizando os dados de treino, com os seguintes parametros: ˆ solver=’liblinear’, random state=42 e max iter=1000. Indique quantas iterac¸oes ˜
foram necessarias para a converg ´ encia do modelo. ˆ
2. Avaliar o modelo utilizando os dados de teste e calcular as seguintes metricas: Acur ´ acia, Precis ´ ao, Recall, ˜
F1-score e Curva ROC com valor de AUC.
Realize o pre-processamento dos dados, incluindo a normalizac¸ ´ ao ou padronizac¸ ˜ ao das vari ˜ aveis independentes, ´
tratamento de valores ausentes, divisao dos dados em conjuntos de treino e teste e binarizac¸ ˜ ao das vari ˜ aveis alvo. 
