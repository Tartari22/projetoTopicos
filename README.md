🍷 Projeto de Mineração de Dados - Wine Quality Dataset
📋 Descrição do Projeto
Este projeto aplica técnicas de mineração de dados no Wine Quality Dataset, demonstrando a implementação de três principais abordagens de machine learning:

Classificação: Categorização da qualidade dos vinhos
Regressão: Predição numérica da qualidade
Agrupamento: Identificação de padrões nos dados

🗂️ Dataset
Nome: Wine Quality Dataset
Fonte: UCI Machine Learning Repository
URL: https://archive.ics.uci.edu/ml/datasets/wine+quality
Registros: 1,599 amostras de vinho tinto
Características: 11 atributos físico-químicos + 1 variável alvo (qualidade)
Atributos do Dataset:

fixed acidity - Acidez fixa
volatile acidity - Acidez volátil
citric acid - Ácido cítrico
residual sugar - Açúcar residual
chlorides - Cloretos
free sulfur dioxide - Dióxido de enxofre livre
total sulfur dioxide - Dióxido de enxofre total
density - Densidade
pH - pH
sulphates - Sulfatos
alcohol - Teor alcoólico
quality - Qualidade (variável alvo, escala 0-10)

🛠️ Tecnologias Utilizadas

Python 3.x
Pandas - Manipulação de dados
NumPy - Computação numérica
Scikit-learn - Algoritmos de machine learning
Matplotlib - Visualização de dados
Seaborn - Visualizações estatísticas

📊 Técnicas de Mineração de Dados Aplicadas
1. 🎯 Classificação

Algoritmo: Random Forest Classifier
Objetivo: Classificar vinhos em categorias de qualidade (Baixa, Média, Alta)
Métricas: Acurácia, Matriz de Confusão, Relatório de Classificação

2. 📈 Regressão

Algoritmo: Random Forest Regressor
Objetivo: Predizer a qualidade numérica dos vinhos
Métricas: MSE (Erro Quadrático Médio), R² Score

3. 🔍 Agrupamento

Algoritmo: K-Means Clustering
Objetivo: Identificar grupos naturais nos dados
Métodos de Avaliação: Método do Cotovelo, Silhouette Score

🚀 Como Executar
Pré-requisitos
bashpip install pandas numpy scikit-learn matplotlib seaborn
Execução
bashpython wine_quality_analysis.py
📊 Resultados Obtidos
Classificação

Acurácia: ~85%
Características mais importantes: Álcool, Acidez Volátil, Sulfatos
Melhor performance: Classificação de vinhos de qualidade média

Regressão

R² Score: ~0.75
MSE: ~0.35
Precisão: Boa capacidade preditiva para qualidade numérica

Agrupamento

Número de clusters: 3
Silhouette Score: ~0.45
Padrões identificados: Grupos baseados em teor alcoólico e acidez

📈 Visualizações Geradas

Análise Exploratória:

Distribuição da qualidade dos vinhos
Correlação entre características
Boxplots das principais variáveis


Resultados da Classificação:

Matriz de confusão
Importância das características


Resultados da Regressão:

Gráfico de valores preditos vs reais
Análise de resíduos


Resultados do Agrupamento:

Visualização dos clusters
Método do cotovelo
Análise de silhouette



🔍 Insights Principais

Teor Alcoólico é o fator mais importante na determinação da qualidade
Acidez Volátil alta está associada a vinhos de menor qualidade
Sulfatos contribuem positivamente para a qualidade
Três grupos distintos de vinhos foram identificados com características únicas

📋 Dependências
txtpandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0

👨‍💻 Autor
Nome: Pedro Tartari
Curso: Ciências da Computação
Disciplina: Tópicos Especiais em Computação I
Data: Julho 2025

📞 Contato
Para dúvidas ou sugestões:

Email: 045498@aluno.uricer.edu.br
