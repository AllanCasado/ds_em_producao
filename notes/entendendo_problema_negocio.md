# Estrutura do curso

1. Descrição dos dados
2. Feature Engineering
3. Filtragem de Variáveis
4. Análise Exploratória dos Dados (EDA)
5. Preparação dos Dados
6. Seleção de Variávels com Algoritmo
7. Modelos de Machine Learning
8. Hyperparameter Fine Tuning
9. Interpretação e Tradução do Erro
10. Deploy do Modelo em Produção


# Entendendo o problema de negócio

1. Entender a motivação:
	* Qual o contexto?

2. Entender a causa raiz do problema:
	* Porque fazer uma previsão de vendas (por exemplo)?
	
3. Entender quem é o dono do problema:
	* Quem será o Stakeholder?
	
4. Entender o formato da solução:
	* Granularidade (quer uma previsão por loja, por produto? diária? mensal?)
	* Tipo do problema (classificação, regressã...)
	* Potenciais métodos (usar uma regressão, uma rede neural...)
	* Formato de entrega (um dashboard, um csv...)


# O problema de negócio abordado no curso

* A motivação: o CFO requisitou essa solução durante uma reunião de resultados mensais.
* A causa raiz do problema: investimento em reforma das lojas então precisa saber o quanto de dinheiro a loja vai trazer pra conseguir investir agora.
* O stakeholder: o CFO.
* O formato da solução:
	* Vendas diárias em R$, nas próximas 6 semanas
	* O problema é de predição
	* Possíveis metodos incluem time series, regressão e redes neurais
	* A entrega será feita de modo que as predições possam ser acessadas via celular
