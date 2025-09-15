# Comparação de Algoritmos de Aprendizado Supervisionado em Problemas de Classificação
###### *Trabalho 1 da disciplina de Machine Learning*
###### Prof: Gustavo Taiji Naozuka
**Objetivo:** Aplicar conceitos de aprendizado supervisionado em problemas reais de classificação, explorando diferentes algoritmos e avaliando seus desempenhos a partir de métricas adequadas.

## Orientações gerais
- O trabalho poderá ser desenvolvido em grupos de até 4 integrantes.
- O grupo deverá escolher uma base de dados pública (ex.: UCI, Kaggle, IBGE, INEP, Open Data Brasil, etc.).
- Um integrante do grupo deverá entregar: um arquivo .pdf do trabalho escrito e um código em Jupyter Notebook. Tanto o trabalho escrito quanto o Jupyter Notebook deverão conter os nomes de todos os integrantes do grupo.
- O trabalho deverá ser escrito em formato de artigo científico de até 7 páginas, seguindo o template da Sociedade Brasileira de Computação (SBC).
- O código deverá ser implementado em Jupyter Notebook, com markdowns explicativos separando as etapas do trabalho.
- **Observação:** Semelhanças significativas de código/trabalho escrito resultarão em nota zero para todos os grupos envolvidos.

## Estrutura do trabalho escrito
- Título, autores e resumo. Não colocar: endereço institucional e abstract.
- **Introdução:** contextualização do problema, justificativa da escolha da base de dados e objetivos.
- **Metodologia:** descrição da base de dados, pré-processamento (se necessário), algoritmos utilizados (mínimo 2 algoritmos, podendo ser aqueles abordados em aula ou outros algoritmos da literatura) e métricas de avaliação empregadas.
- Resultados e Discussão: descrição dos experimentos, comparação e interpretação dos resultados.
- **Conclusão:** síntese do trabalho, limitações e sugestões futuras.
- Referências no padrão ABNT.

## Estrutura do Jupyter Notebook
- Título, autores e objetivos.
- Importação das bibliotecas.
- Carregamento e descrição da base de dados.
- Pré-processamento (se necessário). Ex: dados faltantes, transformação de dados categóricos em numéricos, escalas diferentes, outliers, dados desbalanceados, dados duplicados ou inconsistentes, ...
- Implementação dos algoritmos escolhidos (mínimo 2 algoritmos, podendo ser aqueles abordados em aula ou outros algoritmos da literatura). Para alguns algoritmos, pode ser necessário avaliar diferentes valores para os hiper-parâmetros ou outras opções que o método fornece. Por exemplo, para o k-NN, avaliar diferentes valores de k e métricas de distância.
- Avaliação dos modelos com métricas e gráficos. Lembre-se de utilizar validação cruzada para divisão do conjunto de dados e métricas adequadas ao problema de classificação.
- Discussão dos resultados.

## Dataset utilizado
Os dados do dataset foram coletados na rampa de acesso de Glendale para a rodovia 101 Norte em Los Angeles. O sensor de loop estava localizado próximo o suficiente do estádio do Dodgers para registrar tráfego incomum após um jogo do time, mas não tão próximo ou tão intensamente utilizado pelo tráfego do evento a ponto de o sinal do tráfego extra ser excessivamente óbvio.
- **OBSERVAÇÃO:** Trata-se de uma rampa de acesso próxima ao estádio, portanto o tráfego relacionado ao evento **COMEÇA** no final ou próximo ao final do horário do evento.
- As observações foram feitas ao longo de 25 semanas, com 288 intervalos de tempo por dia (agregados de contagem a cada 5 minutos).
- O objetivo é prever a ocorrência de um jogo de beisebol no estádio dos Dodgers.
- Encontrado em: [https://archive.ics.uci.edu/dataset/157/dodgers+loop+sensor]
