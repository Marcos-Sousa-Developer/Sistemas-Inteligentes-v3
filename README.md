---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Sistemas Inteligentes 2021/2022

## Mini-projeto 3: Aprendizagem Automática



<!-- #region -->
### Introdução

A cirrose é uma fase tardia da cicatrização (fibrose) do fígado causada por muitas formas de doenças e condições hepáticas, tais como a hepatite e o alcoolismo crónico.

O objetivo deste trabalho é classificar o estágio (Stage) da doença de cirrose num paciente considerando as suas características e o seu historial clínico. Para tal vão usar os métodos de Aprendizagem Automática aprendidos nas últimas semanas tal como foi feito nos guiões das aulas de laboratório.

**[Bónus]** Para além da submissão do Notebook, durante o decorrer do tempo do Projeto 3 irá decorrer uma competição na [plataforma Kaggle](https://www.kaggle.com/competitions/sistemas-inteligentes-projeto-3/), onde os alunos poderão submeter as previsões dos seus modelos, tendo feedback imediato e competindo com os restantes grupos. Para se poderem registar na competição, utilizem o seguinte link: https://www.kaggle.com/t/170f2b29bb6246c78ebc5c63af7e0765.

### Dados

Os detalhes do conjunto de dados das características dos pacientes e do seu historial clínico estão descritos nesta [página](https://www.kaggle.com/competitions/sistemas-inteligentes-projeto-3/data).


### Passos a ter em conta

#### 1. Processamento dos Dados

Nesta fase deverão ser identificadas quais as variáveis a discretizar e quais deverão ter valores contínuos e o dataset deverá ser processado para dar uma matriz `X` e um vector de dados de resposta `y`.


#### 2. Ajustamento de modelos de aprendizagem 

Pretende-se que os alunos explorem os seguintes aspetos:

1. Testar Árvores de Decisão, Naive Bayes e K-vizinhos mais próximos
2. Testar diferentes parâmetros para os modelos e combinações de atributos do conjunto de dados

##### 2.1. Ajustamento de modelos e validação

Os modelos deverão ser ajustados e validados apenas usando o conjunto de dados de treino.

Para cada modelo ajustado deverão ser apresentadas as métricas que ache essenciais para a sua selecção.

##### 2.2. Seleção e apresentação do melhor modelo

Os resultados deverão ser apresentados numa tabela final e o melhor modelo deve ser ilustrado e discutido.

##### 3. Validação do modelo final ajustado com um conjunto de validação independente

**Um só modelo** deve ser seleccionado dos modelos testados anteriormente e o seu desempenho avaliado sobre `test.csv`. 
<!-- #endregion -->

