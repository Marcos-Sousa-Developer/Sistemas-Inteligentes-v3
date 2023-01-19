
# Sistemas Inteligentes 2021/2022

## Mini-projeto 3: Aprendizagem Automática



<!-- #region -->
### Introdução

A cirrose é uma fase tardia da cicatrização (fibrose) do fígado causada por muitas formas de doenças e condições hepáticas, tais como a hepatite e o alcoolismo crónico.

O objetivo deste trabalho é classificar o estágio (Stage) da doença de cirrose num paciente considerando as suas características e o seu historial clínico. Para tal vão usar os métodos de Aprendizagem Automática aprendidos nas últimas semanas tal como foi feito nos guiões das aulas de laboratório.

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



## Grupo: 02

Número: 55852 - Nome: Marcos Leitão

Número: 56909 - Nome: Miguel Fernandes



## 1. Processamento dos dados


Usaremos a biblioteca pandas para a leitura do ficheiro csv train e apresentação de dados.

<b>Matriz de dados</b>

```python
import pandas as pd

df_cj = pd.read_csv("train.csv", sep=',')

df_cj
```

<b style='color: red'> Nota !! </b> <br>
Antes de passarmos para o próximo passo temos que separar as variáveis dependentes das independentes e usar apenas as colunas relevantes dentro do dataset. <br>
Vamos também extrair os nomes das colunas, pois serão úteis no futuro para interpretar os resultados e modelos construídos

```python
import numpy as np

col_indexs = [i for i in range(1,df_cj.shape[1]-1)] #indices das colunas relevantes
col_names = np.array(df_cj.columns)[col_indexs] #nomes das colunas relevantes

print('Nome das colunas relevantes e indepentes: ')
print(col_names)
print()

print('Valor das colunas relevantes e indepentes: ')
indpValues = df_cj.values[:,col_indexs] 
print(indpValues)
print()

print('Nome das colunas relevantes e dependentes: ')
print(np.array(df_cj.columns)[df_cj.shape[1]-1])
print()

print('Valor das colunas relevantes e dependentes: ')
deptValues = df_cj.values[:,19] 
print(deptValues)
```

## 2. Ajustamento dos modelos
### 2.1. Validar um modelo usando conjunto de treino e teste
Para validar o modelo vamos fornecer novos dados (que não foram vistos pelo modelo) para os quais temos classes.

<b style= 'color: red' >Nota!!</b> <br>
Os classificadores do scikit-learn, não podem usar dados categóricos na matriz de dados, pelo que estes terão que ser binarizados, ou seja cada valor possível de uma variável é transformado numa coluna que pode ter valores 1 ou 0, consoante esse valor ocorra ou não.

```python
# primeiro construir um dataframe para a matriz X
df_cjX = df_cj[col_names]

# criar um novo DataFrame
df_cjXdum = pd.get_dummies(df_cjX, columns = col_names)

df_cjXdum 

print()

Xdum=df_cjXdum.values
```

Vamos então dividir o nosso conjunto de dados
- o conjunto de treino (x_train)
- o conjunto de teste (x_test)
- dados e as classes de treino (y_train)
- dados e as classes de teste (y_test)

```python
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
```

Vamos aplicar dois testes. <br>
Para uma primeira interação, vamos dividir o nosso conjunto de dados em 0.25 teste e 0.75 treino. <br>
Para um segunda interação, iremos descobrir qual o melhor modelo.


<b> 0.25 teste e 0.75 treino </b>


#### 2.1.1 Árvores de Decisão

```python
x_train, x_test, y_train, y_test = train_test_split(Xdum,list(deptValues),test_size=0.25,train_size=0.75)
```

Treinar uma árvore de decisão usando apenas o conjunto de treino

```python
sk_cj = DecisionTreeClassifier(criterion='entropy')
sk_cj.fit(x_train,y_train)
```

<b> Accuracy </b> <br>
Podemos agora fazer previsões usando o conjunto de teste e comparar essas previsões com as classes verdadeiras (y_test), obtendo assim a exatidão (accuracy) do nosso modelo.

```python
sk_cj.score(x_test, y_test)
```

<b> Cross-validation </b> <br>
Para avaliarmos o desempenho do nosso modelo iremos recorrer a uma técnica mais robusta, designada por validação cruzada, cross-validation, que consiste numa simples divisão do conjunto de dados em dois sub-conjuntos, um de treino e um de teste. 

```python
from sklearn.model_selection import cross_val_score

sk_cj = DecisionTreeClassifier(criterion='entropy')

scores = cross_val_score(sk_cj,X=Xdum,y=list(deptValues),cv=10)

print('CV accuracy:', *scores, sep='\n\t')

print('Average CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
```

#### 2.1.2 K vizinhos mais próximos 
O algoritmo encontra os vizinhos mais próximos no conjunto de treino <br>
Vamos então aplicar o k-NN aos dados, começando com 1 vizinho mais próximo valide o modelo calculando a exatidão nos conjuntos de treino e de teste.
Relembrando que a divisão é feita em 0.25 teste e 0.75 treino

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

knn_cj = KNeighborsClassifier(n_neighbors=1)

x_train, x_test, y_train, y_test = train_test_split(Xdum,list(deptValues),test_size=0.25,train_size=0.75)

```

Treinar usando apenas o conjunto de treino

```python
knn_cj.fit(x_train,y_train)
```

<b> Accuracy </b>

```python
knn_cj.score(x_test, y_test)
```

<b> Cross-validation </b>

```python
scores = cross_val_score(knn_cj,X=Xdum,y=list(deptValues),cv=10)

print('CV accuracy:', *scores, sep='\n\t')

print('Average CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
```

#### 2.1.3 Naive Bayes
Naive Bayes são uma familia de classificadores probabilisticos baseados no teorema de Bayes e com uma forte suposição de independência entre os atributos. <br>
O modelo é chamado de "naive" porque não se espera que os atributos sejam independentes, mesmo condicionados pela classe.

```python
x_train, x_test, y_train, y_test = train_test_split(Xdum,list(deptValues),test_size=0.25,train_size=0.75)

from sklearn.naive_bayes import GaussianNB

nb_cj = GaussianNB()

nb_cj.fit(x_train, y_train)
```

<b> Accuracy </b> <br>

```python
nb_cj.score(x_test, y_test)
```

<b> Cross-validation </b> 

```python
scores = cross_val_score(nb_cj,X=Xdum,y=list(deptValues),cv=10)

print('CV accuracy:', *scores, sep='\n\t')

print('Average CV accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))
```

### 2.2. Seleção e apresentação do melhor modelo
Relembrando: 

- <b> Xdum </b> : corresponde aos valores do novo dataframe binarizado
- <b> deptValues </b>: corresponde aos aos valores dependentes de cada linha

Para selecionar o melhor modelo, vamos fazer uma função que para um determinado modelo testa a Árvores de Decisão, Naive Bayes e K-vizinhos ( seleciona o k que tiver maior accuracy e cross-validation). <br>

```python
def best(test_size=0.5,train_size=0.5,data=Xdum,dependent=list(deptValues)):

    x_train_final= ''
    x_test_final = ''
    y_train_final = ''
    y_test_final = ''

    accuracy_tree = 0
    cross_tree = 0

    accuracy_kv = 0
    cross_kv = 0

    accuracy_b = 0
    cross_b = 0
    i = 0

    x_train, x_test, y_train, y_test = train_test_split(data,dependent,test_size=test_size,train_size=train_size,random_state=2)

    #Árvores de Decisão
    sk_cj = DecisionTreeClassifier(criterion='entropy',random_state=2)
    sk_cj.fit(x_train,y_train)
    accuracy_tree_test = sk_cj.score(x_test, y_test)

    scores = cross_val_score(sk_cj,X=Xdum,y=list(deptValues),cv=10) 
    cross_tree_test = np.mean(scores) 

    accuracy_tree = accuracy_tree_test
    cross_tree = cross_tree_test

    #Naive Bayes
    nb_cj = GaussianNB()
    nb_cj.fit(x_train, y_train)
    accuracy_b_test = nb_cj.score(x_test, y_test)

    scores = cross_val_score(nb_cj,X=Xdum,y=list(deptValues),cv=10)
    cross_b_test = np.mean(scores)

    accuracy_b = accuracy_b_test
    cross_b = cross_b_test

    #K vizinhos mais próximos
    for k in range(1,165):

        knn_cj = KNeighborsClassifier(n_neighbors=k)        
        knn_cj.fit(x_train, y_train)
        accuracy_kv_test = knn_cj.score(x_test, y_test)

        scores = cross_val_score(knn_cj,X=Xdum,y=list(deptValues),cv=10)
        cross_kv_test = np.mean(scores)


        if accuracy_kv_test > accuracy_kv and cross_kv_test> cross_kv:

            i = k
            accuracy_kv = accuracy_kv_test
            cross_kv = cross_kv_test


    x_train_final = x_train
    x_test_final = x_test 
    y_train_final = y_train
    y_test_final = y_test

    return (x_train_final,y_train_final,x_test_final,y_test_final,accuracy_tree,cross_tree,accuracy_kv,cross_kv,i,accuracy_b,cross_b)
```

Vamos testar vários modelos coloca-los numa tabela e escolher o melhor.
Com as seguintes colunas:

- test_size	
- train_size	
- ArvoreTestes_accuracy	
- ArvoreTestes_cross	
- Kvizinhos_accuracy	
- Kvizinhos__cross	
- Kvizinhos_k	
- Nbayes_accuracy	
- Nbayes_accuracy_cross

<b> <span style='color: red'> Nota !!! </span> </b> <br>
A execução do código abaixo, implica um tempo esperado até obter os resultados de aproximadante 8 minutos, dependendo do processador, para resultados mais rápido aumentar o delta <b> default = 0.01 </b>. <br>
<span style='color: red'> Ao aumentar o delta obtem-se menos informação.</span> <br>

```python
test_size=0.5
train_size=0.5

columns = ['test_size','train_size','ArvoreTestes_accuracy','ArvoreTestes_cross','Kvizinhos_accuracy',
                                    'Kvizinhos__cross','Kvizinhos_k','Nbayes_accuracy','Nbayes_accuracy_cross']
data = []
count = 0
while(test_size >= 0.01):

    melhor = best(round(test_size,2),round(train_size,2))
    
    data += [[round(test_size,2),round(train_size,2),melhor[4],melhor[5],melhor[6],melhor[7],
            melhor[8],melhor[9],melhor[10]]]
    
    test_size -= 0.01
    train_size +=0.01

df_modelo = pd.DataFrame(data, columns = columns)

df_modelo
      
```

Faltou-nos testar para o caso de test_size = 0.01 e train_size = 0.99
Vamos testar

```python
melhor = best(0.01, 0.99)

print('Arvore Testes')
print('accuracy',melhor[4])
print('cross',melhor[5])
print()

print('K vizinhos')
print('accuracy',melhor[6])
print('cross',melhor[7])
print('k',melhor[8])
print()

print('N bayes')
print('accuracy',melhor[9])
print('cross',melhor[10])
```

Pela informação disponibilizada pela tabela e pelo ultimo teste, chegamos a conclusão que o melhor modelo a partida será: <br>
- test_size = 0.01 e train_size = 0.99

```python
melhor = best(0.01, 0.99)

print('Conjunto de Treino')
print(melhor[0])
print()

print('Conjunto de Teste')
print(melhor[2])
print()
```

## 3. Validação do modelo final ajustado com um conjunto de validação independente
Mediante a nossa escolha anterior, agora testar o desempenho sobre `test.csv`.


* Preparação do data set

```python
df_cjT = pd.read_csv("test.csv", sep=',')

x = list(df_cjX[col_names].values) #conjunto treino
y = list(df_cjT[col_names].values) #conjunto teste

data = x + y 

df = pd.DataFrame(data, columns = col_names)

df_Final = pd.get_dummies(df, columns = col_names)

col_names_dum=np.array(df_Final.columns)

XdumFinal= df_Final.values #valores dos conjuntos concatenados

df_Final
```

Vamos verificar o desempenho .

```python
melhor = best(0.01, 0.99,XdumFinal[:len(x)],list(deptValues))

print('Conjunto de Treino')
print(melhor[0])
print()

print('Conjunto de Teste')
print(melhor[2])
print()

print('Arvore Testes')
print('accuracy',melhor[4])
print('cross',melhor[5])
print()

print('K vizinhos')
print('accuracy',melhor[6])
print('cross',melhor[7])
print('k',melhor[8])
print()

print('N bayes')
print('accuracy',melhor[9])
print('cross',melhor[10])
print()
```

## 4. Competição

Relembrando:

- x: lista do #conjunto treino
- XdumFinal: #valores dos conjuntos concatenados 
- deptValues: valores dependentes

```python
knn_cj = KNeighborsClassifier(n_neighbors=4)        
knn_cj.fit(XdumFinal[:len(x)], list(deptValues))

count=len(x)
print('ID,Stage')
for i in XdumFinal[len(x):]:
    p = knn_cj.predict([i])
    print(str(count)+','+str(int(p[0])))
    count +=1
        

```

