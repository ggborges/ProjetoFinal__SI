import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


columns = 'obj_ID,alpha,delta,u,g,r,i,z,run_ID,rereun_ID,cam_col,field_ID,spec_obj_ID,class,redshift,plate,MJD,fiber_ID'.split(',')
df = pd.read_csv('C:\\Users\\User\\Documents\\VSCode\\Projeto Final SI\\star_classification.csv', header=None, low_memory=False)
df.columns = columns

print(df.head(10))
print(df.info())
print(df.describe().T)

# Separando atributos e classes

df_ = df.drop(0, axis = 0)
X_ = df_.drop(['obj_ID', 'class'], axis=1).to_numpy()
y_ = df_['class'].values

# pegando 12%  do dataset (12000)

X_useless, X, y_useless, y = train_test_split(X_, y_, 
                                                    test_size = 0.12, 
                                                    random_state = 254325, 
                                                    stratify=y_)

#print(X)
#print(y)

# Pre-processando

le = preprocessing.LabelEncoder()
y = le.fit_transform(y)

#print(y)

# Plotando o dataset

pca = PCA(n_components=3)
X_pca = pca.fit_transform(preprocessing.maxabs_scale(X))

# Gráfica de dispersão

colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, ax = plt.subplots(figsize=(40,32))
classes = le.inverse_transform(sorted(set(y)))

for i, color in enumerate(colors):
  ax.scatter(X_pca[y==i, 0], 
             X_pca[y==i, 1], 
             label=classes[i], 
             color = color, 
             alpha=.5, s = 200)

plt.legend()
ax.grid(True)
plt.show()

# Dividindo entre treino  e teste (15% de teste, 85% de treino)

SEED = 568877441
# best SEED = 568877441
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.15, 
                                                    random_state = SEED, 
                                                    stratify=y)

# Dividindo os dados de treino em dados de treino e dados de validação abaixo
# (90% de treino, 10% de validação)

SEED_ = 56874
# best SEED_ = 56874
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                      test_size = 0.10, 
                                                      random_state = SEED_, 
                                                      stratify=y_train)

# Padronizando as bases

X_train = preprocessing.maxabs_scale(X_train)
X_valid = preprocessing.maxabs_scale(X_valid)
X_test = preprocessing.maxabs_scale(X_test)

# Instanciando o modelo

classificador = GaussianNB()
classificador.fit(X_train, y_train)

# fit the model with the training data
classificador.fit(X_train, y_train)

# predict the target on the train dataset
predict_train = classificador.predict(X_valid)
print('Target on train data',predict_train)

# Accuray Score on train dataset
accuracy_train = accuracy_score(y_valid,predict_train)
print('accuracy_score on validation dataset : ', accuracy_train)

# Plotando curva de acurácio do treino
'''
plt.rcParams["figure.autolayout"] = True

fig, ax = plt.subplots(figsize=(10,8))
ax.set_xlabel("K")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy score vs K")
ax.plot(k_range, k_scores_train, marker='o', label="training")
ax.plot(k_range, k_scores_valid, marker='o', label="validation")
'''

# predict the target on the test dataset
predict_test = classificador.predict(X_test)
print('Target on test data',predict_test) 

# Accuracy Score on test dataset
accuracy_test = accuracy_score(y_test,predict_test)
print('accuracy_score on test dataset : ', accuracy_test)