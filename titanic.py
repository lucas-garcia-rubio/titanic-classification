# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 20:05:13 2020

@author: lucas

Coding for Kaggle Titanic Competition
link: https://www.kaggle.com/c/titanic
"""

#%% Importing packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

#%% Importing dataset dataset

train_data = pd.read_csv(r'C:\Users\lucas\Desktop\Titanic Competition\data\train.csv')
test_data = pd.read_csv(r'C:\Users\lucas\Desktop\Titanic Competition\data\test.csv')

#%% Setting dataset.
# Sex column: Male -> 1 Woman -> 0

for i in range(len(train_data.Sex)):
    if train_data.loc[i, 'Sex'] == 'male':
        train_data.loc[i, 'Sex'] = 1
    else:
        train_data.loc[i, 'Sex'] = 0
        
# Embarked column: C -> 0 Q -> 1 S -> 2

for i in range(len(train_data.Embarked)):
    if train_data.loc[i, 'Embarked'] == 'C':
        train_data.loc[i, 'Embarked'] = 0
    elif train_data.loc[i, 'Embarked'] == 'Q':
        train_data.loc[i, 'Embarked'] = 1
    else:
        train_data.loc[i, 'Embarked'] = 2

#%% Creating the input of the model

X = train_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis = 1)
Y = X.Survived
X = X.drop(["Survived"], axis=1)

#%% To numpy so I can scaler with StandardScaler()
    
X = X.to_numpy()
Y = Y.to_numpy()

#%% Imputting the mean in NaN (just for 'Age' column)

replace = X[5, 2]
imputer = SimpleImputer(missing_values=replace, strategy='median')

np.copyto(X[:, 2].reshape((891, 1)), imputer.fit_transform(X[:, 2].reshape(-1,1)))



#%% Visualizing the data format

print("Formato de X: ", X.shape)
print("Formato de Y: ", Y.shape)

#%% Normalizing X

scaler = StandardScaler()

for i in range(len(X[0])):
    if i == 1:
        continue
    np.copyto(X[:, i].reshape((891, 1)), scaler.fit_transform(X[:, i].reshape(-1,1)))
    
#%% Splitting data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

#%% Visualizing the new data splitted

print("Formato de X_train: ", X_train.shape)
print("Formato de X_test: ", X_test.shape)
print("Formato de Y_train: ", Y_train.shape)
print("Formato de Y_test: ", Y_test.shape)

#%% Treinando com SVM

from sklearn import svm
from sklearn.metrics import accuracy_score

clf = svm.SVC()
clf.fit(X_train, Y_train)
predicted_svm_train = clf.predict(X_train)
predicted_svm_test = clf.predict(X_test)

print('Accuracy training set: ', accuracy_score(Y_train, predicted_svm_train))
print('Accuracy test set: ', accuracy_score(Y_test, predicted_svm_test))


#%% Training with MLP

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(64, 64, 64, 32), activation='relu', solver='adam', batch_size=64)
mlp.fit(X_train, Y_train)
predicted_mlp_train = mlp.predict(X_train)
predicted_mlp_test = mlp.predict(X_test)

print('Accuracy training set: ', accuracy_score(Y_train, predicted_mlp_train))
print('Accuracy test set: ', accuracy_score(Y_test, predicted_mlp_test))

#%% Importing Keras

from keras.layers import Dense, Activation, BatchNormalization
from keras.engine.input_layer import Input
from keras.models import Sequential
from keras.regularizers import l2
from keras.initializers import RandomNormal

#%% Building a model with Keras

dim = len(X_train[0, :])

titanic_model = Sequential([
        Dense(32, input_dim=dim, kernel_regularizer=l2(0.1)),
        BatchNormalization(scale=False, beta_initializer='RandomNormal', gamma_initializer='RandomNormal'),
        Activation('relu'),
        Dense(32, kernel_regularizer=l2(0.1)),
        BatchNormalization(scale=False, beta_initializer='RandomNormal', gamma_initializer='RandomNormal'),
        Activation('relu'),
        Dense(32, kernel_regularizer=l2(0.1)),
        BatchNormalization(scale=False, beta_initializer='RandomNormal', gamma_initializer='RandomNormal'),
        Activation('relu'),
        Dense(1),
        Activation('sigmoid'),
        ])


#%% Compiling the model

titanic_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#%% Training the model

titanic_model.fit(x=X_train, y=Y_train, epochs=500, batch_size=64)

#%% Evaluating the model

titanic_model.evaluate(X_test, Y_test)

#%% Now, after choosing a model, export the file .CSV to submit to Kaggle

for i in range(len(test_data.Sex)):
    if test_data.loc[i, 'Sex'] == 'male':
        test_data.loc[i, 'Sex'] = 1
    else:
        test_data.loc[i, 'Sex'] = 0

for i in range(len(test_data.Embarked)):
    if test_data.loc[i, 'Embarked'] == 'C':
        test_data.loc[i, 'Embarked'] = 0
    elif test_data.loc[i, 'Embarked'] == 'Q':
        test_data.loc[i, 'Embarked'] = 1
    else:
        test_data.loc[i, 'Embarked'] = 2

Z = test_data.drop(["PassengerId", "Name", "Ticket", "Cabin", "Fare"], axis = 1)

PassengerId = test_data.drop(["Name", "Ticket", "Cabin", "Fare"], axis = 1)
PassengerId = PassengerId.PassengerId

Z = Z.to_numpy()

imputer = SimpleImputer(missing_values=replace, strategy='median')

np.copyto(Z[:, 2].reshape((418, 1)), imputer.fit_transform(Z[:, 2].reshape(-1,1)))

scaler = StandardScaler()

for i in range(len(Z[0])):
    if i == 1:
        continue
    np.copyto(Z[:, i].reshape((418, 1)), scaler.fit_transform(Z[:, i].reshape(-1,1)))
    
#%% For predicting with SVM
    
Survived = clf.predict(Z)

#%% For predicting with MLP

Survived = mlp.predict(Z)

#%% Generating the .CSV file 

Survived = pd.DataFrame(data=Survived, columns=['Survived'])
PassengerId = pd.DataFrame(data=PassengerId, columns=['PassengerId'])

PassengerId.reset_index(drop=True, inplace=True)
Survived.reset_index(drop=True, inplace=True)

result = pd.concat([PassengerId, Survived], axis = 1)

result.to_csv(r'C:\Users\lucas\Desktop\Titanic Competition\data\y_final.csv', index = False)
