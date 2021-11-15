#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:30:01 2021

@author: fobor
"""
import numpy as np
import pandas as pd
import math
import os, glob
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor,KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict,GridSearchCV
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import random

# #Combine the two databases we have into one single file

# all_files = glob.glob(os.path.join('results/', "*.csv"))
# df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
# df_merged   = pd.concat(df_from_each_file, ignore_index=True)
# #Save the one dataframe table
# df_merged.to_csv( "merged.csv")

def divideDependent(df):
    #Dependent variable
    X = df.drop("load", axis=1)
    X = X.drop("failure", axis=1)
    #Independent Variable
    y = df[["failure"]]
    
    return X,y

df = pd.read_csv("merged.csv")
df = df.iloc[:,1:len(df)]
#copy the main dataframe to preprocess
preprocessed = df.copy()

crossScores = []
modeller = []

#Understanding the database
df.info()
df.describe().T
#Its an ordered data without extreme std values
df.corr()
#We can see that highest correlation with failure is a1
#but its in opposite direction

# sns.jointplot(x="a1", y="a2", data = df, kind = "reg")
# sns.jointplot(x="a1", y="a3", data = df, kind = "reg")
# sns.jointplot(x="a3", y="a2", data = df, kind = "reg")
#We can see from these plots that all a1, a2 and a3 are very much connected to each other

df["failure"].value_counts().plot.barh()
#In this plot we can see that number of positive and negative outcomes are almost equal

############# Question2
##############################################  PREPROCESSING 
#############
#fitting data to preprocessing function
clf = LocalOutlierFactor(n_neighbors=3, contamination=0.1)
clf.fit_predict(preprocessed)

#taking the results according to the function
scores = clf.negative_outlier_factor_

#choosing a threshold value
threshold = np.sort(scores)[3]

#spotting the values that are out of threshold
outlier_tf = scores > threshold

#show values that are out of threshold
#preprocessed[scores < threshold].head

#removing the spotted values
preprocessed = preprocessed[scores > threshold]


#Train Test Splitting
X,y = divideDependent(preprocessed)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Check if 0 and 1 values are spread evenly
y_train["failure"].value_counts()
y_test["failure"].value_counts()
#We can see that values equal

############# Question3
############################################## Logistic Regression
#Train the model
loj = LogisticRegression(solver="liblinear")
loj_model = loj.fit(x_train,y_train)
#Test the model
y_pred = loj_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(loj_model, x_test, y_test, cv=10).mean())

############################################## Naïve Bayes
#Train the model
nb = GaussianNB()
nb_model = nb.fit(x_train,y_train)
#Test the model
y_pred = nb_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(nb_model, x_test, y_test, cv=10).mean())

############################################## KNN
#Train the model
knn = KNeighborsClassifier()
knn_model = knn.fit(x_train,y_train)
#Test the model
y_pred = knn_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(knn_model, x_test, y_test, cv=10).mean())

############################################## Random Forest
#Train the model
rf = KNeighborsClassifier()
rf_model = rf.fit(x_train,y_train)
#Test the model
y_pred = rf_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(rf_model, x_test, y_test, cv=10).mean())

############################################## SV Classifier
#Train the model
sv = SVC(kernel = "linear")
sv_model = sv.fit(x_train,y_train)
#Test the model
y_pred = sv_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(sv_model, x_test, y_test, cv=10).mean())

############################################## Decision Tree Classifier
#Train the model
dc = DecisionTreeClassifier()
dc_model = dc.fit(x_train,y_train)
#Test the model
y_pred = dc_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(dc_model, x_test, y_test, cv=10).mean())

############################################## AdaBoost Classifier
#Train the model
ab = AdaBoostClassifier()
ab_model = ab.fit(x_train,y_train)
#Test the model
y_pred = ab_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(ab_model, x_test, y_test, cv=10).mean())

############################################## Multi-Layer Perceptron Classifier
#Train the model
mlp = MLPClassifier()
mlp_model = mlp.fit(x_train,y_train)
#Test the model
y_pred = mlp_model.predict(x_test)
accuracy_score(y_test, y_pred)

#Save the Cross validation score
crossScores.append(cross_val_score(mlp_model, x_test, y_test, cv=10).mean())


############# Question 4
#Plotting the Cross Validation results
names = ["Logistic Regression","Naïve Bayes","KNN","Random Forest","SV Classifier","Decision Tree Classifier","AdaBoost Classifier","Multi-Layer Perceptron Classifier"]
dataFrame = pd.DataFrame(columns= ["Models","Cross Scores"])

for i in range(8):
    dff = pd.DataFrame([[names[i],crossScores[i]*100]], columns=["Models","Cross Scores"])
    dataFrame = dataFrame.append(dff)

sns.barplot(x='Cross Scores',y='Models', data=dataFrame, color="b")
plt.xlabel=('Accuracy %')
plt.ylabel=('Cross Score values')


############# Question 5
#We can see that the best resulting algorithm is Multi-Layer perceptron Classifier
#Find the best hyperparameter for the best algorithm
#?mlp_model
hiddenLayerParam = {"hidden_layer_sizes": [100,150,200,250,300,350,400,450,500,550]}
alphaParam = {"alpha": [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.0010]}
learnRateParam = {"learning_rate_init": [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.010]}
HparamAccuracy = []
bestHparam = []

#Apply the first Hyperparameter
mlp = MLPClassifier()
mlp_cv = GridSearchCV(mlp, hiddenLayerParam, cv=10)
mlp_cv.fit(x_train, y_train)
bestHparam.append(mlp_cv.best_params_)

#Test the hyperparameter accuracy and save it
mlp_paramed1 = MLPClassifier(hidden_layer_sizes = bestHparam[0]["hidden_layer_sizes"])
mlp_paramed1.fit(x_train, y_train)

y_pred = mlp_paramed1.predict(x_test)
HparamAccuracy.append(accuracy_score(y_test, y_pred))

#Apply the second Hyperparameter
mlp = MLPClassifier()
mlp_cv = GridSearchCV(mlp, alphaParam, cv=10)
mlp_cv.fit(x_train, y_train)
bestHparam.append(mlp_cv.best_params_)

#Test the hyperparameter accuracy and save it
mlp_paramed2 = MLPClassifier(alpha = bestHparam[1]["alpha"])
mlp_paramed2.fit(x_train, y_train)

y_pred = mlp_paramed2.predict(x_test)
HparamAccuracy.append(accuracy_score(y_test, y_pred))

#Apply the third Hyperparameter
mlp = MLPClassifier()
mlp_cv = GridSearchCV(mlp, learnRateParam, cv=10)
mlp_cv.fit(x_train, y_train)
bestHparam.append(mlp_cv.best_params_)

#Test the hyperparameter accuracy and save it
mlp_paramed3 = MLPClassifier(learning_rate_init = bestHparam[2]["learning_rate_init"])
mlp_paramed3.fit(x_train, y_train)

y_pred = mlp_paramed3.predict(x_test)
HparamAccuracy.append(accuracy_score(y_test, y_pred))

#print the results
print(bestHparam)
print(HparamAccuracy)

############# Question 6
#Use all 3 best parameters as combination
mlp_paramedComb = MLPClassifier(hidden_layer_sizes = bestHparam[0]["hidden_layer_sizes"], alpha = bestHparam[1]["alpha"], learning_rate_init = bestHparam[2]["learning_rate_init"])
mlp_paramedComb.fit(x_train, y_train)

y_pred = mlp_paramedComb.predict(x_test)
HparamCombAcc = accuracy_score(y_test, y_pred)

print(HparamCombAcc)

############# Question 7
#predict results from random inputs
a1, a2, a3, a4 = [],[],[],[]

for i in range(10):
    a1.append(random.uniform(1, 20))
    a2.append(random.uniform(1, 10))
    a3.append(random.uniform(1, 10))
    a4.append(random.uniform(1, 10))

x_rand = {"a1":a1, "a2":a2, "a3":a3, "a4":a4}
x_rand = pd.DataFrame(x_rand, columns = ["a1","a2","a3","a4"])
y_rand_pred = mlp_paramedComb.predict(x_rand)
#add predicted values to the dataframe
x_rand["results"] = y_rand_pred

sns.lmplot(x="a2", y="a1", hue="failure", data=df)
sns.lmplot(x="a3", y="a1", hue="failure", data=df)
sns.lmplot(x="a4", y="a1", hue="failure", data=df)
sns.lmplot(x="a3", y="a2", hue="failure", data=df)
sns.lmplot(x="a4", y="a2", hue="failure", data=df)
sns.lmplot(x="a4", y="a3", hue="failure", data=df)