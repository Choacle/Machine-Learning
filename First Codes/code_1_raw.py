from types import CodeType
from matplotlib.markers import MarkerStyle
from numpy.core.fromnumeric import ravel, shape, size
from pandas.core.frame import DataFrame
from seaborn import colors
import statistics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn.utils import saturate
from sklearn import preprocessing, utils
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn import neighbors,tree
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
###################################################################################

# To Create All of Regression Models
def Multi_RegRession(X_train:np.ndarray,y_train:np.ndarray,X_test:np.ndarray,y_test:np.ndarray,type:str):
    if type == "l":
        Regression = LinearRegression().fit(X_train,y_train)
    elif type == "R":
        Regression = Ridge().fit(X_train,y_train)
    elif type == "L":
        Regression = Lasso().fit(X_train,y_train)
    elif type == "E":
        Regression = ElasticNet().fit(X_train,y_train)
    elif type == "K":
        Regression = neighbors.KNeighborsClassifier().fit(X_train,ravel(y_train))
    elif type == "D":
        Regression = tree.DecisionTreeClassifier().fit(X_train,y_train)
    elif type == "RF":
        Regression = RandomForestClassifier().fit(X_train,ravel(y_train))
    elif type == "S":
        Regression = SVR().fit(X_train,ravel(y_train))
    else: print(f"{type} not found in Regression typies")
    test_pred = Regression.predict(X_test)
    train_pred = Regression.predict(X_train)
    r2 = r2_score(y_test,test_pred)
    Mse = mean_squared_error(y_test,test_pred)

    return Regression, test_pred, train_pred, r2, Mse

# To 10-Fold Cross calculate
def FoldCross(X:np.ndarray, Y:np.ndarray):
    regname = ["Linear", "Ridge", "Lasso", "ElasticNet", "K-NN", "Decision Tree", "Random Forest", "SVR"]
    kf = []
    regressionCode = ["l","R","L","E","K","D","RF","S"]
    skf = StratifiedKFold(n_splits=10, shuffle=True)
    for name in regressionCode:
        R2 = []
        MSE = []
        i=1
        for train,test in skf.split(X,Y):
            R = Multi_RegRession(X[train],Y[train],X[test],Y[test],name)
            R2.append(R[3])
            MSE.append(R[4])
            i+=1
        R2 = statistics.mean(R2)
        MSE = statistics.mean(MSE)
        kf.append([R2,MSE])
    kf = pd.DataFrame(data=kf, index=regname,columns=["R2","MSE"])


    return kf

# To Draw Residual Analysis
def resudual_analysis(y_test:np.ndarray,p_ytest:np.ndarray,y_train:np.ndarray, p_ytrain:np.ndarray, name:str):
    y_test = y_test.reshape(1,-1)
    p_ytest = p_ytest.reshape(1,-1)
    y_train = y_train.reshape(1,-1)
    p_ytrain = p_ytrain.reshape(1,-1)
    plt.figure(figsize=(9,7))
    plt.scatter(p_ytest, (p_ytest - y_test), c='orange', marker='*', s=63, edgecolors="black", label='Test data')
    plt.scatter(p_ytrain, (p_ytrain-y_train), c='m', marker='o', s=63, edgecolors="black", label='Training data',alpha=0.5)
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Residual Analysis of '+ name )
    plt.legend(loc='upper right')
    plt.hlines(y=0, xmin=np.min(y_test)-1, xmax=np.max(y_test)+1, lw=2, color='red')
    plt.grid(True)
    plt.xlim([np.min(p_ytest)-3, np.max(p_ytest)+3])
    plt.show()

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  QUESTİON _ 1_2 ///////////////////////
# READING THE DATA FROM FILE 
df = pd.read_csv("forestfires.csv")
X_columns = ["X","Y","month","day","DMC","DC","ISI","temp","RH","wind","rain","area"] #--> Independent variables
y_column = ["FFMC"] #--> Dependent variable (Target) 
df_x = df[X_columns].values
df_y = df[y_column].values
df_y = df_y.astype('int')

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  QUESTİON _ 3 ///////////////////////
X = df.drop(y_column, axis=1)
X.hist(bins=10,figsize=(16,9),grid=False)

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  QUESTİON _ 4_5 ///////////////////////
kf = FoldCross(df_x,df_y)

labels = kf.index
R2 = kf.R2
MSE = kf.MSE
x = np.arange(len(labels))
sns.set_style('darkgrid')
fig, ax = plt.subplots()
width = 0.35
B1 = ax.bar(x - width/2,R2, width, label='R2',edgecolor="k" )
B2 = ax.bar(x + width/2, MSE, width, label='MSE', edgecolor="k")
ax.set_ylabel('Scores')
ax.set_title('Score Comparisons of Regression Algorithms')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
fig.tight_layout()
plt.show()

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\  QUESTİON _ 6 ///////////////////////
X_train, X_test, Y_train, Y_test = train_test_split(df_x,df_y,test_size=0.2, random_state=99)
regressionCode = ["l","R","L","E","K","D","RF","S"]
regname = ["Linear", "Ridge", "Lasso", "ElasticNet", "K-NN", "Decision Tree", "Random Forest", "SVR"]
i=0
for name in regressionCode:
    p_ytest = Multi_RegRession(X_train, Y_train,X_test,Y_test,name)[1]
    p_ytrain = Multi_RegRession(X_train, Y_train,X_test,Y_test,name)[2]
    resudual_analysis(Y_test,p_ytest,Y_train,p_ytrain,regname[i])
    i+=1
