import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import linear_model
from sklearn.linear_model import SGDRegressor
from category_encoders import TargetEncoder
from sklearn.feature_selection import SelectKBest, f_regression
import xgboost
from xgboost import XGBRegressor

def S2(s1):
    if (s1 is not None):
        return str(s1).replace(' ','')[:4]
    else:
        return s1



dataset=pd.read_csv('training.csv')
dataset
dataset.isnull().any()
# dataset['Gender']=dataset['Gender'].fillna('missing')
# dataset['Profession']=dataset['Profession'].fillna('missing')
# dataset['University Degree']=dataset['University Degree'].fillna('missing')
# dataset['Hair Color']=dataset['Hair Color'].fillna('missing')
# dataset.isnull().any()
# dataset['Year of Record']
from sklearn.impute import SimpleImputer
simpleimputermedian=SimpleImputer(strategy='median')
dataset['Year of Record']=simpleimputermedian.fit_transform(dataset['Year of Record'].values.reshape(-1,1))
dataset['Age']=simpleimputermedian.fit_transform(dataset['Age'].values.reshape(-1,1))
dataset['Body Height [cm]']=simpleimputermedian.fit_transform(dataset['Body Height [cm]'].values.reshape(-1,1))
datasetnoncateg=dataset.drop(['Instance','Hair Color','Wears Glasses','Hair Color'],axis=1)
datasetnoncateg.Profession = list(datasetnoncateg.Profession.map(S2))
datasetnoncateg['Income in EUR']=datasetnoncateg['Income in EUR'].abs()
datasetnoncateg['Income in EUR']=np.log(datasetnoncateg['Income in EUR'])


#datasetcateg=pd.get_dummies(datasetnoncateg, prefix_sep='_')
#colsToDrop = [col for col in datasetcateg.columns if 'missing' in col]
#datasetcategnonmissing=datasetcateg.drop(colsToDrop,axis=1)



M=pd.read_csv('prediciton.csv')
# M.isnull().any()
# M['Gender']=M['Gender'].fillna('missing')
# M['Profession']=M['Profession'].fillna('missing')
# M['University Degree']=M['University Degree'].fillna('missing')
# M['Hair Color']=M['Hair Color'].fillna('missing')
M['Year of Record']=simpleimputermedian.fit_transform(M['Year of Record'].values.reshape(-1,1))
M['Age']=simpleimputermedian.fit_transform(M['Age'].values.reshape(-1,1))
M['Body Height [cm]']=simpleimputermedian.fit_transform(M['Body Height [cm]'].values.reshape(-1,1))
Mnoncateg=M.drop(['Instance','Hair Color','Wears Glasses','Hair Color','Income'],axis=1)
Mnoncateg.Profession = list(Mnoncateg.Profession.map(S2))
# Mcateg=pd.get_dummies(Mnoncateg, prefix_sep='_')
# colsToDropM = [col for col in Mcateg.columns if 'missing' in col]
# Mcategnonmissing=Mcateg.drop(colsToDropM,axis=1)


# for column in datasetcategnonmissing:
#     if column not in Mcategnonmissing:
#         if column != 'Income in EUR':
#             Mcategnonmissing[column]=0


# for column in Mcategnonmissing:
#     if column not in datasetcategnonmissing:
#         datasetcategnonmissing[column]=0         


#datasetcategnonmissing=datasetcategnonmissing.sort_index(axis=1)
#Mcategnonmissing=Mcategnonmissing.sort_index(axis=1)

#z1 = np.abs(stats.zscore(datasetcategnonmissing))
#z2 = np.abs(stats.zscore(Mcategnonmissing))
#datasetcategnonmissing=datasetcategnonmissing[(z1 < 3).all(axis=1)]
#Mcategnonmissing=Mcategnonmissing[(z2 < 3).all(axis=1)]

#X=datasetcategnonmissing.drop('Income in EUR',axis=1).values
#Y=datasetcategnonmissing['Income in EUR'].values


X=datasetnoncateg.drop('Income in EUR',axis=1).values
Y=datasetnoncateg['Income in EUR'].values
#target encoding
t1 = TargetEncoder()
t1.fit(X, Y)
X = t1.transform(X)

#auto feature selection
K= SelectKBest(f_regression, k=5)
K.fit(X,Y)
X=K.transform(X)


Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=0)
# regressor = BayesianRidge()
#regressor = RandomForestRegressor()
#regressor = AdaBoostRegressor()
#regressor = = linear_model.SGDRegressor(max_iter=1000, tol=1e-3)
#regressor = XGBRegressor()
regressor = GradientBoostingRegressor(n_estimators=1000)

fitResult = regressor.fit(Xtrain, Ytrain)
YPredTest = regressor.predict(Xtest)
#learningTest = pd.DataFrame({'Predicted': YPredTest, 'Actual': Ytest })
np.sqrt(metrics.mean_squared_error(Ytest, YPredTest))


A=Mnoncateg.values
A1=t1.transform(A)
A2 = K.transform(A1)
B=regressor.predict(A2)

df2=pd.DataFrame()
df2['Instance']=M['Instance']
df2['Income']=np.exp(B)

df2.to_csv(r'C:\projects\tcd ml 2019-20 income prediction training (with labels).csv\output9.csv',index=False)
