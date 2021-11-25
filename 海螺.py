# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 10:30:49 2021

@author: asus
"""
import pandas as pd
import numpy as np
data=pd.read_excel('gusuiliuhailuo.xlsx')
introduction=data.iloc[0,:]
content_data=data.iloc[1:,:]

#for i in data.columns:
    #print(type(data[i].values[0]))##查看各列数据类型
    
content_data['group']=[str(i) for i in content_data['group'].values]
content_data['sex']=[str(i) for i in content_data['sex'].values]
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(content_data.iloc[:,2:-3],
                                                    content_data.iloc[:,0], 
                                                    test_size = 0.3, 
                                                    random_state = 42)

# 决策树
from sklearn import tree # 分类
clf = tree.DecisionTreeClassifier(criterion="gini",random_state=42) 
clf.fit(Xtrain,ytrain)
# test_data为预测数据,test_label为验证集的标签
predictions=clf.predict(Xtest)
from sklearn.metrics import confusion_matrix
cf_matrix=confusion_matrix(ytest, predictions)
cf_matrix
#array([[96,  5],
       #[11, 48]], dtype=int64)
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy
#Out[22]: 0.9



# 随机森林

from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(oob_score=True, random_state=173)
rfc.fit(Xtrain,ytrain)
# test_data为预测数据,test_label为验证集的标签
predictions=rfc.predict(Xtest)
# 混淆矩阵
cf_matrix=confusion_matrix(ytest, predictions)
cf_matrix
#array([[100,   1],
      # [  7,  52]], dtype=int64)
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy
#Out[116]: 0.95


from sklearn.model_selection import GridSearchCV
parameters = {
 'max_depth':range(3,14,2),
 'min_samples_split':range(50,201,20),
 'n_estimators': [10,20,30,40,50,60,70],
 'min_samples_leaf':range(10,60,10),
}

GS = GridSearchCV(rfc, parameters, cv=5)
GS.fit(Xtrain,ytrain)

GS.best_params_
# 用调参后的模型做预测
#{'max_depth': 3,'min_samples_leaf': 10, 'min_samples_split': 70, 'n_estimators': 60}
for i in range(20):
    rfc=RandomForestClassifier(oob_score=True, random_state=i,
                           max_depth=3,min_samples_leaf= 10,
                           min_samples_split= 70,n_estimators= 60)
    rfc.fit(Xtrain,ytrain)
    # test_data为预测数据,test_label为验证集的标签
    predictions=rfc.predict(Xtest)
    cf_matrix=confusion_matrix(ytest,predictions)
    accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
    print(accuracy)
#i=4
rfc=RandomForestClassifier(oob_score=True, random_state=3,
                           max_depth=3,min_samples_leaf= 10,
                           min_samples_split= 70,n_estimators= 60)
rfc.fit(Xtrain,ytrain)
# test_data为预测数据,test_label为验证集的标签
predictions=rfc.predict(Xtest)
cf_matrix=confusion_matrix(ytest,predictions)
cf_matrix
#Out[126]: 
#array([[99,  2],
      # [12, 47]], dtype=int64)
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
print(accuracy)






# 逻辑回归
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression() 
lr.fit(Xtrain,ytrain)
# test_data为预测数据,test_label为验证集的标签
predictions1=lr.predict(Xtest)
from sklearn.metrics import confusion_matrix
cf_matrix=confusion_matrix(ytest, predictions1)
cf_matrix
#array([[99,  2],
      # [16, 43]], dtype=int64)
      
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy     
#Out[7]: 0.8875     
      
#knn
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(Xtrain,ytrain)
predict=knn.predict(Xtest)
from sklearn.metrics import confusion_matrix
cf_matrix=confusion_matrix(ytest, predict)
cf_matrix
#array([[98,  3],
      # [18, 41]], dtype=int64)
      
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy  
#Out[8]: 0.86875


# svm
##只适用于i数值型数据，所以把最后年龄列删掉
from sklearn.linear_model import SGDClassifier 
svm = SGDClassifier() 
svm.fit(Xtrain.iloc[:,2:],ytrain)
# test_data为预测数据,test_label为验证集的标签
predictions2=svm.predict(Xtest.iloc[:,2:])
cf_matrix=confusion_matrix(ytest, predictions2)
cf_matrix
#array([[85, 16],
       #[10, 49]], dtype=int64)
      
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy  
#Out[58]: 0.8375


# XGBoost
import xgboost as xgb
xgbc = xgb.XGBClassifier()

#将提示的包含错误数据类型这一列进行转换

Xtrain['sex']=[int(i) for i in Xtrain['sex'].values]
Xtest['sex']=[int(i) for i in Xtest['sex'].values]
ytrain=[int(i) for i in ytrain]
ytest=[int(i) for i in ytest]
xgbc.fit(Xtrain.values,ytrain)
#for i in Xtrain.columns:
    #print(type(Xtrain[i].values[0]))##查看各列数据类型

#for i in Xtest.columns:
    #print(type(Xtest[i].values[0]))##查看各列数据类型
    
predictions3=xgbc.predict(Xtest.values)
cf_matrix=confusion_matrix(ytest, predictions3)
cf_matrix   
#array([[97,  4],
       #[15, 44]], dtype=int64)
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy 
#Out[76]: 0.88125




from sklearn.model_selection import GridSearchCV
parameters = {
 'max_depth':[1,3,5,7],
 'min_child_weight':[1,3,5,7],
 'n_estimators': [5,10,20],
 'learning_rate ': [0.3,0.5,0.7]
}

GS = GridSearchCV(xgbc, parameters, cv=10)
GS.fit(Xtrain.values,ytrain)

# 用调参后的模型做预测
GS.best_params_
#{'learning_rate ': 0.3, 'max_depth': 7,'min_child_weight': 1,'n_estimators': 20}

GS.best_score_


xgbc = xgb.XGBClassifier(learning_rate = 0.3,max_depth=7, min_child_weight=1,n_estimators=20)
xgbc.fit(Xtrain.values,ytrain)
predictions331=xgbc.predict(Xtest.values)
cf_matrix=confusion_matrix(ytest, predictions331)
cf_matrix   
#array([[98,  3],
       #[13, 46]], dtype=int64)
accuracy=(cf_matrix[0,0]+cf_matrix[1,1])/np.sum(cf_matrix)
accuracy 
#Out[101]: 0.9










