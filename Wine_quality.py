import numpy as np
import matplotlib.pyplot as plt  
import pandas as pd
import seaborn as sns
from warnings import filterwarnings
filterwarnings(action='ignore')

wine.describe()

print(wine.isna().sum())
fixed acidity           0
volatile acidity        0
citric acid             0
residual sugar          0
chlorides               0
free sulfur dioxide     0
total sulfur dioxide    0
density                 0
pH                      0
sulphates               0
alcohol                 0
quality                 0
dtype: int64

# Create Classification version of target variable
wine['goodquality'] = [1 if x >= 7 else 0 for x in wine['quality']]# Separate feature variables and target variable
X = wine.drop(['quality','goodquality'], axis = 1)
Y = wine['goodquality']
# See proportion of good vs bad wines
wine['goodquality'].value_counts()
0    1382
1     217
Name: goodquality, dtype: int64

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

from sklearn.ensemble import ExtraTreesClassifier
classifiern = ExtraTreesClassifier()
classifiern.fit(X,Y)
score = classifiern.feature_importances_
print(score)
[0.07715307 0.10251091 0.09614824 0.07297706 0.07049225 0.0649774
 0.07984845 0.0864291  0.06666012 0.11133139 0.17147201]
 
 from sklearn.model_selection import train_test_split
 X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3,random_state=7)
 
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
Accuracy Score: 0.8729166666666667

from sklearn.svm import SVC
model = SVC()
model.fit(X_train,Y_train)
pred_y = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,pred_y))
Accuracy Score: 0.86875

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,Y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred))
Accuracy Score: 0.8645833333333334

from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,Y_train)
y_pred3 = model3.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(Y_test,y_pred3))
Accuracy Score: 0.8333333333333334




