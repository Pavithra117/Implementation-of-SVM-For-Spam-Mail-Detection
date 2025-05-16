# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

```
1.Detect file encoding.
2.Load and Inspect the data.
3.Preprocessing the data.
4.Convert text to numerial features.
5.Train the model.
6.Make predictions.
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Pavithra K
RegisterNumber:  212224240112
*/
```
```
import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result=chardet.detect(rawdata.read(100000))
result
{'encoding': 'Windows=1252', 'confidence':0.7270322499829184,'language':''}
import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
![SVM For Spam Mail Detection](sam.png)
![Screenshot 2025-05-16 203818](https://github.com/user-attachments/assets/0738e0e8-1804-4873-90d5-dc9e4b6ecbf5)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
