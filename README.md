# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries & load data using pandas, and preview with df.head().

2.Clean data by dropping sl_no and salary, checking for nulls and duplicates.

3.Encode categorical columns (like gender, education streams) using LabelEncoder.

4.Split features and target:

X = all columns except status

y = status (Placed/Not Placed)

5.Train-test split (80/20) and initialize LogisticRegression.

6.Fit the model and make predictions.

7.Evaluate model with accuracy, confusion matrix, and classification report.
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vinothkumar R
RegisterNumber:  212224040361
*/
```

```
import pandas as pd
data= pd.read_csv('/content/Placement_Data.csv')
data.head()
```
![1](https://github.com/user-attachments/assets/55bc8845-d6ad-4133-bc1d-92a7d19e7ba4)

```
data1=data.copy()
data1=data.drop(['sl_no','salary'],axis=1)
data1.head()
```
![2](https://github.com/user-attachments/assets/f4d6237b-0d94-4992-aef1-508d1705fabe)

```
data1.isnull().sum()
```
![3](https://github.com/user-attachments/assets/1141e501-1d6b-4c19-b0a8-8d52ae3ce30d)

```
data1.duplicated().sum()
```
![4](https://github.com/user-attachments/assets/4d6381ad-0985-4be4-be26-9b7611e0998f)

```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1
```
![5](https://github.com/user-attachments/assets/c893e654-2de6-47b4-9593-07ce0db708a0)

```
x=data1.iloc[:, : -1]
x
```
![6](https://github.com/user-attachments/assets/abf2319a-d52f-44aa-a651-9d4e33565cab)

```
y=data1["status"]
y
```
![7](https://github.com/user-attachments/assets/48747b68-2781-48f0-9134-c5af6e8265c8)

```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(solver="liblinear")
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
```
```
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
confusion=confusion_matrix(y_test,y_pred)
cr=classification_report(y_test,y_pred)
print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)
```
![8](https://github.com/user-attachments/assets/dd613f5c-4bd9-4789-a26b-7c07df3b572b)
```
print(model.predict([[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]))
```
![9](https://github.com/user-attachments/assets/426e84c2-5a1e-4e89-bc93-933fecb2a1b7)

## Output:
![the Logistic Regression Model to Predict the Placement Status of Student](sam.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
