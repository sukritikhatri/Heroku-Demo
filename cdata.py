# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv('C:/Users/SUKRITI KHATRI/.spyder-py3/python_covid.csv')
data.head(8)

data.dtypes

x = data.iloc[:, 1:11]
y = data.iloc[:, -1]
 

x = data[['Age>60','Previous Chronic Disease','Fever','Dry Cough','Fatigue','Mucus in Throat','Breathing Problem','Joint/Muscle Pain','Sore Throat','Headache','Shivering']].values
x[0:5]

y = data['Severity'].values
y[0:5]


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
print('Train set:', x_train.shape,  y_train.shape)
print('Test set:', x_test.shape,  y_test.shape)  


Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Tree

Tree.fit(x_train,y_train)
y_hat = Tree.predict(x_test)

print(y_test[0:5])
print(y_hat[0:5])

print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, y_hat))


pickle.dump(Tree, open('cdata.pkl','wb'))

# Loading model to compare the results
covid_model_tree = pickle.load(open('cdata.pkl','rb'))


