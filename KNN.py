import pandas as pd
import numpy as np
import ast
from sklearn import preprocessing,model_selection
import sys
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets
import seaborn as sns


df = pd.read_excel ('/dataset_env.xlsx')
df.describe()
df.corr()
#df.head(5)
#df.tail(10)
sns.barplot(data=df, x='label',y='co2')
#sns.boxplot(data=df)
#dataset.shape
df.columns

X = np.array(df.drop(["date","id"],1)) #
Y = np.array(df["label"])
Y =df["label"].tolist()


#KNN classifier
from sklearn.neighbors import KNeighborsClassifier

experiments = 100

score = []
for i in range(1,experiments+1):
	# Train,test split
	X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=0.20)

	print("================= RUNNING EXPERIMENT n: "+str(i)+" ================="+"\n")

	# Loss plot
	neigh = KNeighborsClassifier(n_neighbors=1, algorithm='auto' )
	neigh.fit(X_train,y_train)
	print score
	score.append(neigh.score(X_test,y_test))
print np.mean(score)

pred = neigh.predict(X_test)
#confusion_matrix
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, pred)
from sklearn.metrics import classification_report

target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 7']
print(classification_report(y_test, pred, target_names=target_names))


#mse
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test, pred)
