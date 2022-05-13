import pandas as pd
import numpy as np
import keras
from keras.models import Sequential,load_model
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras import regularizers
from sklearn import preprocessing,model_selection
import h5py
import sys
from matplotlib import pyplot as plt
import os


def neural_network(input_size,output_size,learning_rate):
	model = Sequential()
	model.add(Dense(units=n_hidden,input_dim=input_size,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(units=n_hidden-5,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(units=n_hidden-10,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(units=n_hidden-15,activation='relu',kernel_regularizer=regularizers.l2(0.01)))
	model.add(Dense(units=output_size,activation='softmax',kernel_regularizer=regularizers.l2(0.01)))
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate),metrics=['accuracy'])
	return model


model_path = 'M.h5'
# Dataset reading
df = pd.read_excel ('/dataset_env.xlsx')
X = np.array(df.drop(["date","id","label"],1)) #
Y = np.array(df["label"])
Y =df["label"].tolist()

Y_one_hot = np.array(pd.get_dummies(df["label"]))
Y_one_hot
classes = len(Y_one_hot[0])
features = len(X[1])
features
classes
learning_rate = 0.001
n_hidden = 25

experiments = 2

# Train,test split
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y_one_hot,test_size=0.15)

print("Train size: ",len(X_train))
print("Test size: ",len(X_test))


# Train
for i in range(1,experiments+1):

	print("================= RUNNING EXPERIMENT n: "+str(i)+" ================="+"\n")
	model_path = "M"+str(i)+".h5"
	model = neural_network(features,classes,learning_rate)
	model.summary()
	history = model.fit(X_train,y_train,shuffle=True,epochs=500,validation_split=0.15,verbose=2,callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0, mode='min'),
				keras.callbacks.ModelCheckpoint(model_path,monitor='val_loss', save_best_only=True, mode='min', verbose=0)])

# Test
scores = []
p = []
for i in range(1,experiments+1):
	model_path = "Model_Positions_Keras/position"+str(i)+".h5"
	print("================= RUNNING TEST n: "+str(i)+" ================="+"\n")
	estimator = load_model(model_path)
	score=estimator.evaluate(X_test,y_test)
	print(score)
	scores.append(score[1]*100.0)
print(score[1]*100.0)

print("Average accuracy: ", np.mean(scores))
