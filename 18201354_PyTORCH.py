"""
   ---------------------------------------------------------------------------------------------------------
    PYTHON PROJECT 2B
                      Bank Customer Churn Modelling using PyTorch And KERAS.
                                                                                         submitted by,
                                                                                         Justin Joseph.
                                                                                         18201354
   ---------------------------------------------------------------------------------------------------------
   ---------------------------------------------------------------------------------------------------------
    Import the necessary packages.
   ---------------------------------------------------------------------------------------------------------
"""
import warnings
import numpy as np
import pandas as pd
import plotly as pl
import sklearn.preprocessing as skp
import sklearn.model_selection as skm
import sklearn.metrics as skmm
import torch
from torch.autograd import Variable
import torch.utils.data as data_utils
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import keras.layers as kl
import keras.models as km
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.offline as po
import plotly.figure_factory as ff

warnings.filterwarnings("ignore")
"""
   ---------------------------------------------------------------------------------------------------------
    Function definitions.
   ---------------------------------------------------------------------------------------------------------
"""
def encode(a):
    encoder = skp.LabelEncoder()
    a = encoder.fit_transform(a)
    return a
"""
   ---------------------------------------------------------------------------------------------------------
    Main Paragraph.
   ---------------------------------------------------------------------------------------------------------
"""
# Importing the dataset
data=pd.read_csv('C:/Users/HP/Downloads/Study/Python/Assignment/Project 2/Churn_Modelling.csv')
X=data.iloc[:,3:13].values # Removing the columns that are not necessary
y=data.iloc[:,13].values # Target variable

# Pie Chart of churn rate using Plotly
total_churn=len(y)
exited_churn=y.sum()
non_exited_churn=total_churn-exited_churn
labels=['Exited','Non-Exited']
values=[exited_churn,non_exited_churn]
trace=go.Pie(labels=labels,values=values)
fig=go.Figure(data=[trace])
fig.layout.title='Churn Rate'
po.plot(fig,filename='Churn Rate')

# Change both the categorical columns to numerical form
X[:,1]=encode(X[:,1]) # For column Geography
X[:,2]=encode(X[:,2]) # For column Gender
onehotencoder=skp.OneHotEncoder(categorical_features=[1])
X=onehotencoder.fit_transform(X).toarray()
X=X[:,1:]

# Split the data into train (80%) and test (20%)
X_train,X_test,y_train,y_test=skm.train_test_split(X,y,test_size=0.2,random_state=0)

# Scale the data
scaling=skp.StandardScaler()
X_train=scaling.fit_transform(X_train)
X_test=scaling.fit_transform(X_test)
"""
   ##########################################################################################################
    Using PyTorch.
   ##########################################################################################################
"""
# Convert the data into Tensor format
train = data_utils.TensorDataset(torch.from_numpy(X_train).float(),
                                 torch.from_numpy(y_train).float())
test_set = torch.from_numpy(X_test).float()
test_valid = torch.from_numpy(y_test).float()
"""
   ---------------------------------------------------------------------------------------------------------
    Model Creation and defining the parameters
   ---------------------------------------------------------------------------------------------------------
"""
model = torch.nn.Sequential()

module = torch.nn.Linear(11,6)
init.xavier_normal(module.weight)
model.add_module("relu 1", module)

module = torch.nn.Linear(6,6)
init.xavier_normal(module.weight)
model.add_module("relu 2", module)

module = torch.nn.Linear(6,1)
init.xavier_normal(module.weight)
model.add_module("linear 3", module)

model.add_module("sig",torch.nn.Sigmoid())
model.add_module("relu",torch.nn.ReLU())

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)
epochs = 50
dataloader = data_utils.DataLoader(train, batch_size=128, shuffle=False)
history = {"loss": [], "accuracy": [], "loss_val": [], "accuracy_val": []}
"""
   ---------------------------------------------------------------------------------------------------------
    Run the model
   ---------------------------------------------------------------------------------------------------------
"""
for epoch in range(epochs):
    loss = None
    for idx, (batch, target) in enumerate(dataloader):
        y_pred = model(Variable(batch))
        loss = loss_fn(y_pred, Variable(target.float()).reshape(len(Variable(target.float())),1))
        prediction = [1 if x > 0.5 else 0 for x in y_pred.data.numpy()]
        correct = (prediction == target.numpy()).sum()
        y_val_pred = model(Variable(test_set))
        loss_val = loss_fn(y_val_pred, Variable(test_valid.float()).reshape(len(Variable(test_valid.float())),1))
        prediction_val = [1 if x > 0.5 else 0 for x in y_val_pred.data.numpy()]
        correct_val = (prediction_val == test_valid.numpy()).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    history["loss"].append(loss.data[0])
    history["accuracy"].append(100 * correct / len(prediction))
    history["loss_val"].append(loss_val.data[0])
    history["accuracy_val"].append(100 * correct_val / len(prediction_val))
    print("Loss, accuracy, val loss, val acc at epoch", epoch + 1, history["loss"][-1],
          history["accuracy"][-1], history["loss_val"][-1], history["accuracy_val"][-1])

# Confusion Matrix
confusionmatrix1=skmm.confusion_matrix(y_test,prediction_val)
print(confusionmatrix1)
"""
   ---------------------------------------------------------------------------------------------------------
    Plots using Plotly.
   ---------------------------------------------------------------------------------------------------------
"""
# Confusion Matrix plot
x=['False (A)','True (A)']
y=['False (P)','True (P)']
z=confusionmatrix1
fig=ff.create_annotated_heatmap(z,x=x,y=y,annotation_text=z,colorscale='Viridis')
fig.layout.title='Confusion Matrix - PyTorch'
po.plot(fig,filename='Confusion Matrix PyTorch.html')

# Model Accuracy
Train=go.Scatter(x=list(range(epochs)),y=history['accuracy'],name='Train')
Test=go.Scatter(x=list(range(epochs)),y=history['accuracy_val'],name='Test')
data=[Train,Test]
layout =go.Layout(title= 'Model Accuracy - PyTorch',xaxis= {'title':'Epochs'},yaxis={'title':'Accuracy'})
fig=go.Figure(data=data,layout=layout)
po.plot(fig,filename='Model Accuracy PyTorch.html')

# Model Loss
Train=go.Scatter(x=list(range(epochs)),y=history['loss'],name='Train')
Test=go.Scatter(x=list(range(epochs)),y=history['loss_val'],name='Test')
data=[Train,Test]
layout =go.Layout(title= 'Model Loss - PyTorch',xaxis= {'title':'Epochs'},yaxis={'title':'Loss'})
fig=go.Figure(data=data,layout=layout)
po.plot(fig,filename='Model Loss PyTorch.html')
"""
   ##########################################################################################################
    Using KERAS.
   ##########################################################################################################
"""
# Design the Neural Network
classifier=km.Sequential()
classifier.add(kl.Dense(output_dim=6,init='uniform',activation='relu',input_dim=11))
classifier.add(kl.Dense(output_dim=6,init='uniform',activation='relu'))
classifier.add(kl.Dense(output_dim=1,init='uniform',activation='sigmoid'))
classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

# Train the Neural Network
history1=classifier.fit(X_train,y_train,batch_size=128,nb_epoch=50,validation_data=(X_test,y_test))

# Predict using the test data
yhat=classifier.predict(X_test)
yhat=(yhat>0.5)

# Confusion Matrix
confusionmatrix2=skmm.confusion_matrix(y_test,yhat)
print(confusionmatrix2)
"""
   ---------------------------------------------------------------------------------------------------------
    Plots using Plotly.
   ---------------------------------------------------------------------------------------------------------
"""
# Confusion Matrix plot
x=['False (A)','True (A)']
y=['False (P)','True (P)']
z=confusionmatrix2
fig=ff.create_annotated_heatmap(z,x=x,y=y,annotation_text=z,colorscale='Viridis')
fig.layout.title='Confusion Matrix - KERAS'
po.plot(fig,filename='Confusion Matrix KERAS.html')

# Model Accuracy
Train=go.Scatter(x=list(range(epochs)),y=history1.history['acc'],name='Train')
Test=go.Scatter(x=list(range(epochs)),y=history1.history['val_acc'],name='Test')
data=[Train,Test]
layout =go.Layout(title= 'Model Accuracy - KERAS',xaxis= {'title':'Epochs'},yaxis={'title':'Accuracy'})
fig=go.Figure(data=data,layout=layout)
po.plot(fig,filename='Model Accuracy KERAS.html')

# Model Loss
Train=go.Scatter(x=list(range(epochs)),y=history1.history['loss'],name='Train')
Test=go.Scatter(x=list(range(epochs)),y=history1.history['val_loss'],name='Test')
data=[Train,Test]
layout =go.Layout(title= 'Model Loss - KERAS',xaxis= {'title':'Epochs'},yaxis={'title':'Loss'})
fig=go.Figure(data=data,layout=layout)
po.plot(fig,filename='Model Loss KERAS.html')