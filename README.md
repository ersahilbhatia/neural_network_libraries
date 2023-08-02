When choosing choice of neural network, there are various libraries you can consider. 
This document provides an overview of different options.

[[_TOC_]]

# Libraries

## PyTorch

PyTorch is an open-source deep learning framework developed by Facebook's AI Research lab (FAIR). It is widely used by researchers, developers, and machine learning practitioners for building and training artificial neural networks.

Merits

- GPU Support: Pytorch can use tensors which can run on top of GPU making computation very fast. Torch library convert numpy array to tensor.
- Neural Network Module: PyTorch provides a module called torch.nn that offers pre-defined layers, loss functions, and optimization algorithms for building and training neural networks efficiently.
- Support for Distributed Computing: PyTorch provides tools for distributed training, allowing users to scale their deep learning workloads across multiple GPUs and machines seamlessly.
- Dynamic Computational Graph: PyTorch utilizes a dynamic computational graph, which means the graph is built on-the-fly as operations are executed. This dynamic nature offers flexibility and ease of debugging, making it easier to work with complex models and varying input sizes.
- TorchScript and ONNX Support: PyTorch's support for TorchScript and ONNX (Open Neural Network Exchange) enables users to convert PyTorch models to other formats for inference and deployment on various platforms and devices.

Like ONNX, TorchScript allow PyTorch model run in other environment like C++, Java, and JavaScript. 

Sample code for implementation of a neural network using PyTorch (CPU)

```
import torch.nn.functional as nnf
import pandas as pd
import seaborn as sns
df=pd.read_csv('diabetes.csv')
df.head()

X=df.drop('Outcome',axis=1).values### independent features
y=df['Outcome'].values###dependent features

# Split into train/test
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#### Libraries From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

##### Creating Tensors
X_train=torch.FloatTensor(X_train)
X_test=torch.FloatTensor(X_test)
y_train=torch.LongTensor(y_train)
y_test=torch.LongTensor(y_test)

df.shape

#### Creating Modelwith Pytorch
class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x

####instantiate my ANN_model
torch.manual_seed(20)
model=ANN_Model()

model.parameters

###Backward Propogation-- Define the loss_function,define the optimizer
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

epochs=500
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss.item())
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### plot the loss function
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')

#### Method to find the probability of 2 classes for y_pred
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        y_probability = torch.sigmoid(y_pred)
        print(y_probability)

#### Prediction In X_test data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)        
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
cm

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

#### Save the model
torch.save(model,'model.pt')

#### Save And Load the model
model=torch.load('model.pt')

### Predcition of new data point
lst1=[6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]
new_data=torch.tensor(lst1)

#### Predict new data using Pytorch
with torch.no_grad():
    new_data_output = model(new_data)
    print(new_data_output)
    print(new_data_output.argmax().item())

#### find the probability of 2 classes for y_pred
y_probability = torch.sigmoid(new_data_output)
print(y_probability)
```

Sample code for implementation of a neural network using PyTorch (GPU)

```
#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
#### Libraries From Pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

# Check if cuda is available
torch.cuda.is_available()

# get device id of GPU
torch.cuda.current_device()

# get device name of GPU
torch.cuda.get_device_name(0)

# How much memory is cached 
torch.cuda.memory_cached()

# check how much memory allocated in GPU
torch.cuda.memory_allocated()

#cuda() function allocate variable to GPU
var1=torch.FloatTensor([1.0,2.0,3.0]).cuda()

var1

var1.device

df=pd.read_csv('diabetes.csv')
df.head()

df.head()

X=df.drop('Outcome',axis=1).values### independent features
y=df['Outcome'].values###dependent features

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

##### Creating Tensors
X_train=torch.FloatTensor(X_train).cuda()
X_test=torch.FloatTensor(X_test).cuda()
y_train=torch.LongTensor(y_train).cuda()

# X_train is stored in GPU
X_train.device

#### Creating Model with Pytorch
class ANN_Model(nn.Module):
    def __init__(self,input_features=8,hidden1=20,hidden2=20,out_features=2):
        super().__init__()
        self.f_connected1=nn.Linear(input_features,hidden1)
        self.f_connected2=nn.Linear(hidden1,hidden2)
        self.out=nn.Linear(hidden2,out_features)
    def forward(self,x):
        x=F.relu(self.f_connected1(x))
        x=F.relu(self.f_connected2(x))
        x=self.out(x)
        return x

####instantiate my ANN_model
torch.manual_seed(20)
model=ANN_Model()

model.parameters

for i in model.parameters():
    print(i.is_cuda)

# By using cuda() function, model will run in GPU
model=model.cuda()

### Backward Propogation-- Define the loss_function,define the optimizer
loss_function=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)

import time
start_time=time.time()
epochs=10000
final_losses=[]
for i in range(epochs):
    i=i+1
    y_pred=model.forward(X_train)
    loss=loss_function(y_pred,y_train)
    final_losses.append(loss.item())
    if i%10==1:
        print("Epoch number: {} and the loss : {}".format(i,loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(time.time()-start_time)

### plot the loss function
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(range(epochs),final_losses)
plt.ylabel('Loss')
plt.xlabel('Epoch')

#### Method to find the probability of 2 classes for y_pred
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        y_probability = torch.sigmoid(y_pred)
        print(y_probability)

#### Prediction In X_test data
predictions=[]
with torch.no_grad():
    for i,data in enumerate(X_test):
        y_pred=model(data)
        predictions.append(y_pred.argmax().item())
        print(y_pred.argmax().item())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,predictions)
cm

plt.figure(figsize=(10,6))
sns.heatmap(cm,annot=True)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')

#### Save the model
torch.save(model,'model.pt')

#### Save And Load the model
model=torch.load('model.pt')

#### New Data
lst1=[6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]
#new_data=torch.tensor(lst1)
new_data=torch.FloatTensor(lst1).cuda()

#### Predict new data using Pytorch
with torch.no_grad():
    print(model(new_data))
    print(model(new_data).argmax().item())
```

GPU can be used in the following cases.
- Large-scale machine learning.
- Data preprocessing which is computationally very expensive.
- Deep Learning: Deep learning models, such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs).
- Image and video processing.
- Natural language processing.

## PySpark

- Distributed Computing: PySpark leverages Spark's distributed computing model, allowing it to process data across multiple nodes in a cluster.
- Scalability: PySpark is designed to scale horizontally, which means it can handle increasing amounts of data by adding more nodes to the cluster, allowing it to grow with the data volume and computational needs.
- Library Support: PySpark provides a vast ecosystem of libraries like for machine learning (PySpark MLlib), streaming data processing (PySpark Streaming), and graph processing (GraphFrames).
- Rich Documentation: https://spark.apache.org/
- Databricks support: For distributed Python workloads, Databricks support PySpark.
- PySpark itself use libraries like Keras for implementation of neural network.


## Keras

Keras is an open-source high-level neural networks API written in Python. Initially developed as an independent project, it later became part of the TensorFlow ecosystem as the official high-level API for TensorFlow.

- Simplicity and Ease of Use: Keras provides a simple and user-friendly API that allows developers to create neural network models with minimal code.
- Built-in Support for Popular Architectures: Keras includes pre-defined implementations of various popular deep learning architectures, such as Convolutional Neural Networks (CNNs) for image processing.
- Easy Model Visualization with TensorBoard: Keras provides built-in integration with TensorBoard, a powerful visualization tool, making it convenient to monitor and analyze model training and performance.

Sample code for implementation of a neural network using Keras

```
# Importing Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

# Importing other lib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 1 - Data Preprocessing
# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13]
y = dataset.iloc[:, 13]

y.value_counts()

#Create dummy variables
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X['Gender'],drop_first=True)

## Concatenate the Data Frames

X=pd.concat([X,geography,gender],axis=1)

## Drop Unnecessary columns
X=X.drop(['Geography','Gender'],axis=1)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, activation='relu'))
# Adding the output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train,validation_split=0.33, batch_size = 10, epochs = 10)

# Part 3 - Making the predictions and evaluating the model
# Predicting the Test set results
y_pred = classifier.predict(X_test)

y_pred_proba = classifier.predict_proba(X_test)

y_pred

y_pred = (y_pred > 0.5)

unique, counts = np.unique(y_pred, return_counts=True)
```

## Tensorflow

TensorFlow is an open-source deep learning framework developed by the Google Brain team. TensorFlow provides a flexible and scalable ecosystem for various machine learning tasks, making it suitable for both research and production applications.

- Ease of Use: TensorFlow is designed to be user-friendly, with a high-level API (Keras) that simplifies model creation and training.
- GPU Acceleration: TensorFlow can leverage GPUs to accelerate computation. It also supports distributed computing.
- TensorBoard: TensorFlow comes with TensorBoard, a powerful visualization tool that allows users to visualize training metrics, model graphs, and more, making it easier to monitor and debug models.



## Keras vs Tensorflow

Keras run on top of tensorflow, yet has some differences.

| Keras | Tensorflow |
|--|--|
| High level neural network API | Low level neural network API |
| Require less code | Require more code |
| Less control over model architecture | More control over model architecture |


## SciKeras

The goal of scikeras is to make it possible to use Keras/TensorFlow with sklearn. This is achieved by providing a wrapper around Keras that has an Scikit-Learn interface.
The main advantage of using Scikeras is it offers full compatibility with the Scikit-Learn API, including grid searches, ensembles, transformers, etc.

```
import numpy as np
from sklearn.datasets import make_classification
from tensorflow import keras

from scikeras.wrappers import KerasClassifier

X, y = make_classification(1000, 20, n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

def get_model(hidden_layer_dim, meta):
    # note that meta is a special argument that will be
    # handed a dict containing input metadata
    n_features_in_ = meta["n_features_in_"]
    X_shape_ = meta["X_shape_"]
    n_classes_ = meta["n_classes_"]

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(n_features_in_, input_shape=X_shape_[1:]))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(hidden_layer_dim))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(n_classes_))
    model.add(keras.layers.Activation("softmax"))
    return model

clf = KerasClassifier(
    get_model,
    loss="sparse_categorical_crossentropy",
    hidden_layer_dim=100,
)

clf.fit(X, y)
y_proba = clf.predict_proba(X)

y_proba

# Sigmoid is used for binary classification methods where we only have 2 classes, while SoftMax applies to multiclass problems. 
# In fact, the SoftMax function is an extension of the Sigmoid function.
```

# Comparison

| Approach | Pros | Cons |
|--|--|--|
| PyTorch | Use tensors which run on GPU for fast computation | Could not provide a good probability score for binary classes |
| | Support distributed computing |  |
| | Torch.nn module makes it easier to build neural network model |  |
| Keras | High level neural network API | |
| | Easy to use |  |
| | Use tensorflow and is a part of it |  |
| | Visualization using tensorboard|  |
| Tensorflow | Provide GPU support | Low level neural network API |
| | Visualization using tensorboard|  |
| SciKeras | Same as Keras/Tensorflow |  |
| | Support Scikit-Learn API |  |


# Conclusion

Since SciKeras offers compatibility with the Scikit-Learn API,  it could be implemented in sklearn pipelines. By using SciKeras, the way to classify document as malware or benign is by using predict_proba() which gives probability from 0 to 1. 

Besides SciKeras, Keras can be used which is built on top of tensorflow. Using Keras, predict() method gives the probability. The document can be classified using the following code line.

`y_pred = (y_pred > 0.5)`

PyTorch is another option for neural network choice. It provides support for both CPU and GPU (using tensors). GPU computing is effective for large amount of data. It also supports distributed computing. One downside of PyTorch is the probability of 2 classes may not sum up to 1. For example, the following are the probabilities of a sample data which is obtained by applying sigmoid function to the output of the model.

```
tensor([0.8465, 0.9868])
tensor([0.8949, 0.2657])
tensor([0.9549, 0.5150])
tensor([0.5228, 0.7866])
tensor([0.8979, 0.8269])
tensor([0.9765, 0.1642])
tensor([0.2588, 0.8376])
tensor([0.4270, 0.8791])
tensor([0.4046, 0.3991])
```

PySpark offers MLlib library which supports many regression and classification algorithms, but it does not support neural network.
