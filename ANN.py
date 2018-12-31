# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv("Churn_modelling.csv")
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:,-1].values
df = pd.DataFrame(X)

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features =  [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]
#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state =42)

#Feature Scaling required deep NN owing to high computation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Importing the Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()
#Adding the first hidden layer and input layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu', input_dim = 11))
#Adding the second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
#Adding the final layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))
#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])
#Fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size = 1, epochs = 100)
#predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#Importing the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

""" Predicting the new customer
Geography: France
Credit Score: 600
Gender: Male
Age: 40 years old
Tenure: 3 years
Balance: $60000
Number of Products: 2
Does this customer have a credit card ? Yes
Is this customer an Active Member: Yes
Estimated Salary: $50000 """

new_prediction = classifier.predict(sc.transform(np.array([[0.0,0,600,1,40,3,60000,2,1,1,50000]])))
new_prediction = (new_prediction > 0.5)


#Evaluating Imoroving and tuning the ANN
#Evaluating the Ann with K-fold cross validation
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100, batch_size = 10)
accuracies = cross_val_score(estimator = classifier,X = X_train,y = y_train, cv = 10, n_jobs = 1 )
accuracy = accuracies.mean()
varirance = accuracies.std()    

#Improving the Ann with Dropout
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

#Initializing the ANN
classifier = Sequential()
#Adding the first hidden layer and input layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu', input_dim = 11))
classifier.add(Dropout(rate = 0.1))
#Adding the second hidden layer
classifier.add(Dense(output_dim = 6,init = 'uniform',activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

#Adding the final layer
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

#Fitting the ANN to training set
classifier.fit(X_train,y_train,batch_size = 1, epochs = 100)

#Hyperparameter tunnning the Ann
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(6, input_dim = 11, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'epochs': [100,500],
             'batch_size': [25,32],
             'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_
