import pickle


with open('datamros1103', 'rb') as file_handler:
    data = pickle.load(file_handler)
    X, Y = data.get('X', []).values, data.get('Y', []).values


import pickle
import numpy as np
import tensorflow as tf
import random as rn

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

# fix random seed for reproducibility
seed = 7
#
# # The below is necessary for starting Numpy generated random numbers
# # in a well-defined initial state.
np.random.seed(seed)
# # The below is necessary for starting core Python generated random numbers
# # in a well-defined state.
rn.seed(seed)

# according to keras documentation, numpy seed should be set before importing keras
# information regarding setup for obtaining reproducible results using Keras during development in the following link
# https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import label_binarize
from keras.utils import to_categorical
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# The below tf.set_random_seed() will make random number generation
# in the TensorFlow backend have a well-defined initial state.
#tf.set_random_seed(seed)
# Y = label_binarize(Y, classes=[0,1])

batch_size = 100
num_classes = 2

optimizer = 'adamax'
from keras import optimizers


number_of_data = X.shape[0]
number_of_train_data = int(.8*number_of_data)
number_of_test_data = number_of_data-number_of_train_data

# load dataset
x_train, x_test = X[:number_of_train_data, :], X[number_of_train_data:, :]
#mean_train_data = np.mean(train_data, axis=0)
#std_train_data = np.std(train_data, axis=0)
#x_train = (train_data - mean_train_data) / std_train_data  # mean variance normalization
#x_test = (test_data - mean_train_data) / std_train_data  # mean variance normalization
y_train, y_test = Y[:number_of_train_data], Y[number_of_train_data:]
#x_test, y_test = sm.fit_resample(X,Y)

# X_df, Xtest, Y_df, ytest = train_test_split(X_df, Y_df, test_size=0.2)
# Xtrain, ytrain = sm.fit_resample(X_df, Y_df)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = y_train.astype('int32')
y_test = y_test.astype('int32')
# y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
def create_model():
    model = Sequential()
    model.add(Dense(20, input_dim=27, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(20, kernel_initializer='normal', activation='sigmoid'))
    #model.add(Dense(10, kernel_initializer='normal', activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer= optimizer, metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
    return model

    # model = Sequential()
    # model.add(Dense(13, input_dim=19, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(6, kernel_initializer='normal', activation='relu'))
    # model.add(Dense(1, kernel_initializer='normal'))
    # # Compile model
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # return model
sm = SMOTE(random_state=2, ratio = 1.0)
x_train_s, y_train_s = sm.fit_resample(x_train, y_train)

model = create_model()
model.fit(x_train_s, y_train_s, epochs=1, batch_size=batch_size, verbose=1)

y_pred = model.predict(x_test)

print(y_pred, ' Predicted Y')
    
def get_data(plot=True):

    yscore_raw = model.predict_proba(x_test)
    yscore = [s[0] for s in yscore_raw]
    fpr, tpr, thresh = roc_curve(y_test, yscore)
    
    #final = [(lambda i: 0 if i <= np.average(thresholds) else 1)(i) for i in y_pred]
    final = [(lambda i: 0 if i <= np.average(thresh) else 1)(i) for i in y_pred]
    print(confusion_matrix(y_test, final))
    
    auc = metrics.roc_auc_score(y_test, yscore)
    
    
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(fpr, tpr, label='%s ROC (area = %0.3f)' % ('MLP', auc))
        # Custom settings for the plot
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('1-Specificity(False Positive Rate)')
        plt.ylabel('Sensitivity(True Positive Rate)')
        plt.title('Receiver Operating Characteristic - MLP')
        plt.legend(loc="lower right")
        plt.savefig('mlp_MOF.png')
        plt.show()   # Display
    return fpr, tpr, thresh, auc

if __name__ == '__main__':
    get_data()